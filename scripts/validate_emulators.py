#!/usr/bin/env python
"""Validate HiFast emulators on their reconstructed held-out test rows.

The training pipeline joins the thin, std, and ext datasets before making one
global train/test split. This script reconstructs that split from the metadata
stored in each emulator joblib, but loads and evaluates only one source FITS
file at a time. Region reports are cumulative by source: std contains thin and
std test rows, while ext contains test rows from all three sources.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import scipy.interpolate as interp
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))

from hi_fast import HiFast  # noqa: E402
from hi_fast.io import FitsFile  # noqa: E402


REGIONS = ('thin', 'std', 'ext')
ABSOLUTE_DIFFERENCE = {'cl_TE_lensed', 'cl_Tp_lensed'}
THRESHOLDS_PERCENT = (0.01, 0.05, 0.1, 1.0)
WORST_COUNT = 3


def _pyplot():
    """Import the non-interactive plotting backend only when requested."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _fits_stem(path):
    """Return a FITS filename without its optional compression suffix."""
    name = Path(path).name
    if name.endswith('.fits.gz'):
        return name[:-8]
    if name.endswith('.fits'):
        return name[:-5]
    return Path(name).stem


def _region_from_path(path):
    """Infer the named sampling region from a dataset filename."""
    suffix = _fits_stem(path).rsplit('_', 1)[-1]
    if suffix not in REGIONS:
        raise ValueError(
            'Cannot infer thin/std/ext region from {}'.format(path))
    return suffix


def _find_dataset(data_root, configured_path):
    """Resolve a training path against a portable local data root."""
    data_root = Path(data_root).expanduser().resolve()
    configured = Path(configured_path)
    candidates = [
        data_root / configured.name,
        data_root / 'sample' / configured.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(data_root.rglob(configured.name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            'Could not find {} below {}'.format(configured.name, data_root))
    raise ValueError(
        'Multiple files named {} found below {}: {}'.format(
            configured.name, data_root, matches))


def _training_configuration(emulator_root, model, spectrum):
    """Read dataset order and split settings from one exported joblib."""
    path = Path(emulator_root) / model / '{}.joblib'.format(spectrum)
    content = joblib.load(path)
    try:
        datasets = content['training_params']['datasets']
        configured_paths = datasets['paths']
        fraction_train = datasets['frac_train']
        seed = datasets['train_test_random_seed']
        remove_non_finite = datasets['remove_non_finite']
    except KeyError as error:
        raise KeyError(
            '{} lacks training split metadata: {}'.format(path, error))
    if configured_paths is None:
        raise ValueError(
            '{} was not trained from combined dataset paths'.format(path))
    return {
        'paths': configured_paths,
        'fraction_train': fraction_train,
        'random_seed': seed,
        'remove_non_finite': remove_non_finite,
    }


def _ordered_region_paths(data_root, configured_paths):
    """Resolve and return the configured files in thin/std/ext order."""
    resolved = []
    configured_regions = []
    for configured_path in configured_paths:
        region = _region_from_path(configured_path)
        if region in configured_regions:
            raise ValueError('Multiple {} datasets configured'.format(region))
        configured_regions.append(region)
        resolved.append(_find_dataset(data_root, configured_path))
    if tuple(configured_regions) != REGIONS:
        raise ValueError(
            'Training dataset order is {}; expected {}. The original order '
            'must be preserved to reconstruct the global split.'.format(
                tuple(configured_regions), REGIONS))
    return resolved


def _finite_rows(path, spectrum):
    """Return the finite-y row mask used before the training split."""
    values = np.asarray(FitsFile(str(path)).get_data(spectrum))
    return np.all(np.isfinite(values), axis=1)


def _reconstruct_test_mask(paths, spectrum, fraction_train, seed,
                           remove_non_finite):
    """Reconstruct the global post-merge test membership without merging."""
    counts = []
    for path in paths:
        finite = _finite_rows(path, spectrum)
        counts.append(int(np.count_nonzero(finite))
                      if remove_non_finite else len(finite))
        del finite

    indices = np.arange(sum(counts))
    _, test_indices = train_test_split(
        indices, train_size=fraction_train, random_state=seed)
    test_mask = np.zeros(len(indices), dtype=bool)
    test_mask[test_indices] = True
    return test_mask, counts


def _read_test_dataset(path, spectrum, global_test_mask, offset,
                       remove_non_finite):
    """Load one source file and retain only its globally held-out rows."""
    start = time.perf_counter()
    fits = FitsFile(str(path))
    x = np.asarray(fits.get_data('x_data'))
    y = np.asarray(fits.get_data(spectrum))
    names = list(fits.get_header(0)['params'])
    finite = np.all(np.isfinite(y), axis=1)
    retained = \
        np.flatnonzero(finite) if remove_non_finite else np.arange(len(y))
    source_test = global_test_mask[offset:offset + len(retained)]
    rows = retained[source_test]

    if spectrum.startswith(('pk_', 'fk_')):
        grid = np.asarray(fits.get_data('k_range_{}'.format(spectrum)))
        z_reference = np.asarray(fits.get_data('z_array'))
    else:
        grid = np.asarray(fits.get_data('ell_range_{}'.format(spectrum)))
        z_reference = None

    return {
        'path': path,
        'x': x[rows],
        'y': y[rows],
        'original_rows': rows,
        'params': names,
        'grid': grid,
        'z_reference': z_reference,
        'reference': np.asarray(
            fits.get_data('ref_{}'.format(spectrum)))[0],
        'load_time': time.perf_counter() - start,
    }


def _row_parameters(cosmo, spectrum, names, row):
    """Build public HiFast parameters and redshift for one stored row."""
    expected = cosmo._params[spectrum]._emu
    if names != expected:
        raise ValueError('Parameters for {} differ: {} != {}'.format(
            spectrum, names, expected))
    params = {name: value for name, value in zip(names, row)
              if name != 'z_pk'}
    for name in cosmo._params[spectrum]._additional:
        params.setdefault(name, cosmo._spectra[spectrum].class_args[name])
    z = row[names.index('z_pk')] if 'z_pk' in names else None
    return params, z


def _predict_batch(cosmo, spectrum, grid, params, redshifts, from_pk):
    """Evaluate one public HiFast batch."""
    if spectrum.startswith('pk_'):
        return cosmo.get_pk(
            grid, redshifts, params,
            name=spectrum.removeprefix('pk_'), paired=True)
    if spectrum.startswith('fk_'):
        return cosmo.get_fk(
            grid, redshifts, params,
            name=spectrum.removeprefix('fk_'), get_from_pk=from_pk,
            paired=True)
    name = spectrum.removeprefix('cl_').removesuffix('_lensed')
    return cosmo.get_cell(grid, params, name=name)


def _batch_errors(spectrum, prediction, data):
    """Return valid per-mode errors, per-sample RMS, and retained rows."""
    valid = np.all(np.isfinite(prediction) & np.isfinite(data), axis=1)
    prediction = prediction[valid]
    data = data[valid]
    if spectrum in ABSOLUTE_DIFFERENCE:
        errors = prediction - data
        kind = 'absolute'
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = prediction / data - 1.0
        finite = np.all(np.isfinite(errors), axis=1)
        valid_indices = np.flatnonzero(valid)
        valid[:] = False
        valid[valid_indices[finite]] = True
        prediction = prediction[finite]
        data = data[finite]
        errors = errors[finite]
        kind = 'relative'
    rms = np.sqrt(np.mean(errors**2, axis=1))
    return prediction, data, errors, rms, valid, kind


def _merge_worst(current, candidates, count=WORST_COUNT):
    """Retain only the largest-RMS records."""
    return sorted(current + candidates,
                  key=lambda item: item['rms'], reverse=True)[:count]


def evaluate_test_rows(cosmo, spectrum, dataset, batch_size, from_pk=False):
    """Evaluate held-out rows in bounded memory and return RMS summaries."""
    start_total = time.perf_counter()
    rms_parts = []
    worst = []
    emulator_time = 0.0
    reference_spline = None
    if dataset['z_reference'] is not None:
        reference_spline = interp.make_splrep(
            dataset['z_reference'], dataset['reference'].T, s=0)

    n_rows = len(dataset['x'])
    for first in range(0, n_rows, batch_size):
        last = min(first + batch_size, n_rows)
        rows = dataset['x'][first:last]
        batch = [_row_parameters(
            cosmo, spectrum, dataset['params'], row) for row in rows]
        params = [item[0] for item in batch]
        redshifts = (np.asarray([item[1] for item in batch])
                     if dataset['z_reference'] is not None else None)

        start = time.perf_counter()
        values = np.asarray(_predict_batch(
            cosmo, spectrum, dataset['grid'], params, redshifts, from_pk))
        emulator_time += time.perf_counter() - start

        reference = dataset['reference']
        if reference_spline is not None:
            reference = np.asarray(reference_spline(redshifts))
            if reference.shape != values.shape:
                reference = reference.T
        prediction = values / reference
        prediction, data, errors, rms, valid, kind = _batch_errors(
            spectrum, prediction, dataset['y'][first:last])
        rms_parts.append(rms)

        valid_rows = dataset['original_rows'][first:last][valid]
        candidates = []
        for index in np.argsort(rms)[-WORST_COUNT:]:
            candidates.append({
                'rms': float(rms[index]),
                'grid': dataset['grid'].copy(),
                'prediction': prediction[index].copy(),
                'data': data[index].copy(),
                'errors': errors[index].copy(),
                'source': dataset['path'].name,
                'row': int(valid_rows[index]),
            })
        worst = _merge_worst(worst, candidates)

    rms = np.concatenate(rms_parts) if rms_parts else np.empty(0)
    total_time = time.perf_counter() - start_total
    timing = {
        'total': total_time,
        'emulator': emulator_time,
        'other': total_time - emulator_time,
    }
    return rms, worst, kind, timing


def _summarize(spectrum, method, region, rms, kind, sources):
    """Build one JSON-serializable cumulative validation record."""
    thresholds = [value / 100.0 for value in THRESHOLDS_PERCENT]
    percent_within = {
        str(label): (float(100 * np.mean(rms <= threshold))
                     if len(rms) else None)
        for label, threshold in zip(THRESHOLDS_PERCENT, thresholds)
    }
    quantiles = np.percentile(rms, [50, 95, 99]) if len(rms) else [None]*3
    return {
        'observable': spectrum,
        'method': method,
        'region': region,
        'metric': '{}_rms'.format(kind),
        'samples_valid': int(len(rms)),
        'sources': list(sources),
        'percent_within': percent_within,
        'rms_median': (float(quantiles[0]) if len(rms) else None),
        'rms_p95': (float(quantiles[1]) if len(rms) else None),
        'rms_p99': (float(quantiles[2]) if len(rms) else None),
        'rms_max': float(np.max(rms)) if len(rms) else None,
    }


def _plot_histogram(rms, spectrum, method, region, kind, save_dir):
    """Plot one cumulative held-out RMS distribution."""
    plt = _pyplot()
    positive = rms[rms > 0]
    floor = np.finfo(float).tiny if not len(positive) else positive.min()/10
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(np.maximum(rms, floor)), bins=30, log=True)
    for threshold in THRESHOLDS_PERCENT:
        ax.axvline(np.log10(threshold/100), color='r', lw=0.7)
    ax.set(
        title='{} — {} — {} cumulative test set'.format(
            spectrum, method, region),
        xlabel=r'$\log_{{10}}$ RMS {} error'.format(kind),
        ylabel='Number of test models')
    fig.tight_layout()
    filename = 'hist_{}_{}_{}.png'.format(spectrum, method, region)
    fig.savefig(save_dir / filename, dpi=150)
    plt.close(fig)


def _plot_worst(worst, spectrum, method, region, kind, save_dir):
    """Plot the worst cumulative held-out predictions."""
    if not worst:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(worst) + 1, 1, figsize=(8, 3.2*(len(worst) + 1)))
    error_scale = 100.0 if kind == 'relative' else 1.0
    for rank, record in enumerate(worst, 1):
        label = 'rank {}, {} row {}'.format(
            rank, record['source'], record['row'])
        axes[0].plot(record['grid'],
                     np.abs(record['errors'])*error_scale,
                     label=label)
        axes[rank].plot(record['grid'], record['data'], label='stored data')
        axes[rank].plot(
            record['grid'], record['prediction'], '--', label='HiFast')
        axes[rank].set_ylabel(label)
        axes[rank].legend()
    axes[0].set_yscale('log')
    axes[0].set_ylabel(
        'relative error [%]' if kind == 'relative' else 'absolute error')
    axes[0].legend()
    axes[-1].set_xlabel(
        'k [h/Mpc]' if spectrum.startswith(('pk_', 'fk_')) else r'$\ell$')
    if spectrum.startswith(('pk_', 'fk_')):
        for axis in axes:
            axis.set_xscale('log')
    fig.suptitle('Worst test models: {} — {} — {}'.format(
        spectrum, method, region))
    fig.tight_layout()
    filename = 'worst_{}_{}_{}.png'.format(spectrum, method, region)
    fig.savefig(save_dir / filename, dpi=150)
    plt.close(fig)


def _validate_spectrum(cosmo, spectrum, paths, split, batch_size,
                       save_dir, make_plots):
    """Validate one observable while loading one source region at a time."""
    test_mask, counts = _reconstruct_test_mask(
        paths, spectrum, split['fraction_train'], split['random_seed'],
        split['remove_non_finite'])
    methods = [('direct', False)]
    if spectrum.startswith('fk_'):
        methods.append(('from_pk', True))
    cumulative = {
        method: {'rms': [], 'worst': [], 'sources': [], 'kind': None}
        for method, _ in methods
    }
    results = []
    offset = 0

    for region, path, count in zip(REGIONS, paths, counts):
        print('\n  Loading {} source: {}'.format(region, path), flush=True)
        dataset = _read_test_dataset(
            path, spectrum, test_mask, offset, split['remove_non_finite'])
        offset += count
        print('    held-out rows: {} | load: {:.3f} s'.format(
            len(dataset['x']), dataset['load_time']))

        for method, from_pk in methods:
            print('    evaluating {}...'.format(method), flush=True)
            rms, worst, kind, timing = evaluate_test_rows(
                cosmo, spectrum, dataset, batch_size, from_pk=from_pk)
            state = cumulative[method]
            state['rms'].append(rms)
            state['worst'] = _merge_worst(state['worst'], worst)
            state['sources'].append(path.name)
            state['kind'] = kind
            cumulative_rms = np.concatenate(state['rms'])
            result = _summarize(
                spectrum, method, region, cumulative_rms, kind,
                state['sources'])
            results.append(result)
            percentages = ', '.join(
                '≤{}%: {}'.format(
                    threshold,
                    ('{:.3f}%'.format(
                        result['percent_within'][str(threshold)])
                     if result['percent_within'][str(threshold)] is not None
                     else 'N/A'))
                for threshold in THRESHOLDS_PERCENT)
            print('      {} cumulative test models | {}'.format(
                len(cumulative_rms), percentages))
            print('      timing: {:.3f} s total ({:.3f} s HiFast, '
                  '{:.3f} s other)'.format(
                      timing['total'], timing['emulator'], timing['other']))
            if make_plots:
                _plot_histogram(
                    cumulative_rms, spectrum, method, region, kind, save_dir)
                _plot_worst(
                    state['worst'], spectrum, method, region, kind, save_dir)

        del dataset
    return results, len(test_mask), int(np.count_nonzero(test_mask))


def main():
    parser = argparse.ArgumentParser(
        description='Validate emulators on reconstructed held-out test rows.')
    parser.add_argument(
        'data_root', help='Folder containing the configured FITS datasets')
    parser.add_argument('--model', '-m', default='lcdm')
    parser.add_argument(
        '--emulator-root', default=str(REPO_ROOT / 'emu'),
        help='Directory containing emulator bundles')
    parser.add_argument('--batch-size', '-b', type=int, default=256)
    parser.add_argument(
        '--spectrum', action='append',
        help='Observable to validate; repeatable. Default: all observables')
    parser.add_argument(
        '--output', help='Validation JSON path. Default: bundle directory')
    parser.add_argument(
        '--plot-dir', default=str(REPO_ROOT / 'validation_plots'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    emulator_root = Path(args.emulator_root).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output else
              emulator_root / args.model / 'validation.json')
    save_dir = Path(args.plot_dir).expanduser().resolve() / args.model
    if not args.no_plots:
        save_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    cosmo = HiFast(args.model, root=str(emulator_root), verbose=True)
    print('Model loading: {:.3f} s'.format(time.perf_counter() - start))
    spectra = args.spectrum or sorted(cosmo._spectra)
    unknown = sorted(set(spectra) - set(cosmo._spectra))
    if unknown:
        parser.error('Unknown observables: {}'.format(unknown))

    if args.spectrum and output.is_file():
        with open(output, encoding='utf-8') as file:
            report = json.load(file)
        if (report.get('schema_version') != 1
                or report.get('model') != args.model):
            raise ValueError(
                'Existing report {} has an incompatible schema or model'
                .format(output))
        selected = set(spectra)
        report['results'] = [
            result for result in report.get('results', [])
            if result.get('observable') not in selected
        ]
        for spectrum in selected:
            report.setdefault('splits', {}).pop(spectrum, None)
    else:
        report = {
            'schema_version': 1,
            'model': args.model,
            'thresholds_percent': list(THRESHOLDS_PERCENT),
            'region_membership': 'cumulative_source_datasets',
            'results': [],
            'splits': {},
        }
    for spectrum in spectra:
        print('\nValidating {}'.format(spectrum), flush=True)
        configuration = _training_configuration(
            emulator_root, args.model, spectrum)
        paths = _ordered_region_paths(
            args.data_root, configuration['paths'])
        results, n_total, n_test = _validate_spectrum(
            cosmo, spectrum, paths, configuration, args.batch_size,
            save_dir, not args.no_plots)
        report['results'].extend(results)
        report['splits'][spectrum] = {
            'type': 'held_out',
            'source': 'reconstructed_from_seed',
            'fraction_train': configuration['fraction_train'],
            'random_seed': configuration['random_seed'],
            'samples_after_filtering': n_total,
            'samples_test': n_test,
            'dataset_order': [path.name for path in paths],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    print('\nWrote {}'.format(output))


if __name__ == '__main__':
    main()
