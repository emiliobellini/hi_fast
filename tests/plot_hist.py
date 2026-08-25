"""Plot HiFast error histograms and its three worst predictions.

Usage: python tests/plot_hist.py path/to/data_folder -m lcdm
"""
import argparse
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from hi_fast import HiFast
from hi_fast.io import FitsFile
plt.switch_backend('Agg')


ABSOLUTE_DIFFERENCE = {'cl_TE_lensed', 'cl_Tp_lensed'}
THRESHOLDS = (0.01, 0.05, 0.1, 1.0)


def find_data_files(path):
    """Return a FITS path or all FITS files below a directory."""
    path = Path(path).expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError('Data path does not exist: {}'.format(path))
    files = sorted(set(path.rglob('*.fits')) | set(path.rglob('*.fits.gz')))
    if not files:
        raise FileNotFoundError('No FITS files found below {}'.format(path))
    return files


def read_dataset(path, spectrum):
    """Read arrays for one spectrum; return None if it is not in the file."""
    start = time.perf_counter()
    fits = FitsFile(str(path))
    try:
        data = {
            'suffix': os.path.splitext(os.path.split(path)[-1])[0],
            'path': path,
            'x': np.asarray(fits.get_data('x_data')),
            'y': np.asarray(fits.get_data(spectrum)),
            'params': list(fits.get_header(0)['params']),
        }
        if spectrum.startswith(('pk_', 'fk_')):
            data['grid'] = np.asarray(
                fits.get_data('k_range_{}'.format(spectrum)))
            data['z'] = np.asarray(fits.get_data('z_array'))
        else:
            data['grid'] = np.asarray(
                fits.get_data('ell_range_{}'.format(spectrum)))
            data['z'] = None
        data['reference'] = np.asarray(
            fits.get_data('ref_{}'.format(spectrum)))[0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    data['load_time'] = time.perf_counter() - start
    return data


def row_parameters(cosmo, spectrum, names, row):
    """Build the HiFast parameters and redshift for a stored sample."""
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


def predict(cosmo, spectrum, dataset, get_from_pk=False):
    """Evaluate HiFast and convert physical output to stored-data units."""
    start_total = time.perf_counter()
    emulator_time = 0.0
    evaluations = 0
    output = np.full(dataset['y'].shape, np.nan, dtype=float)
    name = spectrum.split('_')[1]
    n_samples = len(dataset['x'])
    progress_step = max(n_samples // 10, 1)

    # The reference grid is identical for every sample. Constructing this
    # spline inside the loop is much more expensive than evaluating it and
    # used to dominate the total prediction time for large datasets.
    reference_spline = None
    if dataset['z'] is not None:
        reference_spline = interp.make_splrep(
            dataset['z'], dataset['reference'].T, s=0)

    for index, row in enumerate(dataset['x']):
        if np.any(~np.isfinite(row)):
            continue
        params, z = row_parameters(cosmo, spectrum, dataset['params'], row)
        start_emulator = time.perf_counter()
        if spectrum.startswith('pk_'):
            value = cosmo.get_pk(
                dataset['grid'], z, params, name=name, squeeze=True)
        elif spectrum.startswith('fk_'):
            # get_fk(..., get_from_pk=True) mutates params, hence the copy.
            value = cosmo.get_fk(
                dataset['grid'], z, params.copy(), name=name,
                get_from_pk=get_from_pk, squeeze=True)
        else:
            cl_name = spectrum.removeprefix('cl_').removesuffix('_lensed')
            value = cosmo.get_cell(
                dataset['grid'], params, name=cl_name, squeeze=True)
        emulator_time += time.perf_counter() - start_emulator
        evaluations += 1

        reference = (dataset['reference'] if reference_spline is None
                     else reference_spline(z))
        output[index] = np.asarray(value) / reference
        completed = index + 1
        if completed % progress_step == 0 or completed == n_samples:
            elapsed = time.perf_counter() - start_total
            rate = completed / elapsed
            remaining = (n_samples - completed) / rate if rate else 0.0
            print('  Prediction progress: {:6.1f}% | elapsed {:8.2f} s | '
                  'ETA {:8.2f} s'.format(
                      100 * completed / n_samples, elapsed, remaining),
                  flush=True)

    total_time = time.perf_counter() - start_total
    timing = {
        'total': total_time,
        'emulator': emulator_time,
        'other': total_time - emulator_time,
        'evaluations': evaluations,
    }
    return output, timing


def calculate_errors(spectrum, prediction, data):
    """Filter invalid samples and calculate pointwise and per-sample errors."""
    valid = np.all(np.isfinite(prediction) & np.isfinite(data), axis=1)
    prediction, data = prediction[valid], data[valid]
    if spectrum in ABSOLUTE_DIFFERENCE:
        errors, kind = prediction - data, 'absolute'
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = prediction / data - 1.0
        valid = np.all(np.isfinite(errors), axis=1)
        prediction, data, errors = \
            prediction[valid], data[valid], errors[valid]
        kind = 'relative'
    rms = np.sqrt(np.mean(errors**2, axis=1))
    return prediction, data, errors, rms, kind


def print_summary(label, rms, kind):
    values = []
    for threshold in THRESHOLDS:
        percentage = 100 * np.mean(rms > threshold / 100)
        values.append('>{}%: {:.3f}%'.format(threshold, percentage))
    print('{} ({} RMS, {} samples): {}'.format(
        label, kind, len(rms), ', '.join(values)))


def plot_histogram(rms, spectrum, label, kind, save_dir):
    positive = rms[rms > 0]
    floor = np.finfo(float).tiny if not len(positive) else positive.min() / 10
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(np.maximum(rms, floor)), bins=30, log=True)
    for threshold in THRESHOLDS:
        ax.axvline(np.log10(threshold / 100), color='r', lw=0.7)
    ax.set(title='{} — {}'.format(spectrum, label),
           xlabel=r'$\log_{{10}}$ RMS {} error'.format(kind),
           ylabel='Number of samples')
    fig.tight_layout()
    filename = 'hist_{}_{}.png'.format(spectrum, label)
    fig.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)
    return filename


def plot_worst(grid, prediction, data, errors, rms, spectrum, label, kind,
               save_dir, count=3):
    count = min(count, len(rms))
    if not count:
        return None
    worst = np.argsort(rms)[::-1][:count]
    fig, axes = plt.subplots(count + 1, 1, figsize=(8, 3.2*(count + 1)))
    for rank, index in enumerate(worst, 1):
        axes[0].plot(grid, np.abs(errors[index])*100,
                     label='rank {}, sample {}'.format(rank, index))
        axes[rank].plot(grid, data[index], label='stored data')
        axes[rank].plot(grid, prediction[index], '--', label='HiFast')
        axes[rank].set_ylabel('rank {}, sample {}'.format(rank, index))
        axes[rank].legend()
    axes[0].set_yscale('log')
    axes[0].set_ylabel('{} error [%]'.format(kind))
    axes[0].legend()
    axes[-1].set_xlabel('k [h/Mpc]' if spectrum.startswith(('pk_', 'fk_'))
                        else r'$\ell$')
    if spectrum.startswith(('pk_', 'fk_')):
        for ax in axes:
            ax.set_xscale('log')
    fig.suptitle('Worst modes: {} — {}'.format(spectrum, label))
    fig.tight_layout()
    filename = 'worst_modes_{}_{}.png'.format(spectrum, label)
    fig.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)
    return filename


def process(cosmo, spectrum, dataset, save_dir, label, from_pk=False):
    start_process = time.perf_counter()
    print('  Dataset loading: {:.3f} s'.format(dataset['load_time']))
    print('  Starting {} emulator evaluations...'.format(len(dataset['x'])),
          flush=True)
    prediction, prediction_timing = predict(
        cosmo, spectrum, dataset, get_from_pk=from_pk)
    n_evaluations = prediction_timing['evaluations']
    average = (prediction_timing['emulator'] / n_evaluations
               if n_evaluations else float('nan'))
    print('  Prediction total: {:.3f} s'.format(prediction_timing['total']))
    print('    HiFast evaluations: {:.3f} s total, {:.6f} s/sample '
          '({} samples)'.format(
              prediction_timing['emulator'], average, n_evaluations))
    print('    Parameter/reference/output operations: {:.3f} s'.format(
        prediction_timing['other']))

    start = time.perf_counter()
    prediction, data, errors, rms, kind = calculate_errors(
        spectrum, prediction, dataset['y'])
    print('  Error calculation and filtering: {:.3f} s'.format(
        time.perf_counter() - start))
    if not len(rms):
        print('Skipping {} {}: no finite samples'.format(spectrum, label))
        return
    print_summary('{} {}'.format(spectrum, label), rms, kind)
    start = time.perf_counter()
    filename = plot_histogram(rms, spectrum, label, kind, save_dir)
    print('  Histogram: {:.3f} s (saved {})'.format(
        time.perf_counter() - start, filename))
    start = time.perf_counter()
    filename = plot_worst(
        dataset['grid'], prediction, data, errors, rms, spectrum, label,
        kind, save_dir)
    print('  Worst-modes plot: {:.3f} s (saved {})'.format(
        time.perf_counter() - start, filename))
    print('  Complete {} processing: {:.3f} s'.format(
        label, time.perf_counter() - start_process))


def main():
    parser = argparse.ArgumentParser(
        description='Plot emulator error histograms and worst modes.')
    parser.add_argument('data_folder', help='FITS file or folder of FITS data')
    parser.add_argument('--model', '-m', default='lcdm', help='Model to use')
    parser.add_argument('--save-dir', '-s', default='output')
    args = parser.parse_args()

    save_dir = os.path.abspath(os.path.join(args.save_dir, args.model))
    os.makedirs(save_dir, exist_ok=True)
    start = time.perf_counter()
    cosmo = HiFast(args.model, root='emu', timeit=True, verbose=True)
    print('HiFast model loading: {:.3f} s'.format(time.perf_counter() - start))
    start = time.perf_counter()
    paths = find_data_files(args.data_folder)
    print('Dataset discovery: {:.3f} s ({} FITS files)'.format(
        time.perf_counter() - start, len(paths)))

    for spectrum in cosmo._spectra:
        start = time.perf_counter()
        datasets = [data for path in paths
                    if (data := read_dataset(path, spectrum)) is not None]
        print('\nDataset loading/search for {}: {:.3f} s '
              '({} matching files)'.format(
                  spectrum, time.perf_counter() - start, len(datasets)))
        if not datasets:
            print('Skipping {}: no matching dataset'.format(spectrum))
            continue
        for dataset in datasets:
            print('\nProcessing {} from {}'.format(spectrum, dataset['path']))
            process(cosmo, spectrum, dataset, save_dir,
                    'direct_' + dataset['suffix'])
            if spectrum.startswith('fk_'):
                process(cosmo, spectrum, dataset, save_dir,
                        'from_pk_' + dataset['suffix'], from_pk=True)


if __name__ == '__main__':
    main()
