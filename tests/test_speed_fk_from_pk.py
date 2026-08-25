"""Benchmark direct growth-rate emulation against differentiation of P(k,z).

Example
-------
python tests/test_speed_fk_from_pk.py data.fits -m lcdm -i 0 -r 20 -w 3
"""
import argparse
import time

import numpy as np
import scipy.interpolate as interp

from hi_fast import HiFast
from hi_fast.io import FitsFile


def elapsed_call(function, *args, **kwargs):
    """Return a function's result and its wall-clock execution time."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def load_samples(data_file, spectrum, start_index, n_rows):
    """Load a validation slice and reconstruct its physical growth rates."""
    start = time.perf_counter()
    fits = FitsFile(data_file)
    stop_index = start_index + n_rows
    x_data = np.asarray(fits.get_data('x_data')[start_index:stop_index])
    if not len(x_data):
        raise IndexError('No rows found starting at index {}'.format(
            start_index))
    names = list(fits.get_header(0)['params'])
    k = np.asarray(fits.get_data('k_range_{}'.format(spectrum)))
    z_array = np.asarray(fits.get_data('z_array'))
    stored_ratio = np.asarray(
        fits.get_data(spectrum)[start_index:stop_index])
    reference = np.asarray(fits.get_data('ref_{}'.format(spectrum)))[0]
    redshifts = x_data[:, names.index('z_pk')]
    reference_spline = interp.make_splrep(z_array, reference.T, s=0)
    reference_at_z = np.asarray(
        [reference_spline(z) for z in redshifts])
    stored_fk = reference_at_z * stored_ratio
    return (x_data, names, k, redshifts, stored_fk,
            time.perf_counter() - start)


def make_parameters(cosmo, spectrum, names, values):
    """Convert a FITS input row into arguments accepted by HiFast."""
    expected = cosmo._params[spectrum]._emu
    if names != expected:
        raise ValueError(
            'Parameters in the data file do not match {}: {} != {}'.format(
                spectrum, names, expected))
    params = {name: value for name, value in zip(names, values)
              if name != 'z_pk'}
    for name in cosmo._params[spectrum]._additional:
        params.setdefault(name, cosmo._spectra[spectrum].class_args[name])
    return params


def evaluate(cosmo, k, z, params, name, method, from_pk):
    """Evaluate one of the direct, old, or new growth-rate paths."""
    # get_from_pk=True adds A_s and n_s to its input dictionary.
    get_fk = getattr(cosmo, method)
    return np.asarray(get_fk(
        k, z, params.copy(), name=name, get_from_pk=from_pk,
        nonlinear=False, squeeze=True))


def benchmark(cosmo, k, redshifts, parameters, name, method, from_pk,
              warmups, repeats):
    """Warm up and time one evaluation path over every selected row."""
    for _ in range(warmups):
        evaluate(cosmo, k, redshifts[0], parameters[0], name, method,
                 from_pk)

    n_rows = len(redshifts)
    timings = np.empty((repeats, n_rows))
    results = np.empty((n_rows, len(k)))
    for repeat in range(repeats):
        for index, (z, params) in enumerate(zip(redshifts, parameters)):
            start = time.perf_counter()
            results[index] = evaluate(
                cosmo, k, z, params, name, method, from_pk)
            timings[repeat, index] = time.perf_counter() - start
    return results, timings.ravel()


def relative_statistics(result, reference):
    """Return robust relative-difference statistics."""
    scale = np.maximum(np.abs(reference), np.finfo(float).tiny)
    difference = np.abs(result - reference) / scale
    return {
        'mean': np.mean(difference),
        'p99': np.percentile(difference, 99),
        'max': np.max(difference),
    }


def print_timings(label, timings):
    print('{} timing over {} measured calls:'.format(label, len(timings)))
    print('  total:  {:.6f} s'.format(np.sum(timings)))
    print('  mean:   {:.6f} s'.format(np.mean(timings)))
    print('  median: {:.6f} s'.format(np.median(timings)))
    print('  min:    {:.6f} s'.format(np.min(timings)))
    print('  max:    {:.6f} s'.format(np.max(timings)))


def print_validation(label, result, reference):
    stats = relative_statistics(result, reference)
    print('{} relative difference:'.format(label))
    print('  mean: {:.6e} ({:.6e}%)'.format(
        stats['mean'], 100 * stats['mean']))
    print('  p99:  {:.6e} ({:.6e}%)'.format(
        stats['p99'], 100 * stats['p99']))
    print('  max:  {:.6e} ({:.6e}%)'.format(
        stats['max'], 100 * stats['max']))


def run_spectrum(cosmo, args, spectrum):
    """Load, benchmark, and validate one fk spectrum."""
    print('\n' + '=' * 72)
    print('Spectrum: {}'.format(spectrum))
    sample = load_samples(
        args.data_file, spectrum, args.idx_data, args.n_rows)
    values, names, k, redshifts, stored_fk, load_time = sample
    parameters = [make_parameters(cosmo, spectrum, names, row)
                  for row in values]
    name = spectrum.removeprefix('fk_')
    print('Dataset loading and reference reconstruction: {:.6f} s'.format(
        load_time))
    print('Rows: {}--{} ({} rows), k modes: {}'.format(
        args.idx_data, args.idx_data + len(values) - 1,
        len(values), len(k)))
    print('Warm-up calls: {}, passes over selected rows: {}'.format(
        args.warmups, args.repeats))

    _, direct_times = benchmark(
        cosmo, k, redshifts, parameters, name, 'get_fk_old', False,
        args.warmups, args.repeats)
    old, old_times = benchmark(
        cosmo, k, redshifts, parameters, name, 'get_fk_old', True,
        args.warmups, args.repeats)
    new, new_times = benchmark(
        cosmo, k, redshifts, parameters, name, 'get_fk', True,
        args.warmups, args.repeats)

    print_timings('Direct fk', direct_times)
    print_timings('Old fk derived from pk', old_times)
    print_timings('New fk derived from pk', new_times)
    print('New/old mean-time ratio: {:.3f}x'.format(
        np.mean(new_times) / np.mean(old_times)))

    print('\nOutput validation')
    print_validation('Old fk from pk versus stored data', old, stored_fk)
    print_validation('New fk from pk versus stored data', new, stored_fk)
    print_validation('Old fk from pk versus new fk from pk', old, new)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark direct fk and fk derived from pk.')
    parser.add_argument('data_file', help='Validation FITS file')
    parser.add_argument('--idx-data', '-i', type=int, default=0,
                        help='First validation row to use')
    parser.add_argument('--n-rows', '-n', type=int, default=1000,
                        help='Maximum number of consecutive rows to use')
    parser.add_argument('--model', '-m', default='lcdm',
                        help='HiFast emulator family')
    parser.add_argument('--repeats', '-r', type=int, default=1,
                        help='Number of passes over the selected rows')
    parser.add_argument('--warmups', '-w', type=int, default=3,
                        help='Unmeasured TensorFlow warm-up calls')
    parser.add_argument(
        '--spectra', nargs='+', default=('fk_m', 'fk_cb', 'fk_weyl'),
        choices=('fk_m', 'fk_cb', 'fk_weyl'),
        help='Growth-rate spectra to benchmark')
    args = parser.parse_args()
    if args.repeats < 1 or args.n_rows < 1 or args.warmups < 0:
        parser.error('--repeats and --n-rows must be positive; '
                     '--warmups must be non-negative')

    cosmo, load_time = elapsed_call(
        HiFast, args.model, root='emu', timeit=False, verbose=True)
    print('HiFast model loading: {:.6f} s'.format(load_time))
    for spectrum in args.spectra:
        run_spectrum(cosmo, args, spectrum)


if __name__ == '__main__':
    main()
