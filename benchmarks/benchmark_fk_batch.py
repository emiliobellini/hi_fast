"""Benchmark direct and P(k,z)-derived growth-rate evaluation.

Example
-------
python benchmarks/benchmark_fk_batch.py data.fits \
    -m lcdm -n 1000 -b 256
"""
import time

import numpy as np
import scipy.interpolate as interp

from hi_fast import HiFast
from hi_fast.io import FitsFile

from _common import (benchmark, elapsed_call, parse_arguments,
                     report_comparison, report_header)


SPECTRA = ('fk_m', 'fk_cb', 'fk_weyl')


def load_rows(path, spectrum, start_index, n_rows):
    """Load rows and reconstruct their physical growth rates."""
    start = time.perf_counter()
    fits = FitsFile(path)
    stop_index = start_index + n_rows
    rows = np.asarray(fits.get_data('x_data')[start_index:stop_index])
    if not len(rows):
        raise IndexError('No rows found starting at index {}'.format(
            start_index))
    names = list(fits.get_header(0)['params'])
    k = np.asarray(fits.get_data('k_range_{}'.format(spectrum)))
    redshifts = rows[:, names.index('z_pk')]
    ratio = np.asarray(
        fits.get_data(spectrum)[start_index:stop_index])
    reference_z = np.asarray(fits.get_data('z_array'))
    reference = np.asarray(fits.get_data('ref_{}'.format(spectrum)))[0]
    spline = interp.make_splrep(reference_z, reference.T, s=0)
    data = np.asarray([spline(z) for z in redshifts]) * ratio
    return (rows, names, k, redshifts, data,
            time.perf_counter() - start)


def make_parameters(cosmo, spectrum, names, rows):
    """Convert FITS rows into dictionaries accepted by HiFast."""
    if names != cosmo._params[spectrum]._emu:
        raise ValueError('Dataset parameters do not match {}'.format(
            spectrum))
    parameters = []
    for row in rows:
        values = {key: value for key, value in zip(names, row)
                  if key != 'z_pk'}
        for key in cosmo._params[spectrum]._additional:
            values.setdefault(key, cosmo._spectra[spectrum].class_args[key])
        parameters.append(values)
    return parameters


def run_method(cosmo, args, name, k, redshifts, parameters, from_pk):
    """Benchmark one growth-rate evaluation method."""
    n_rows = len(redshifts)

    def evaluate(first, last, batch_size):
        rows = [row.copy() for row in parameters[first:last]]
        return cosmo.get_fk(
            k, redshifts[first:last], rows, name=name,
            get_from_pk=from_pk, batch_size=batch_size)

    one_row, one_row_time = benchmark(
        evaluate, n_rows, 1, args.warmups, args.repeats)
    batch, batch_time = benchmark(
        evaluate, n_rows, args.batch_size, args.warmups, args.repeats)
    return one_row, one_row_time, batch, batch_time


def run_spectrum(cosmo, args, spectrum):
    """Load, benchmark, and validate one growth-rate spectrum."""
    sample = load_rows(
        args.data_file, spectrum, args.idx_data, args.n_rows)
    rows, names, k, redshifts, data, load_time = sample
    parameters = make_parameters(cosmo, spectrum, names, rows)
    name = spectrum.removeprefix('fk_')
    n_rows = len(rows)

    direct = run_method(
        cosmo, args, name, k, redshifts, parameters, from_pk=False)
    from_pk = run_method(
        cosmo, args, name, k, redshifts, parameters, from_pk=True)

    report_header(spectrum, n_rows, len(k), 'k', load_time)
    report_comparison(
        'Direct', *direct, data, n_rows, args.batch_size)
    report_comparison(
        'From-pk', *from_pk, data, n_rows, args.batch_size)


def main():
    args = parse_arguments(
        'Benchmark direct and P(k,z)-derived growth rates.', SPECTRA)
    cosmo, load_time = elapsed_call(
        HiFast, args.model, root='emu', timeit=False, verbose=False)
    print('HiFast model loading: {:.6f} s'.format(load_time))
    for spectrum in args.spectra:
        run_spectrum(cosmo, args, spectrum)


if __name__ == '__main__':
    main()
