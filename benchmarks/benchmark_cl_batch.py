"""Benchmark one-row versus multi-row CMB-spectrum evaluation.

Example
-------
python benchmarks/benchmark_cl_batch.py data.fits -m lcdm -n 1000 -b 256
"""
import time

import numpy as np

from hi_fast import HiFast
from hi_fast.io import FitsFile

from _common import (benchmark, elapsed_call, parse_arguments,
                     report_comparison, report_difference, report_header)


SPECTRA = (
    'cl_TT_lensed', 'cl_TE_lensed', 'cl_EE_lensed',
    'cl_BB_lensed', 'cl_Tp_lensed', 'cl_pp_lensed')


def load_rows(path, spectrum, start_index, n_rows):
    """Load rows and reconstruct their physical CMB spectra."""
    start = time.perf_counter()
    fits = FitsFile(path)
    stop_index = start_index + n_rows
    rows = np.asarray(fits.get_data('x_data')[start_index:stop_index])
    if not len(rows):
        raise IndexError('No rows found starting at index {}'.format(
            start_index))
    names = list(fits.get_header(0)['params'])
    ell = np.asarray(fits.get_data('ell_range_{}'.format(spectrum)))
    ratio = np.asarray(
        fits.get_data(spectrum)[start_index:stop_index])
    reference = np.asarray(fits.get_data('ref_{}'.format(spectrum)))[0]
    data = ratio * reference
    return rows, names, ell, data, time.perf_counter() - start


def make_parameters(cosmo, spectrum, names, rows):
    """Convert FITS rows into dictionaries accepted by HiFast."""
    parameters = []
    for row in rows:
        values = dict(zip(names, row))
        for key in cosmo._params[spectrum]._additional:
            values.setdefault(key, cosmo._spectra[spectrum].class_args[key])
        parameters.append(values)
    return parameters


def run_spectrum(cosmo, args, spectrum):
    """Load, benchmark, and validate one CMB spectrum."""
    sample = load_rows(
        args.data_file, spectrum, args.idx_data, args.n_rows)
    rows, names, ell, data, load_time = sample
    parameters = make_parameters(cosmo, spectrum, names, rows)
    name = spectrum.removeprefix('cl_').removesuffix('_lensed')
    n_rows = len(rows)
    report_header(spectrum, n_rows, len(ell), 'ell', load_time)

    def evaluate(first, last, batch_size):
        return cosmo.get_cell(
            ell, parameters[first:last], name=name,
            batch_size=batch_size)

    print('  Running one-cosmology batches...', flush=True)
    one_row, one_row_time = benchmark(
        evaluate, n_rows, 1, args.warmups, args.repeats)
    print('  Running batches of {} evaluations...'.format(
        args.batch_size), flush=True)
    batch, batch_time = benchmark(
        evaluate, n_rows, args.batch_size, args.warmups, args.repeats)

    report_comparison(
        '', one_row, one_row_time, batch, batch_time, n_rows,
        args.batch_size)
    report_difference('one-row/data', one_row, data)
    report_difference('batch/data', batch, data)


def main():
    args = parse_arguments(
        'Benchmark one-row and multi-row CMB-spectrum evaluation.', SPECTRA)
    cosmo, load_time = elapsed_call(
        HiFast, args.model, root='emu', timeit=False, verbose=False)
    print('HiFast model loading: {:.6f} s'.format(load_time))
    for spectrum in args.spectra:
        print('\nLoading {} validation data...'.format(spectrum), flush=True)
        run_spectrum(cosmo, args, spectrum)


if __name__ == '__main__':
    main()
