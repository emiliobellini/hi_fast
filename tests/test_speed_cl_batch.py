"""Time and validate scalar-old versus batch-first CMB spectra."""
import argparse
import time

import numpy as np

from hi_fast import HiFast
from hi_fast.io import FitsFile


def report(label, value, reference):
    scale = np.maximum(np.abs(reference), np.finfo(float).tiny)
    difference = np.abs(value - reference) / scale
    print('{}: mean={:.6e}, p99={:.6e}, max={:.6e}'.format(
        label, np.mean(difference), np.percentile(difference, 99),
        np.max(difference)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('-m', '--model', default='lcdm')
    parser.add_argument('-n', '--n-rows', type=int, default=1000)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('--spectra', nargs='+', default=(
        'cl_TT_lensed', 'cl_TE_lensed', 'cl_EE_lensed',
        'cl_BB_lensed', 'cl_Tp_lensed', 'cl_pp_lensed'))
    args = parser.parse_args()

    cosmo = HiFast(args.model, root='emu', verbose=True)
    fits = FitsFile(args.data_file)
    names = list(fits.get_header(0)['params'])
    x = np.asarray(fits.get_data('x_data')[:args.n_rows])

    for spectrum in args.spectra:
        start_load = time.perf_counter()
        ell = np.asarray(fits.get_data('ell_range_{}'.format(spectrum)))
        ratio = np.asarray(fits.get_data(spectrum)[:args.n_rows])
        reference = np.asarray(fits.get_data('ref_{}'.format(spectrum)))[0]
        data = ratio * reference
        params = []
        for row in x:
            values = dict(zip(names, row))
            for key in cosmo._params[spectrum]._additional:
                values.setdefault(
                    key, cosmo._spectra[spectrum].class_args[key])
            params.append(values)
        load_time = time.perf_counter() - start_load
        name = spectrum.removeprefix('cl_').removesuffix('_lensed')
        old = np.empty_like(data)

        start = time.perf_counter()
        for index, row in enumerate(params):
            old[index] = cosmo.get_cell_old(
                ell, row, name=name, squeeze=True)
        old_time = time.perf_counter() - start

        batch = np.empty_like(data)
        start = time.perf_counter()
        for first in range(0, len(params), args.batch_size):
            last = min(first + args.batch_size, len(params))
            batch[first:last] = cosmo.get_cell(
                ell, params[first:last], name=name)
        batch_time = time.perf_counter() - start

        print('\n{}: {} rows, {} ell modes'.format(
            spectrum, len(params), len(ell)))
        print('Data loading: {:.6f} s'.format(load_time))
        print('Old scalar:  {:.6f} s ({:.6f} s/row)'.format(
            old_time, old_time / len(params)))
        print('Batch:       {:.6f} s ({:.6f} s/row)'.format(
            batch_time, batch_time / len(params)))
        print('Batch/old time ratio: {:.6f}x'.format(
            batch_time / old_time))
        report('old/data', old, data)
        report('batch/data', batch, data)
        report('old/batch', old, batch)


if __name__ == '__main__':
    main()
