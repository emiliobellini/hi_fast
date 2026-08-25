"""Time and validate scalar-old versus batch-first P(k,z) evaluation."""
import argparse
import time

import numpy as np
import scipy.interpolate as interp

from hi_fast import HiFast
from hi_fast.io import FitsFile


def stats(value, reference):
    scale = np.maximum(np.abs(reference), np.finfo(float).tiny)
    difference = np.abs(value - reference) / scale
    return np.mean(difference), np.percentile(difference, 99), np.max(difference)


def report(label, value, reference):
    mean, p99, maximum = stats(value, reference)
    print('{}: mean={:.6e}, p99={:.6e}, max={:.6e}'.format(
        label, mean, p99, maximum))


def load_rows(path, spectrum, n_rows):
    start = time.perf_counter()
    fits = FitsFile(path)
    x = np.asarray(fits.get_data('x_data')[:n_rows])
    names = list(fits.get_header(0)['params'])
    k = np.asarray(fits.get_data('k_range_{}'.format(spectrum)))
    z = x[:, names.index('z_pk')]
    ratio = np.asarray(fits.get_data(spectrum)[:n_rows])
    ref_z = np.asarray(fits.get_data('z_array'))
    ref = np.asarray(fits.get_data('ref_{}'.format(spectrum)))[0]
    spline = interp.make_splrep(ref_z, ref.T, s=0)
    data = np.asarray([spline(z_one) for z_one in z]) * ratio
    return x, names, k, z, data, time.perf_counter() - start


def parameters(cosmo, spectrum, names, rows):
    if names != cosmo._params[spectrum]._emu:
        raise ValueError('Dataset parameters do not match {}'.format(spectrum))
    output = []
    for row in rows:
        params = {key: value for key, value in zip(names, row)
                  if key != 'z_pk'}
        for key in cosmo._params[spectrum]._additional:
            params.setdefault(key, cosmo._spectra[spectrum].class_args[key])
        output.append(params)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('-m', '--model', default='lcdm')
    parser.add_argument('-n', '--n-rows', type=int, default=1000)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('--spectra', nargs='+',
                        default=('pk_m', 'pk_cb', 'pk_weyl'))
    args = parser.parse_args()

    cosmo = HiFast(args.model, root='emu', verbose=True)
    for spectrum in args.spectra:
        rows, names, k, z, data, load_time = load_rows(
            args.data_file, spectrum, args.n_rows)
        params = parameters(cosmo, spectrum, names, rows)
        short_name = spectrum.removeprefix('pk_')
        old = np.empty_like(data)

        start = time.perf_counter()
        for index, (z_one, row) in enumerate(zip(z, params)):
            old[index] = cosmo.get_pk_old(
                k, z_one, row, name=short_name, squeeze=True)
        old_time = time.perf_counter() - start

        batch = np.empty_like(data)
        start = time.perf_counter()
        for first in range(0, len(z), args.batch_size):
            last = min(first + args.batch_size, len(z))
            batch[first:last] = cosmo.get_pk(
                k, z[first:last], params[first:last], name=short_name)
        batch_time = time.perf_counter() - start

        print('\n{}: {} rows, {} k modes'.format(spectrum, len(z), len(k)))
        print('Data loading: {:.6f} s'.format(load_time))
        print('Old scalar:  {:.6f} s ({:.6f} s/row)'.format(
            old_time, old_time / len(z)))
        print('Batch:       {:.6f} s ({:.6f} s/row)'.format(
            batch_time, batch_time / len(z)))
        print('Batch/old time ratio: {:.6f}x'.format(
            batch_time / old_time))
        report('old/data', old, data)
        report('batch/data', batch, data)
        report('old/batch', old, batch)


if __name__ == '__main__':
    main()
