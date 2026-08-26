"""Shared utilities for the batch benchmark scripts."""
import argparse
import time

import numpy as np


def parse_arguments(description, default_spectra):
    """Parse the command-line interface shared by every benchmark."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_file', help='Validation FITS file')
    parser.add_argument('-m', '--model', default='lcdm',
                        help='HiFast emulator family')
    parser.add_argument('-i', '--idx-data', type=int, default=0,
                        help='First validation row to use')
    parser.add_argument('-n', '--n-rows', type=int, default=1000,
                        help='Maximum number of consecutive rows to use')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='Model inputs evaluated per model call')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='Number of measured evaluations')
    parser.add_argument('-w', '--warmups', type=int, default=3,
                        help='Number of unmeasured warm-up evaluations')
    parser.add_argument('--spectra', nargs='+', default=default_spectra,
                        help='Spectra to benchmark')
    args = parser.parse_args()

    if args.idx_data < 0:
        parser.error('--idx-data must be non-negative')
    if args.n_rows < 1 or args.batch_size < 1 or args.repeats < 1:
        parser.error('--n-rows, --batch-size, and --repeats must be positive')
    if args.warmups < 0:
        parser.error('--warmups must be non-negative')
    return args


def elapsed_call(function, *args, **kwargs):
    """Return a function result together with its wall-clock duration."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def benchmark(evaluate, n_rows, batch_size, warmups, repeats):
    """Warm up and time a public API call using the requested batch size."""
    warmup_rows = min(batch_size, n_rows)
    for _ in range(warmups):
        evaluate(0, warmup_rows, batch_size)

    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = np.asarray(evaluate(0, n_rows, batch_size))
        elapsed.append(time.perf_counter() - start)
    return result, np.mean(elapsed)


def report_header(spectrum, n_rows, n_modes, mode_name, load_time):
    """Print the compact per-spectrum heading shared by all benchmarks."""
    print('\n{}: {} cosmologies, {} {} modes'.format(
        spectrum, n_rows, n_modes, mode_name))
    print('Data loading: {:.6f} s'.format(load_time))


def report_difference(label, value, reference):
    """Print compact relative-difference statistics."""
    scale = np.maximum(np.abs(reference), np.finfo(float).tiny)
    difference = np.abs(value - reference) / scale
    print('{}: mean={:.6e}, p99={:.6e}, max={:.6e}'.format(
        label, np.mean(difference), np.percentile(difference, 99),
        np.max(difference)))


def report_comparison(label, one_row, one_row_time, batch, batch_time,
                      n_evaluations, batch_size):
    """Print timings and validation for one-row and multi-row batching."""
    timing_prefix = '{} '.format(label) if label else ''
    result_prefix = '{} '.format(label.lower()) if label else ''
    print('{}One-row: {:.6f} s ({:.6f} s/evaluation)'.format(
        timing_prefix, one_row_time, one_row_time / n_evaluations))
    print('{}Batch ({}): {:.6f} s ({:.6f} s/evaluation)'.format(
        timing_prefix, batch_size, batch_time,
        batch_time / n_evaluations))
    print('{}Batch/one-row time ratio: {:.6f}x'.format(
        timing_prefix, batch_time / one_row_time))
    report_difference(
        '{}one-row/batch'.format(result_prefix), one_row, batch)
