# Test that the CMB spectra emulated are consistent
# with CLASS and the reference data.
# Example of usage:
# python test_cl.py path/to/data.fits -i 0 -m lcdm
import argparse
import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
from hi_fast import HiFast


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_file',
        type=str,
        help='Path to the data file')
    parser.add_argument(
        '--idx-data',
        '-i',
        type=int,
        default=0,
        help='Index of the data to plot')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='lcdm',
        help='Model to use')
    args = parser.parse_args()

    # Load HiFast instance
    cosmo = HiFast(
        args.model,
        root='emu',
        timeit=True,
        verbose=True)
    # cosmo.print_input_params()

    for spectrum in ['cl_TT', 'cl_TE', 'cl_EE', 'cl_BB', 'cl_Tp', 'cl_pp']:

        # Load reference Cl from file
        fits = io.FitsFile(args.data_file)
        x_data = fits.get_data('x_data')[args.idx_data]
        params = {
            'h': x_data[0],
            'Omega_m': x_data[1],
            'Omega_b': x_data[2],
            'ln_A_s_1e10': x_data[3],
            'n_s': x_data[4],
            'tau_reio': x_data[5],
        }

        ref_ell = fits.get_data('ell_range_{}_lensed'.format(spectrum))

        cl_over_cl_ref = fits.get_data(
            '{}_lensed'.format(spectrum))[args.idx_data]
        ref_cl = fits.get_data('ref_{}_lensed'.format(spectrum))[0]
        cl_data = ref_cl * cl_over_cl_ref

        cl_emu = cosmo.get_cell(
            ref_ell,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            timeit=True)

        cl_class_0 = cosmo.get_cell_from_class(
            ref_ell,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=0,
            timeit=True)

        cl_class_1 = cosmo.get_cell_from_class(
            ref_ell,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=1,
            timeit=True)

        cl_class_2 = cosmo.get_cell_from_class(
            ref_ell,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=2,
            timeit=True)

        # Plotting
        plt.title(
            'Test precision Cl {}.\n Data file: {},\n Index: {}'
            ''.format(spectrum, args.data_file, args.idx_data))
        plt.plot(
            ref_ell,
            np.abs(cl_emu/cl_class_2 - 1.)*100.,
            label='HiFast Emu/Class(high prec)')
        plt.plot(
            ref_ell,
            np.abs(cl_class_0/cl_class_2 - 1.)*100.,
            label='Class(low prec)/Class(high prec)')
        plt.plot(
            ref_ell,
            np.abs(cl_class_1/cl_class_2 - 1.)*100.,
            label='Class(std prec)/Class(high prec)')
        plt.plot(
            ref_ell,
            np.abs(cl_data/cl_class_2 - 1.)*100.,
            label='Data/Class(high prec)')
        plt.axhline(0.1, c='k', lw=0.1)

        plt.xlabel('ell')
        plt.ylabel('perc_rel_diff [%]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/test_{}_{}.pdf'.format(args.model, spectrum))
        plt.close()
