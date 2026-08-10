# Test that the growth rates emulated are consistent
# with CLASS and the reference data.
# Example of usage:
# python test_pk.py path/to/data.fits -i 0 -m lcdm
import argparse
import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
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

    for spectrum in ['pk_m', 'pk_cb', 'pk_weyl']:

        # Load reference Pk from file
        fits = io.FitsFile(args.data_file)
        x_data = fits.get_data('x_data')[args.idx_data]
        z = x_data[0]
        params = {
            'h': x_data[1],
            'Omega_m': x_data[2],
            'Omega_b': x_data[3],
            'tau_reio': x_data[4],
            'ln_A_s_1e10': cosmo._spectra[spectrum].class_args['ln_A_s_1e10'],
            'n_s': cosmo._spectra[spectrum].class_args['n_s'],
        }

        ref_k = fits.get_data('k_range_{}'.format(spectrum))
        ref_z = fits.get_data('z_array')

        pk_over_pk_ref = fits.get_data(spectrum)[args.idx_data]
        ref_pk = interp.make_splrep(
            ref_z, fits.get_data('ref_{}'.format(spectrum))[0].T, s=0)(z)
        pk_data = ref_pk * pk_over_pk_ref

        pk_emu = cosmo.get_pk(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            nonlinear=False,
            timeit=True)

        pk_class_0 = cosmo.get_pk_from_class(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=0,
            nonlinear=False,
            timeit=True)

        pk_class_1 = cosmo.get_pk_from_class(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=1,
            nonlinear=False,
            timeit=True)

        pk_class_2 = cosmo.get_pk_from_class(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=2,
            nonlinear=False,
            timeit=True)

        # Plotting
        plt.title(
            'Test precision Pk {} at z={:.2f}.\n Data file: {},\n Index: {}'
            ''.format(spectrum, z, args.data_file, args.idx_data))
        plt.plot(
            ref_k,
            np.abs(pk_emu[:, 0]/pk_class_2[:, 0] - 1.)*100.,
            label='HiFast Emu/Class(high prec)')
        plt.plot(
            ref_k,
            np.abs(pk_class_0[:, 0]/pk_class_2[:, 0] - 1.)*100.,
            label='Class(low prec)/Class(high prec)')
        plt.plot(
            ref_k,
            np.abs(pk_class_1[:, 0]/pk_class_2[:, 0] - 1.)*100.,
            label='Class(std prec)/Class(high prec)')
        plt.plot(
            ref_k,
            np.abs(pk_data/pk_class_2[:, 0] - 1.)*100.,
            label='Data/Class(high prec)')
        plt.axhline(0.1, c='k', lw=0.1)

        plt.xlabel('k [h/Mpc]')
        plt.ylabel('perc_rel_diff [%]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/test_{}.pdf'.format(spectrum))
        plt.close()
