# Test that the growth rates emulated are consistent
# with CLASS and the reference data.
# Example of usage:
# python tests/test_fk.py path/to/data.fits -i 0 -m lcdm
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
    # cosmo.print_info()

    for spectrum in ['fk_m', 'fk_cb', 'fk_weyl']:

        # Load reference fk from file
        fits = io.FitsFile(args.data_file)
        x_data = fits.get_data('x_data')[args.idx_data]
        params_data = [key for key in fits.get_header(0)['params']]

        # Check that parameters in the data file are
        # consistent with the emulator
        if params_data != cosmo._params[spectrum]._emu:
            raise ValueError(
                'Parameters in the data file are not consistent with the'
                ' emulator. Data file parameters: {}, Emulator parameters: {}'
                ''.format(params_data, cosmo._params[spectrum]._emu))

        # Assign parameters to the emulator
        z = x_data[params_data.index('z_pk')]
        # Varied parameters
        params = {key: val
                  for key, val in zip(params_data, x_data) if key != 'z_pk'}
        # Extract additional parameters from the default values
        for param in cosmo._params[spectrum]._additional:
            if param not in params:
                params[param] = cosmo._spectra[spectrum].class_args[param]

        ref_k = fits.get_data('k_range_{}'.format(spectrum))
        ref_z = fits.get_data('z_array')

        fk_over_fk_ref = fits.get_data(spectrum)[args.idx_data]
        ref_fk = interp.make_splrep(
            ref_z, fits.get_data('ref_{}'.format(spectrum))[0].T, s=0)(z)
        fk_data = ref_fk * fk_over_fk_ref

        fk_emu = cosmo.get_fk(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            nonlinear=False,
            timeit=True)

        fk_class_0 = cosmo.get_fk_from_class(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=0,
            nonlinear=False,
            timeit=True)

        fk_class_1 = cosmo.get_fk_from_class(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=1,
            nonlinear=False,
            timeit=True)

        fk_class_2 = cosmo.get_fk_from_class(
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
            'Test precision fk {} at z={:.2f}.\n Data file: {},\n Index: {}'
            ''.format(spectrum, z, args.data_file, args.idx_data))
        plt.plot(
            ref_k,
            np.abs(fk_emu[:, 0]/fk_class_2[:, 0] - 1.)*100.,
            label='HiFast Emu/Class(high prec)')
        plt.plot(
            ref_k,
            np.abs(fk_class_0[:, 0]/fk_class_2[:, 0] - 1.)*100.,
            label='Class(low prec)/Class(high prec)')
        plt.plot(
            ref_k,
            np.abs(fk_class_1[:, 0]/fk_class_2[:, 0] - 1.)*100.,
            label='Class(std prec)/Class(high prec)')
        plt.plot(
            ref_k,
            np.abs(fk_data/fk_class_2[:, 0] - 1.)*100.,
            label='Data/Class(high prec)')
        plt.axhline(0.1, c='k', lw=0.1)

        plt.xlabel('k [h/Mpc]')
        plt.ylabel('perc_rel_diff [%]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/test_{}_{}.pdf'.format(args.model, spectrum))
        plt.close()
