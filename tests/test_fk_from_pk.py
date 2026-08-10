# Test the accuracy of the emulator for the
# growth rate f(k,z) when computed from the
# power spectrum P(k,z) and from the emulator directly.
# Example of usage:
# python test_fk_from_pk.py path/to/data.fits -i 0 -m lcdm
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

    for spectrum in ['fk_m', 'fk_cb', 'fk_weyl']:

        # Load reference fk from file
        fits = io.FitsFile(args.data_file)
        x_data = fits.get_data('x_data')[args.idx_data]
        z = x_data[0]
        params = {
            'h': x_data[1],
            'Omega_m': x_data[2],
            'Omega_b': x_data[3],
            'tau_reio': x_data[4],
        }

        ref_k = fits.get_data('k_range_{}'.format(spectrum))
        ref_z = fits.get_data('z_array')

        fk_over_fk_ref = fits.get_data(spectrum)[args.idx_data]
        ref_fk = interp.make_splrep(
            ref_z, fits.get_data('ref_{}'.format(spectrum))[0].T, s=0)(z)
        fk_data = ref_fk * fk_over_fk_ref

        fk_emu_direct = cosmo.get_fk(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            get_from_pk=False,
            squeeze=False,
            nonlinear=False,
            timeit=True)

        fk_emu_from_pk = cosmo.get_fk(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            get_from_pk=True,
            squeeze=False,
            nonlinear=False,
            timeit=True)

        fk_class = cosmo.get_fk_from_class(
            ref_k,
            z,
            params,
            name=spectrum.split('_')[-1],
            squeeze=False,
            precision=1,
            nonlinear=False,
            timeit=True)

        # Plotting
        plt.title(
            'Test fk {} from Pk at z={:.2f}.\n Data file: {},\n Index: {}'
            ''.format(spectrum, z, args.data_file, args.idx_data))
        plt.plot(
            ref_k,
            np.abs(fk_emu_direct[:, 0]/fk_class[:, 0] - 1.)*100.,
            label='HiFast Emu(direct)/Class')
        plt.plot(
            ref_k,
            np.abs(fk_emu_from_pk[:, 0]/fk_class[:, 0] - 1.)*100.,
            label='HiFast Emu(from Pk)/Class')
        plt.plot(
            ref_k,
            np.abs(fk_data/fk_class[:, 0] - 1.)*100.,
            label='Data/Class')
        plt.axhline(0.1, c='k', lw=0.1)

        plt.xlabel('k [h/Mpc]')
        plt.ylabel('perc_rel_diff [%]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/test_fk_from_pk_{}_{}.pdf'.format(
            args.model, spectrum))
        plt.close()
