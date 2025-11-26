import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from hi_fast import HiFast

# Load HiFast instance
cosmo = HiFast(
    'lcdm',
    root='/ceph/hpc/home/bellinie/hi_fast/output/lcdm/export_new',
    timeit=True,
    verbose=True)
# cosmo.print_input_params()

# Index to plot
idx_data = 1

for spectrum in ['pk_m', 'pk_cb', 'pk_weyl']:

    # Load reference Pk from file
    fits = io.FitsFile(
        '/ceph/hpc/home/bellinie/hi_fast/output/lcdm/sample/pk_100_thin.fits')
    x_data = fits.get_data('x_data')[idx_data]
    z = x_data[0]
    params = {
        'h': x_data[1],
        'Omega_m': x_data[2],
        'Omega_b': x_data[3],
        'tau_reio': x_data[4],
        'ln_A_s_1e10': cosmo._emu[spectrum].class_args['ln_A_s_1e10'],
        'n_s': cosmo._emu[spectrum].class_args['n_s'],
    }

    ref_k = fits.get_data('k_range_{}'.format(spectrum))
    ref_z = fits.get_data('z_array')

    pk_over_pk_ref = fits.get_data(spectrum)[idx_data]
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
    plt.title('Test precision Pk {} at z={:.2f}'.format(spectrum, z))
    plt.plot(ref_k,
             np.abs(pk_emu[:, 0]/pk_data - 1.)*100.,
             label='HiFast Emu/Data')
    plt.plot(ref_k,
             np.abs(pk_class_0[:, 0]/pk_data - 1.)*100.,
             label='Class_0/Data')
    plt.plot(ref_k,
             np.abs(pk_class_1[:, 0]/pk_data - 1.)*100.,
             label='Class_1/Data')
    plt.plot(ref_k,
             np.abs(pk_class_2[:, 0]/pk_data - 1.)*100.,
             label='Class_2/Data')
    plt.axhline(0.1, c='k', lw=0.1)

    plt.xlabel('k [h/Mpc]')
    plt.ylabel('perc_rel_diff [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/test_{}.pdf'.format(spectrum))
    plt.close()
