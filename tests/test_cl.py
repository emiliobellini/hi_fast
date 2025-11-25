import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
from hi_fast import HiFast

# Load HiFast instance
cosmo = HiFast(
    'lcdm',
    root='/ceph/hpc/home/bellinie/hi_fast/output',
    timeit=True,
    verbose=True)
# cosmo.print_cosmo_params()

# Index to plot
idx_data = 1

# for spectrum in ['cl_TT', 'cl_TE', 'cl_EE', 'cl_BB', 'cl_Tp', 'cl_pp']:
for spectrum in ['cl_EE']:

    # Load reference Pk from file
    fits = io.FitsFile(
        '/ceph/hpc/data/s25r06-05-users/old/lcdm/cl_100_thin.fits')
    x_data = fits.get_data('x_data')[idx_data]
    params = {
        'h': x_data[0],
        'Omega_m': x_data[1],
        'Omega_b': x_data[2],
        'ln_A_s_1e10': x_data[3],
        'n_s': x_data[4],
        'tau_reio': x_data[5],
    }

    ref_ell = fits.get_data('ell_range_{}_lensed'.format(spectrum))

    cl_over_cl_ref = fits.get_data('{}_lensed'.format(spectrum))[idx_data]
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
    plt.title('Test precision Cl {}'.format(spectrum))
    plt.plot(ref_ell,
             np.abs(cl_emu/cl_data - 1.)*100.,
             label='HiFast Emu/Data')
    plt.plot(ref_ell,
             np.abs(cl_class_0/cl_data - 1.)*100.,
             label='Class_0/Data')
    plt.plot(ref_ell,
             np.abs(cl_class_1/cl_data - 1.)*100.,
             label='Class_1/Data')
    plt.plot(ref_ell,
             np.abs(cl_class_2/cl_data - 1.)*100.,
             label='Class_2/Data')
    plt.axhline(0.1, c='k', lw=0.1)

    plt.xlabel('ell')
    plt.ylabel('perc_rel_diff [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/test_{}.pdf'.format(spectrum))
    plt.close()
