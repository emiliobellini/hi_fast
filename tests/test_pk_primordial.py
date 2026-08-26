import matplotlib.pyplot as plt
import numpy as np
from hi_fast import HiFast

# Load HiFast instance
cosmo = HiFast(
    'lcdm',
    root='emu',
    timeit=True,
    verbose=True)
# cosmo.print_info()

for spectrum in ['pk_m', 'pk_cb', 'pk_weyl']:

    z = 0.3
    params = {
        'h': 0.67,
        'Omega_m': 0.3,
        'Omega_b': 0.04,
        'tau_reio': 0.06,
        'ln_A_s_1e10': 3.04,
        'n_s': 0.966,
    }

    ref_k = np.logspace(-5., np.log10(50.), num=600)

    pk_emu_std = cosmo.get_pk(
        ref_k,
        z,
        params,
        name=spectrum.split('_')[-1],
        squeeze=False,
        nonlinear=False,
        timeit=True)

    pk_class_std = cosmo.get_pk_from_class(
        ref_k,
        z,
        params,
        name=spectrum.split('_')[-1],
        squeeze=False,
        precision=1,
        nonlinear=False,
        timeit=True)

    # Rescale primordial Pk
    params['ln_A_s_1e10'] = 4.
    params['n_s'] = 1.2

    pk_emu_res = cosmo.get_pk(
        ref_k,
        z,
        params,
        name=spectrum.split('_')[-1],
        squeeze=False,
        nonlinear=False,
        timeit=True)

    pk_class_res = cosmo.get_pk_from_class(
        ref_k,
        z,
        params,
        name=spectrum.split('_')[-1],
        squeeze=False,
        precision=1,
        nonlinear=False,
        timeit=True)

    # Plotting
    plt.title('Test precision Pk {} at z={:.2f}'.format(spectrum, z))
    plt.plot(ref_k,
             np.abs(pk_emu_std[0, 0]/pk_class_std[0, 0] - 1.)*100.,
             label='HiFast Emu/Class Standard')
    plt.plot(ref_k,
             np.abs(pk_emu_res[0, 0]/pk_class_res[0, 0] - 1.)*100.,
             '--',
             label='HiFast Emu/Class Rescaled')
    plt.axhline(0.1, c='k', lw=0.1)

    plt.xlabel('k [h/Mpc]')
    plt.ylabel('perc_rel_diff [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/test_{}_primordial.pdf'.format(spectrum))
    plt.close()
