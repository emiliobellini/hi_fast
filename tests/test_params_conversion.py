import numpy as np
from hi_fast import HiFast

# Load HiFast instance
cosmo = HiFast(
    'lcdm',
    root='output',
    timeit=True,
    verbose=True)

z = 0.3
params = {
    # 'h': 0.67,
    'H0': 68.,
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
    name='m',
    squeeze=False,
    nonlinear=False,
    timeit=True)

pk_emu_std = cosmo.get_pk_from_class(
    ref_k,
    z,
    params,
    name='m',
    squeeze=False,
    nonlinear=False,
    timeit=True)
