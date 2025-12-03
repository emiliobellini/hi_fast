import matplotlib.pyplot as plt
import numpy as np
from hi_fast import HiFast

z = 0.3
ref_k = np.logspace(-5., np.log10(50.), num=600)
ref_ell = np.arange(2, 2500)

# Load HiFast instance
cosmo = HiFast(
    'lcdm',
    timeit=True,
    verbose=True)

# A_S
params = {
    'H0': 68.,
    'Omega_m': 0.3,
    'Omega_b': 0.04,
    'tau_reio': 0.06,
    'ln_A_s_1e10': 3.04,
    'n_s': 0.966,
}

# Class reference spectra
cl_class_As = cosmo.get_cell_from_class(
    ref_ell,
    params,
    name='TT',
    squeeze=False,
    precision=1,
    verbose=True,
    timeit=True)

pk_class_As = cosmo.get_pk_from_class(
    ref_k,
    z,
    params,
    name='m',
    squeeze=False,
    nonlinear=False,
    precision=1,
    verbose=True,
    timeit=True)

# HiFast emulated spectra
cl_emu_As = cosmo.get_cell(
    ref_ell,
    params,
    name='TT',
    squeeze=False,
    verbose=True,
    timeit=True)

pk_emu_As = cosmo.get_pk(
    ref_k,
    z,
    params,
    name='m',
    squeeze=False,
    nonlinear=False,
    verbose=True,
    timeit=True)


# Sigma8_m
params = {
    'H0': 68.,
    'Omega_m': 0.3,
    'Omega_b': 0.04,
    'tau_reio': 0.06,
    'sigma8_m': 0.837413,
    'n_s': 0.966,
}

# Class reference spectra
cl_class_s8 = cosmo.get_cell_from_class(
    ref_ell,
    params,
    name='TT',
    squeeze=False,
    precision=1,
    verbose=True,
    timeit=True)

pk_class_s8 = cosmo.get_pk_from_class(
    ref_k,
    z,
    params,
    name='m',
    squeeze=False,
    nonlinear=False,
    precision=1,
    verbose=True,
    timeit=True)

# HiFast emulated spectra
cl_emu_s8 = cosmo.get_cell(
    ref_ell,
    params,
    name='TT',
    squeeze=False,
    check_params_names=True,
    check_params_values=True,
    verbose=True,
    timeit=True)

pk_emu_s8 = cosmo.get_pk(
    ref_k,
    z,
    params,
    name='m',
    squeeze=False,
    nonlinear=False,
    verbose=False,
    timeit=True)


# Plot relarive differences
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ref_ell, np.abs(cl_emu_s8/cl_class_As-1)*100., label='Emu sigma_8')
plt.plot(ref_ell, np.abs(
    cl_emu_As/cl_class_As-1)*100., '--', label='Emu ln A_s')
plt.plot(ref_ell, np.abs(
    cl_class_s8/cl_class_As-1)*100., ':', label='Class sigma_8')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('ell')
plt.ylabel('Relative difference [%]')
plt.title('C_ell (reference: Class ln A_s)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ref_k, np.abs(pk_emu_s8/pk_class_As-1)*100., label='Emu sigma_8')
plt.plot(ref_k, np.abs(pk_emu_As/pk_class_As-1)*100., '--', label='Emu ln A_s')
plt.plot(ref_k, np.abs(
    pk_class_s8/pk_class_As-1)*100., ':', label='Class sigma_8')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k [h/Mpc]')
plt.ylabel('Relative difference [%]')
plt.title('P(k) (reference: Class ln A_s)')
plt.legend()

plt.tight_layout()
plt.savefig('output/params_conversion.pdf')
