import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from hi_fast import HiFast

# Load HiFast instance
cosmo = HiFast(
    'lcdm',
    root='/ceph/hpc/home/bellinie/emu_like/output',
    timeit=True,
    verbose=True)

# cosmo.print_cosmo_params()

# Load reference Pk from file
fits = io.FitsFile(
    '/ceph/hpc/data/s25r06-05-users/lcdm/sample/pk_01_thin.fits')
x_data = fits.get_data('x_data')
pk = fits.get_data('pk_m')
ref_pk = fits.get_data('ref_pk_m')
ref_k = fits.get_data('k_range_pk_m')
ref_z = fits.get_data('z_array')

idx_data = 1
params = {
    'h': x_data[idx_data, 1],
    'Omega_m': x_data[idx_data, 2],
    'Omega_b': x_data[idx_data, 3],
    'tau_reio': x_data[idx_data, 4],
    'ln_A_s_1e10': cosmo._emu['pk_m'].class_args['ln_A_s_1e10'],
    'n_s': cosmo._emu['pk_m'].class_args['n_s'],
}
z = x_data[idx_data, 0]
pk_data = pk[idx_data] * interp.make_splrep(ref_z, ref_pk.T, s=0)(z)[:, 0]

pk_class_1 = cosmo.get_pk_from_class(
    ref_k,
    z,
    params,
    name='m',
    squeeze=True,
    precision=1,
    nonlinear=False,
    timeit=True)

# Plotting
plt.title('Test precision Pk at z={:.2f}'.format(z))
# plt.plot(ref_k, np.abs(pk_emu/pk_data - 1.)*100., label='HiFast Emu/Data')
# plt.plot(ref_k, np.abs(pk_class_0/pk_data - 1.)*100., label='Class_0/Data')
plt.plot(ref_k, np.abs(pk_class_1/pk_data - 1.)*100., label='Class_1/Data')
# plt.plot(ref_k, np.abs(pk_class_2/pk_data - 1.)*100., label='Class_2/Data')

plt.xlabel('k [h/Mpc]')
plt.ylabel('perc_rel_diff [%]')
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('output/test_pk.pdf')
plt.close()

exit()
# Test primordial spectrum
params['ln_A_s_1e10'] = 4.0
params['n_s'] = 1.2

pk_emu_new = cosmo.get_pk(
    ref_k,
    z,
    params,
    name='m',
    squeeze=True,
    nonlinear=False,
    timeit=True)

pk_class_new = cosmo.get_pk_from_class(
    ref_k,
    z,
    params,
    name='m',
    squeeze=True,
    precision=1,
    nonlinear=False,
    timeit=True)

# Plotting
plt.title('Test primordial Pk at z={:.2f}'.format(z))
plt.plot(ref_k, np.abs(pk_emu_new/pk_class_new - 1.)*100.,
         label='HiFast Emu/Class diff A_s, n_s')
plt.plot(ref_k, np.abs(pk_emu/pk_class_1 - 1.)*100., '--',
         label='HiFast Emu/Class')
plt.xlabel('k [h/Mpc]')
plt.ylabel('perc_rel_diff [%]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('output/test_pk_primordial.pdf')
plt.close()
