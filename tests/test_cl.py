import emu_like.io as io
import matplotlib.pyplot as plt
import numpy as np
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
    '/ceph/hpc/data/s25r06-05-users/lcdm/sample/cl_100_std.fits')
x_data = fits.get_data('x_data')
cl = fits.get_data('cl_TT_lensed')
ref_cl = fits.get_data('ref_cl_TT_lensed')[0]
ref_ell = fits.get_data('ell_range_cl_TT_lensed')

idx_data = 0
params = {
    'h': x_data[idx_data, 0],
    'Omega_m': x_data[idx_data, 1],
    'Omega_b': x_data[idx_data, 2],
    'ln_A_s_1e10': x_data[idx_data, 3],
    'n_s': x_data[idx_data, 4],
    'tau_reio': x_data[idx_data, 5],
}

cl_data = cl[idx_data] * ref_cl

cl_emu = cosmo.get_cell(
    ref_ell,
    params,
    name='TT',
    squeeze=True,
    timeit=True)

cl_class_0 = cosmo.get_cell_from_class(
    ref_ell,
    params,
    name='TT',
    squeeze=True,
    precision=0,
    timeit=True)

cl_class_1 = cosmo.get_cell_from_class(
    ref_ell,
    params,
    name='TT',
    squeeze=True,
    precision=1,
    timeit=True)

cl_class_2 = cosmo.get_cell_from_class(
    ref_ell,
    params,
    name='TT',
    squeeze=True,
    precision=2,
    timeit=True)

# Plotting
plt.title('Test precision Cell')
plt.plot(ref_ell, np.abs(cl_emu/cl_data - 1.)*100., label='HiFast Emu/Data')
plt.plot(ref_ell, np.abs(cl_class_0/cl_data - 1.)*100., label='Class_0/Data')
plt.plot(ref_ell, np.abs(cl_class_1/cl_data - 1.)*100., label='Class_1/Data')
plt.plot(ref_ell, np.abs(cl_class_2/cl_data - 1.)*100., label='Class_2/Data')
plt.xlabel('ell')
plt.ylabel('abs_rel_diff [%]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('output/test_cl.pdf')
plt.close()
