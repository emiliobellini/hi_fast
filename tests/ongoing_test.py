import matplotlib.pyplot as plt
import numpy as np
from hi_fast import HiFast

sp = HiFast('lcdm', timeit=True, verbose=True)

params = [0., 0.7, 0.3, 0.05, 0.05, 2., 0.06]
params_dict = {name: val for name, val in zip(sp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')

k = np.logspace(-4., 1., num=600)
z = 0.
ell = np.arange(2, 2500)
# cell = sp.get_cell_from_class(ell, params_dict, name='BB', precision=0, timeit=True)
# pk = sp.get_pk_from_class(k, z, params_dict, name='weyl', precision=0, nonlinear=False, timeit=True)
fk = sp.get_fk_from_class(k, z, params_dict, name='m', precision=0, nonlinear=False, timeit=True)

# fact = ell*(ell+1.)/2/np.pi
# T_cmb = 2.7255
# fact2 = (T_cmb*1.e6)**2.
# plt.plot(ell, fact2*cell)
# plt.xscale('log')

# plt.plot(k, pk)
# plt.xscale('log')
# plt.yscale('log')

plt.plot(k, fk)
plt.xscale('log')

plt.savefig('test.pdf')


    # class_fprintf_double(clfile, factor*pow(pba->T_cmb*1.e6,2)*cl[phr->index_ct_bb], phr->has_bb);
    # class_fprintf_double(clfile, sqrt(l*(l+1))*factor*pba->T_cmb*1.e6*cl[phr->index_ct_tp], phr->has_tp);
    # class_fprintf_double(clfile, sqrt(l*(l+1))*factor*pba->T_cmb*1.e6*cl[phr->index_ct_ep], phr->has_ep);

exit()

# params_dict['ppp'] = 4.

z = np.linspace(0., 10., num=6)

ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)

params = [0., 0.7, 0.3, 0.05, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)

params = [0., 0.6, 0.3, 0.05, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)

params = [0., 0.6, 0.3, 0.04, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)


k = np.logspace(-4., 1., num=300)
z = np.linspace(0., 10., num=4)

params = [0., 0.7, 0.3, 0.05, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)

params = [0., 0.6, 0.3, 0.05, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)

params = [0., 0.6, 0.3, 0.04, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk(k, z, params_dict, name='m', nonlinear=False, timeit=True)


params = [0., 0.6, 0.3, 0.04, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_fk(k, z, params_dict, name='m', nonlinear=False, timeit=True)


params = [0., 0.6, 0.3, 0.04, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_cell([2, 3], params_dict, name='TT', timeit=True)


params = [0., 0.7, 0.3, 0.05, 0.05, 2., 0.16]
params_dict = {name: val for name, val in zip(ppp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')
ppp.get_pk_from_class(k, z, params_dict, name='m', precision=0, nonlinear=False, timeit=True)
ppp.get_pk_from_class(k, z, params_dict, name='cb', precision=0, nonlinear=False, timeit=True)
ppp.get_pk_from_class(k, z, params_dict, name='weyl', precision=0, nonlinear=False, timeit=True)


ppp.get_fk_from_class(k, z, params_dict, name='m', precision=0, nonlinear=False, timeit=True)
ppp.get_fk_from_class(k, z, params_dict, name='cb', precision=0, nonlinear=False, timeit=True)
ppp.get_fk_from_class(k, z, params_dict, name='weyl', precision=0, nonlinear=False, timeit=True)


ppp.get_cell_from_class([2, 3], params_dict, name='TT', precision=0, timeit=True)
ppp.get_cell_from_class([2, 3], params_dict, name='pp', precision=0, timeit=True)
