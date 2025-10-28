import matplotlib.pyplot as plt
import numpy as np
from hi_fast import HiFast

sp = HiFast('lcdm', timeit=True, verbose=True)

params = [0., 0.7, 0.3, 0.04, 0.04, 2., 0.06]
params_dict = {name: val
               for name, val in zip(sp._emu['cl_pp'].x_names, params)}
params_dict.pop('z_pk')


k = np.logspace(-4., 1., num=600)
z = np.linspace(0., 1., num=10)
pk1 = sp.get_pk(k, z, params_dict, name='m',
                squeeze=False, nonlinear=False, timeit=True)
pk2 = sp.get_pk(k, z, params_dict, name='m',
                squeeze=True, nonlinear=False, timeit=True)
print(pk1.shape, pk2.shape)


fk = sp.get_fk_from_class(k, z, params_dict, name='m',
                          precision=0, nonlinear=False, timeit=True)

k = 0.1
z = np.linspace(0., 10., num=30)
fk = sp.get_fk_from_class(k, z, params_dict, name='m',
                          precision=0, nonlinear=False, timeit=True)

# fact = ell*(ell+1.)/2/np.pi
# T_cmb = 2.7255
# fact2 = (T_cmb*1.e6)**2.
# plt.plot(ell, fact2*cell)
# plt.xscale('log')

# plt.plot(k, pk)
# plt.xscale('log')
# plt.yscale('log')

plt.plot(z, fk[0])
# plt.xscale('log')

plt.savefig('test.pdf')
