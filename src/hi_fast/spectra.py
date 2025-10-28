import classy
import numpy as np
import scipy.interpolate as interp
import hi_fast.io as io


# ------------------- Spectrum -----------------------------------------------#

class Spectrum(object):

    def __init__(self, **kwargs):
        # Init common attributes
        self.name = kwargs['name']
        self.x_names = kwargs['x_names']
        self.x_ranges = kwargs['x_ranges']
        self.x_scaler = kwargs['x_scaler']
        self.y_scaler = kwargs['y_scaler']
        self.x_pca = kwargs['x_pca']
        self.y_pca = kwargs['y_pca']
        self.model = kwargs['model']
        self.class_vars = kwargs['class_vars']
        self.class_args = kwargs['class_args']
        self.params_cosmo = [x for x in self.x_names if x != 'z_pk']
        if np.all(kwargs['ref'] == 1):
            self.ref = None
        else:
            self.ref = kwargs['ref']

        # Placeholders
        self.has_k_and_z = False
        self.has_ell = False
        pass

    def _to_numpy_array(self, x):
        if isinstance(x, float):
            x = np.array([x])
        elif isinstance(x, int):
            x = np.array([float(x)])
        elif isinstance(x, list):
            x = np.array(x)
        return x

    def _check_param_names(self, params):
        """
        Check that the parameters passed are
        exactly those expected.
        """
        if set(list(params.keys())) != set(self.params_cosmo):
            raise Exception(
                'I expected parameters {}, and I got {}!'
                ''.format(self.x_names, params.keys()))
        return

    def _check_param_values(self, params):
        """
        Check that the parameters are within the emulator range.
        """
        for name in params:
            low = self.class_vars[name]['prior']['min']
            high = self.class_vars[name]['prior']['max']
            in_range = low <= params[name] <= high
            if not in_range:
                raise Exception(
                    'Parameter {} = {} out of range [{} - {}]'
                    ''.format(name, params[name], low, high))
        return

    def _is_same_array(self, x, x_ref):
        """
        Check if an array k, or z, are the same as the ones stored
        to avoid reinterpolation of reference spectra.
        """
        if x_ref is None:
            return False
        else:
            return np.array_equal(x, x_ref)

    @staticmethod
    def choose_one(**kwargs):
        """
        Main function to get the correct Spectrum.

        Arguments:
            - spectrum_type (str): type of spectrum.

        Return:
            - Spectrum (object): get the correct
              spectrum and initialize it.
        """
        # Pk
        if kwargs['name'].startswith('pk_'):
            return Pk(**kwargs)
        # Growth rates
        elif kwargs['name'].startswith('f_'):
            return Pk(**kwargs)
        # Cl
        elif kwargs['name'].startswith('cl_'):
            return Cell(**kwargs)
        else:
            raise ValueError(
                'Spectrum {} not recognized!'.format(kwargs['name']))

    @io.timeit
    def _eval_emu(self, x, timeit=False):
        """
        Evaluate the emulator at a given point.
        Arguments:
        - x (array or list): these are the input parameters.
          They can be passed as an array or as a dictionary
          with the names of x as keys.
        It returns the value(s) for y
        """

        # Scale x
        if self.x_scaler is None:
            x_scaled = np.array([x])
        else:
            x_scaled = self.x_scaler.transform(np.array([x]))

        # PCA x
        if self.x_pca is None:
            x_scaled_pca = x_scaled
        else:
            x_scaled_pca = self.x_pca.transform(x_scaled)

        # Emulate y
        y_scaled_pca = self.model(x_scaled_pca, training=False)

        # inverse PCA y
        if self.y_pca is None:
            y_scaled = y_scaled_pca
        else:
            y_scaled = self.y_pca.inverse_transform(y_scaled_pca)

        # Scale back y
        if self.y_scaler is None:
            y = y_scaled
        else:
            y = self.y_scaler.inverse_transform(y_scaled)[0]

        return y


# ------------------- Pk -----------------------------------------------------#

class Pk(Spectrum):

    def __init__(self, **kwargs):
        Spectrum.__init__(self, **kwargs)
        # Init specific attributes
        self.has_k_and_z = True
        self.z_array = kwargs['z_array']
        self.k_array = kwargs['k_array']

        # Placeholders
        self.k_stored = None
        self.z_stored = None
        self.ref_stored = None
        pass

    def _check_k_values(self, k):
        """
        Check that k is within the emulator range.
        """
        low = self.k_array.min()
        high = self.k_array.max()
        if k.min() < low or k.max() > high:
            raise Exception(
                'k (h/Mpc) = [{} - {}] out of range [{} - {}]'.format(
                    k.min(), k.max(), low, high))
        return

    def _check_z_values(self, z):
        """
        Check that z is within the emulator range.
        """
        low = self.class_vars['z_pk']['prior']['min']
        high = self.class_vars['z_pk']['prior']['max']
        if z.min() < low or z.max() > high:
            raise Exception(
                'z = [{} - {}] out of range [{} - {}]'.format(
                    z.min(), z.max(), low, high))
        return

    def get(self, k, z, params):
        """
        Main method to get the power spectrum P(k, z) or the
        growth rate f(k, z) = dln P(k, z)/dln a at some z and k.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        NOTE: If k and z remains unchanged, i.e. we are varying only
        the parameters, the computation is much faster, since there
        is not need to interpolate the reference spectra (for the
        spectra we emulate the ratio) again.
        """

        # Check parameters
        # 1) Parameter names: no missing and not unrecognized
        self._check_param_names(params)
        # 2) Parameter values: not outside the ranges
        self._check_param_values(params)

        # Check k
        k = self._to_numpy_array(k)
        # 1) is the same as stored?
        same_k = self._is_same_array(k, self.k_stored)
        # 2) is inside the emulated ranges?
        if not same_k:
            self._check_k_values(k)

        # Check z
        z = self._to_numpy_array(z)
        # 1) is the same as stored?
        same_z = self._is_same_array(z, self.z_stored)
        # 2) is inside the emulated ranges?
        if not same_z:
            self._check_z_values(k)

        # If z or k changed, reinterpolate reference
        if not same_k or not same_z:
            self.k_stored = k
            self.z_stored = z
            # This is done only if the emulator emulates the ratio
            if self.ref is not None:
                ref = interp.make_splrep(self.z_array, self.ref.T, s=0)(z)
                self.ref_stored = interp.make_splrep(
                    self.k_array, ref.T, s=0)(k)

        # Prepare parameters list
        params_emu = [params[name] if (name != 'z_pk') else None
                      for name in self.x_names]
        idx_z_pk = self.x_names.index('z_pk')

        # Init output
        out_emu = np.zeros((len(self.k_array), len(self.z_stored)))

        # Iterate over each redshift and evaluate emulator
        for nz, z_one in enumerate(self.z_stored):
            params_emu[idx_z_pk] = z_one
            # Evaluate emulator
            out_emu[:, nz] = self._eval_emu(params_emu)

        # Interpolate at the correct k
        out_emu = interp.make_splrep(self.k_array, out_emu, s=0)(self.k_stored)

        # Multiply by reference
        if self.ref is not None:
            out = out_emu * self.ref_stored
        else:
            out = out_emu

        return out

    def get_from_class(self, k, z, params, precision=0):
        """
        Main method to get the power spectrum P(k, z) or the
        growth rate f(k, z) = dln P(k, z)/dln a at some z and k
        from Class.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters;
        - precision (0, 1, 2, or dict): for default precisions use:
            - 0: standard class precision;
            - 1: precision parameters used for this emulator;
            - 2: high precision parameters.
          Eitherwise, it is possible to pass directly a
          dictionary of precision parameters.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        NOTE: If k and z remains unchanged, i.e. we are varying only
        the parameters, the computation is much faster, since there
        is not need to interpolate the reference spectra (for the
        spectra we emulate the ratio) again.
        """

        k = self._to_numpy_array(k)
        z = self._to_numpy_array(z)

        # Define high precision parameters
        high_prec = {
            'k_per_decade_for_pk': 1000,
            'k_per_decade_for_bao': 2000,
            'l_logstep': 1.026,
            'l_linstep': 25,
            'perturbations_sampling_stepsize': 0.01,
            'l_switch_limber': 20,
            'accurate_lensing': 1,
            'delta_l_max': 1000,
            'k_max_tau0_over_l_max': 8,
            }
        # Get additional Class arguments needed to run smoothly
        class_args = {n: self.class_args[n] for n in self.class_args
                      if n not in high_prec}
        class_args['output'] = 'mPk, dTk'
        class_args['P_k_max_h/Mpc'] = k.max()
        class_args['z_max_pk'] = max(z.max(), 0.1)

        # Fix precision parameters
        if isinstance(precision, int):
            if precision == 0:
                prec = {}
            elif precision == 1:
                prec = {n: self.class_args[n] for n in self.class_args
                        if n in high_prec}
            elif precision == 2:
                prec = high_prec
            else:
                raise Exception('precision can be 0, 1, 2 or a dictionary!')
        elif isinstance(precision, dict):
            prec = precision
        else:
            raise Exception('precision can be 0, 1, 2 or a dictionary!')

        # Compute
        cosmo = classy.Class()
        cosmo.set(params | class_args | prec)
        cosmo.compute()

        # convert k in units of 1/Mpc
        k = k * cosmo.h()

        # Get correct spectrum
        if self.name.endswith('_m'):
            pk_out, k_out, z_out = cosmo.get_pk_and_k_and_z(
                nonlinear=False,
                only_clustering_species=False,
                h_units=False)
        elif self.name.endswith('_cb'):
            pk_out, k_out, z_out = cosmo.get_pk_and_k_and_z(
                nonlinear=False,
                only_clustering_species=True,
                h_units=False)
        elif self.name.endswith('_weyl'):
            pk_out, k_out, z_out = cosmo.get_Weyl_pk_and_k_and_z(
                nonlinear=False,
                h_units=False)

        # Flip z_array (for the interpolation it has to be increasing)
        z_out = np.flip(z_out)
        pk_out = np.flip(pk_out, axis=1)
        pk_out = interp.make_splrep(k_out, pk_out, s=0)(k)
        pk = interp.make_splrep(z_out, pk_out.T, s=0)(z).T

        if self.name.startswith('pk_'):
            # The output is in units Mpc**3 and I want (Mpc/h)**3.
            return pk*cosmo.h()**3.
        elif self.name.startswith('f_'):
            # Calculate derivative if growth rate f
            dpkdz = interp.make_splrep(z_out, pk_out.T, s=0).derivative()(z).T
            fk = -0.5 * (1+z) * dpkdz/pk
            return fk


# ------------------- Cell ---------------------------------------------------#

class Cell(Spectrum):

    def __init__(self, **kwargs):
        Spectrum.__init__(self, **kwargs)
        # Init specific attributes
        self.has_ell = True
        self.ell_array = kwargs['ell_array']

        # Placeholders
        self.ell_stored = None
        self.ell_stored_indices = None
        self.ref_stored = None
        pass

    def _check_ell_values(self, ell):
        """
        Check that ell is within the emulator range.
        """
        low = self.ell_array.min()
        high = self.ell_array.max()
        if ell.min() < low or ell.max() > high:
            raise Exception(
                'ell = [{} - {}] out of range [{} - {}]'.format(
                    ell.min(), ell.max(), low, high))
        return

    def _to_numpy_array(self, x):
        x = Spectrum._to_numpy_array(self, x)
        return x.astype(int)

    def get(self, ell, params):
        """
        Main method to get the Cell(ell).
        Arguments:
        - ell (float, list or array): single/list of wavenumbers;
        - params (dict): dictionary with the cosmo parameters.
        """

        # Check parameters
        # 1) Parameter names: no missing and not unrecognized
        self._check_param_names(params)
        # 2) Parameter values: not outside the ranges
        self._check_param_values(params)

        # Check k
        ell = self._to_numpy_array(ell)
        # 1) is the same as stored?
        same_ell = self._is_same_array(ell, self.ell_stored)
        # 2) is inside the emulated ranges?
        if not same_ell:
            self._check_ell_values(ell)

        # If ell range changed, re-cut reference
        if not same_ell:
            self.ell_stored = ell
            self.ell_stored_indices = np.where(np.isin(self.ell_array, ell))[0]
            # This is done only if the emulator emulates the ratio
            # NOTE: For the Cells we could have just extracted the elements,
            # but it was quicker to just copy/past from Pk

            if self.ref is not None:
                self.ref_stored = self.ref[self.ell_stored_indices]

        # Prepare parameters list
        params_emu = [params[name] if (name != 'z_pk') else 0.
                      for name in self.x_names]

        # Evaluate emulator
        out_emu = self._eval_emu(params_emu)[self.ell_stored_indices]

        # Multiply by reference
        if self.ref is not None:
            out = out_emu * self.ref_stored
        else:
            out = out_emu

        return out

    def get_from_class(self, ell, params, precision=0):
        """
        Main method to get the Cell(ell) from Class.
        Arguments:
        - ell (float, list or array): single/list of wavenumbers;
        - params (dict): dictionary with the cosmo parameters.
        - precision (0, 1, 2, or dict): for default precisions use:
            - 0: standard class precision;
            - 1: precision parameters used for this emulator;
            - 2: high precision parameters.
          Eitherwise, it is possible to pass directly a
          dictionary of precision parameters.
        """

        ell = self._to_numpy_array(ell)

        # Define high precision parameters
        high_prec = {
            'k_per_decade_for_pk': 1000,
            'k_per_decade_for_bao': 2000,
            'l_logstep': 1.026,
            'l_linstep': 25,
            'perturbations_sampling_stepsize': 0.01,
            'l_switch_limber': 20,
            'accurate_lensing': 1,
            'delta_l_max': 1000,
            'k_max_tau0_over_l_max': 8,
            }
        # Get additional Class arguments needed to run smoothly
        class_args = {n: self.class_args[n] for n in self.class_args
                      if n not in high_prec}
        class_args['output'] = 'tCl, pCl, lCl'
        class_args['l_max_scalars'] = ell.max()
        class_args['lensing'] = 'yes'

        # Fix precision parameters
        if isinstance(precision, int):
            if precision == 0:
                prec = {}
            elif precision == 1:
                prec = {n: self.class_args[n] for n in self.class_args
                        if n in high_prec}
            elif precision == 2:
                prec = high_prec
            else:
                raise Exception('precision can be 0, 1, 2 or a dictionary!')
        elif isinstance(precision, dict):
            prec = precision
        else:
            raise Exception('precision can be 0, 1, 2 or a dictionary!')

        # Compute
        cosmo = classy.Class()
        cosmo.set(params | class_args | prec)
        cosmo.compute()

        # Get Cells
        cl_type = self.name.split('_')[1].lower()
        try:
            out = cosmo.lensed_cl(lmax=ell.max())[cl_type]
        except KeyError:
            out = cosmo.raw_cl(lmax=ell.max())[cl_type]

        # Mask Cells
        ell_out = np.arange(ell.max()+1)
        idx = np.where(np.isin(ell_out, ell))[0]
        out = out[idx]

        # Normalize Cells
        out *= ell*(ell+1.)/2./np.pi

        return out
