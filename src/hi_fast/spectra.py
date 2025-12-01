import classy
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as optimize
from . import io as io
from .params import Params
from .pca import PCA
from .scalers import Scaler


# ------------------- Spectrum -----------------------------------------------#

class Spectrum(object):

    def __init__(self, **kwargs):
        # Name of the spectrum (str)
        self.name = kwargs['name']
        # Parameters names needed by the emulator (list of str)
        self.x_names = kwargs['x_names']
        # Placeholder for values needed by the emulator (list of floats)
        self.x_values = [None for _ in self.x_names]
        # Parameters ranges (list of list of floats)
        self.x_ranges = kwargs['x_ranges']
        # Scalers used by the emulator. TODO: implement portability
        self.x_scaler = Scaler.choose_one(**kwargs['x_scaler'])
        self.y_scaler = Scaler.choose_one(**kwargs['y_scaler'])
        # PCA used by the emulator. TODO: implement portability
        self.x_pca = PCA(**kwargs['x_pca'])
        self.y_pca = PCA(**kwargs['y_pca'])
        # Emulator model
        self.model = kwargs['model']
        # Dictionary for reference parameters and spectra. Keys:
        # - params (dict): Class parameters used to calculate the spectrum
        # - spectrum (array): reference spectrum, binned with
        #     ref_k, ref_z or ref_ell
        self.ref = {
            'params': kwargs['ref_params'],
            'spectrum': None if np.all(
                kwargs['ref_spectrum'] == 1) else kwargs['ref_spectrum'],
        }
        # Arguments used by Class to generate the training/validation datasets
        self.class_args = kwargs['class_args']

        # Define high precision parameters for Class
        self.class_high_prec = {
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

        # Placeholder for input parameters (list). These are the parameters
        # that the user should provide in order to correctly evaluate the
        # spectrum. Not all the parameters used by the emulator go here (e.g.
        # z_pk, which is given as an argument of the Spectrum.get method). In
        # addition, tt contains the parameters that are used by external
        # routines (e.g. ln_A_s_1e10, n_s, k_pivot to calculate the primordial
        # power spectrum). Check the params.Params.add_defaults method to
        # verify which are the ones that are not strictly necessary as a
        # default value has been specified.
        self.input_params_names = []

        pass

    def _to_numpy_array(self, x):
        if isinstance(x, float):
            x = np.array([x])
        elif isinstance(x, int):
            x = np.array([float(x)])
        elif isinstance(x, list):
            x = np.array(x)
        return x

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
        elif kwargs['name'].startswith('fk_'):
            return Fk(**kwargs)
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

    def _get_input_params_class(
            self, params, precision, class_args, verbose=False):

        params_in = params.copy()

        # Fix precision parameters
        if isinstance(precision, int):
            if precision == 0:
                prec = {}
            elif precision == 1:
                prec = {n: self.class_args[n] for n in self.class_args
                        if n in self.class_high_prec}
            elif precision == 2:
                prec = self.class_high_prec
            else:
                raise Exception('precision can be 0, 1, 2 or a dictionary!')
        elif isinstance(precision, dict):
            prec = precision
        else:
            raise Exception('precision can be 0, 1, 2 or a dictionary!')

        if verbose:
            verb_params = {
                'input_verbose': 1,
                'background_verbose': 1,
                'thermodynamics_verbose': 1,
                'perturbations_verbose': 1,
                'transfer_verbose': 1,
                'primordial_verbose': 1,
                'harmonic_verbose': 1,
                'fourier_verbose': 1,
                'lensing_verbose': 1,
                'distortions_verbose': 1,
                'output_verbose': 1,
            }
        else:
            verb_params = {}

        # Fix parameters
        renaming_rules = [
            # hi_fast_name, class_name
            ('sigma8_cb', 'sigma8'),
        ]

        for hi_fast_name, class_name in renaming_rules:
            if hi_fast_name in params_in:
                params_in[class_name] = params_in.pop(hi_fast_name)

        # Remove cosmo parameters from class_args if they are in params_in
        if 'sigma8' in params_in and 'ln_A_s_1e10' in class_args:
            class_args.pop('ln_A_s_1e10')
        if 'A_s' in params_in and 'ln_A_s_1e10' in class_args:
            class_args.pop('ln_A_s_1e10')

        all_params = verb_params | class_args | prec | params_in

        return all_params


# ------------------- Pk -----------------------------------------------------#

class Pk(Spectrum):

    def __init__(self, **kwargs):
        Spectrum.__init__(self, **kwargs)

        # Input parameters (primordial power spectrum)
        self.input_params_names = [nm for nm in self.x_names if nm != 'z_pk']
        self.input_params_names += ['ln_A_s_1e10', 'n_s', 'k_pivot']

        # Add ref_z and ref_k to ref dictionary
        self.ref['z'] = kwargs['ref_z']
        self.ref['k'] = kwargs['ref_k']

        # Store min and max
        self.k_min = np.min(self.ref['k'])
        self.k_max = np.max(self.ref['k'])
        idx_z_pk = self.x_names.index('z_pk')
        self.z_min, self.z_max = self.x_ranges[idx_z_pk]

        # Placeholder for stored spectra. This is to avoid
        # interpolating multiple times the reference spectra
        # if k and z arrays do not change.
        self.stored = {
            'k': None,
            'z': None,
            'ref_spectrum': None,
        }

        pass

    def _check_k_values(self, k):
        """
        Check that k is within the emulator range.
        """
        if k.min() < self.k_min or k.max() > self.k_max:
            raise Exception(
                'k (h/Mpc) = [{} - {}] out of range [{} - {}]'.format(
                    k.min(), k.max(), self.k_min, self.k_max))
        return

    def _check_z_values(self, z):
        """
        Check that z is within the emulator range.
        """
        if z.min() < self.z_min or z.max() > self.z_max:
            raise Exception(
                'z = [{} - {}] out of range [{} - {}]'.format(
                    z.min(), z.max(), self.z_min, self.z_max))
        return

    def _sigma_R_integral(self, k, pk, R):
        """
        Compute sigma_R integral.
        Arguments:
        - k (array): wavenumbers in h/Mpc;
        - pk (array): power spectrum at k in (Mpc/h)^3;
        - R (float): smoothing scale in Mpc/h.
        Returns:
        - sigma_R (float): sigma_R value.
        """

        # Window function
        x = k * R
        W = 3. * (np.sin(x) - x * np.cos(x)) / x**3.

        integrand = k**2. * pk * W**2.

        sigma_R = np.sqrt(
            np.trapz(integrand, x=k) / (2. * np.pi**2.))

        return sigma_R

    def _get_primordial_pk(self, ln_A_s_1e10, n_s, k_pivot, k):
        """
        Get primordial power spectrum at given k.
        Arguments:
        - ln_A_s_1e10 (float): log amplitude of primordial spectrum;
        - n_s (float): spectral index;
        - k_pivot (float): pivot scale in h/Mpc;
        - k (array): wavenumbers in h/Mpc.
        Returns:
        - P_primordial (array): primordial power spectrum at k.
        """

        log_P_1e10 = ln_A_s_1e10 + (n_s-1.)*np.log(k/k_pivot)
        P_primordial = np.exp(log_P_1e10) / 1e10

        return P_primordial

    def _get_weyl_pk(self, k, z, n_k, n_z, n_mu):
        """
        Get nonlinear Weyl power spectrum from Class.
        Arguments:
        - k (array): wavenumbers in 1/Mpc. To be consistent with
          the other calls, it has to be 3D;
        - z (array): redshifts;
        - n_k (int): number of k values;
        - n_z (int): number of z values;
        - n_mu (int): number of mu values.
        Returns:
        - P_weyl (array): Weyl power spectrum at k and z.
        """

        # TODO: for now we just call the linear one
        pk = self._get_weyl_pk_lin(k, z, n_k, n_z, n_mu)

        return pk

    def _get_weyl_pk_lin(self, k, z, n_k, n_z, n_mu):
        """
        Get linear Weyl power spectrum from Class.
        Arguments:
        - k (array): wavenumbers in 1/Mpc;
        - z (array): redshifts;
        - n_k (int): number of k values;
        - n_z (int): number of z values;
        - n_mu (int): number of mu values.
        Returns:
        - P_weyl (array): Weyl power spectrum at k and z.
        """
        pk_array, k_array, z_array = self.cosmo.get_Weyl_pk_and_k_and_z(
            nonlinear=False,
            h_units=False
        )

        # Flip z_array (for the interpolation it has to be increasing)
        z_array = np.flip(z_array)
        pk_array = np.flip(pk_array, axis=1)

        # Evaluate pk at the requested range
        pk = np.zeros((n_k, n_z, n_mu))
        pk_int = interp.make_splrep(k_array, pk_array, s=0)(k[:, 0, 0])
        for nzval, zval in enumerate(z):
            pk[:, nzval, 0] = interp.make_splrep(z_array, pk_int.T, s=0)(zval)

        return pk

    def _get_pk_or_fk_common(self, k, z, params, nonlinear=False):
        """
        Common steps to get pk or fk from the emulator
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        """

        # Check k
        k = self._to_numpy_array(k)
        # 1) is the same as stored?
        same_k = self._is_same_array(k, self.stored['k'])
        # 2) is inside the emulated ranges?
        if not same_k:
            self._check_k_values(k)

        # Check z
        z = self._to_numpy_array(z)
        # 1) is the same as stored?
        same_z = self._is_same_array(z, self.stored['z'])
        # 2) is inside the emulated ranges?
        if not same_z:
            self._check_z_values(z)

        # If z or k changed, reinterpolate reference
        if not same_k or not same_z:
            self.stored['k'] = k
            self.stored['z'] = z
            # This is done only if the emulator emulates the ratio
            if self.ref['spectrum'] is not None:
                ref = interp.make_splrep(
                    self.ref['z'], self.ref['spectrum'].T, s=0)(z)
                self.ref_spectrum_stored = interp.make_splrep(
                    self.ref['k'], ref.T, s=0)(k)

        # Init output
        out_emu = np.zeros((len(self.ref['k']), len(self.stored['z'])))

        # Iterate over each redshift and evaluate emulator
        for nz, z_one in enumerate(self.stored['z']):
            # Get parameters list for emulator
            self.x_values = params.get_values_emu(z_pk=z_one)
            # Evaluate emulator
            out_emu[:, nz] = self._eval_emu(self.x_values)

        # Interpolate at the correct k
        out_emu = interp.make_splrep(
            self.ref['k'], out_emu, s=0)(self.stored['k'])

        # Multiply by reference
        if self.ref['spectrum'] is not None:
            out = out_emu * self.ref_spectrum_stored
        else:
            out = out_emu

        return out, params

    def get(self, k, z, params, nonlinear=False):
        """
        Main method to get the power spectrum P(k, z) or the
        growth rate f(k, z) = dln P(k, z)/dln a at some z and k.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Get emulator output
        out, params = self._get_pk_or_fk_common(
            k, z, params, nonlinear=nonlinear)

        # Adjust shape with primordial Pk
        ref_primordial = self._get_primordial_pk(
            self.ref['params']['ln_A_s_1e10'],
            self.ref['params']['n_s'],
            self.ref['params']['k_pivot'],
            self.stored['k']*self.ref['params']['h'])
        primordial = self._get_primordial_pk(
            params._out['ln_A_s_1e10'],
            params._out['n_s'],
            params._out['k_pivot'],
            self.stored['k']*params._out['h'])
        out *= primordial[:, np.newaxis] / ref_primordial[:, np.newaxis]

        return out

    def get_from_class(
            self, k, z, params, nonlinear=False, precision=0, verbose=False):
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

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        k = self._to_numpy_array(k)
        z = self._to_numpy_array(z)

        # Get additional Class arguments needed to run smoothly
        class_args = {n: self.class_args[n] for n in self.class_args
                      if n not in self.class_high_prec}
        class_args['output'] = 'tCl, pCl, lCl, mPk, dTk'
        class_args['P_k_max_h/Mpc'] = k.max()
        class_args['z_max_pk'] = max(z.max(), 0.1)

        # Prepare parameters list
        params = self._get_input_params_class(
            params, precision, class_args, verbose=verbose)

        # Compute
        self.cosmo = classy.Class()
        self.cosmo.set(params)
        self.cosmo.compute()

        # Get correct spectrum
        if nonlinear is True and self.name.endswith('_cb'):
            fun = self.cosmo.get_pk_cb
        elif nonlinear is True and self.name.endswith('_m'):
            fun = self.cosmo.get_pk
        elif nonlinear is True and self.name.endswith('_weyl'):
            fun = self._get_weyl_pk
        elif nonlinear is False and self.name.endswith('_cb'):
            fun = self.cosmo.get_pk_cb_lin
        elif nonlinear is False and self.name.endswith('_m'):
            fun = self.cosmo.get_pk_lin
        elif nonlinear is False and self.name.endswith('_weyl'):
            fun = self._get_weyl_pk_lin

        # convert k in units of 1/Mpc
        n_mu = 1
        n_z = len(z)
        n_k = len(k)
        k_3D = np.broadcast_to(
            k[:, np.newaxis, np.newaxis], (n_k, n_z, n_mu)) * self.cosmo.h()
        out = fun(k_3D, z, n_k, n_z, n_mu) * self.cosmo.h()**3.
        out = out[:, :, 0]

        return out

    def get_sigma_R(self, R, z, params, nonlinear=False):
        """
        Main method to get sigma_R(R, z).
        Arguments:
        - R (float): smoothing scale in Mpc/h;
        - z (float): redshift;
        - params (dict): dictionary with the cosmo parameters.

        NOTE: P(k, z) is in units of (Mpc/h)^3.
        """

        # Use the reference k array
        k = self.ref['k']

        # Get P(k, z)
        pk = self.get(k, z, params, nonlinear=nonlinear)[:, 0]

        # Compute sigma_R
        sigma_R = self._sigma_R_integral(k, pk, R)

        return sigma_R

    def get_As_from_sigma_8(self, sigma8_cb, params):
        """
        Get ln_A_s_1e10 from sigma8_cb by shooting method.
         Arguments:
         - params (dict): dictionary with the cosmo parameters.
         - sigma8_cb (float): target sigma8_cb value.
         Returns:
         - ln_A_s_1e10 (float): corresponding ln_A_s_1e10 value.
         """

        params_in = params.copy()
        params_in.pop('sigma8_cb')
        params_in = Params._add_defaults(
            params_in, required_params=self.input_params_names)

        def _fun(ln_A_s_1e10):
            params_in['ln_A_s_1e10'] = ln_A_s_1e10
            params_obj = Params(params_in, self)
            sigma8_cb_new = self.get_sigma_R(8, 0.0, params_obj)
            return sigma8_cb_new - sigma8_cb

        # Find root
        opt_ln_A_s_1e10 = optimize.root(_fun, 3.0).x[0]
        return opt_ln_A_s_1e10


# ------------------- Pk -----------------------------------------------------#

class Fk(Pk):

    def __init__(self, **kwargs):
        Pk.__init__(self, **kwargs)

        # Input parameters
        self.input_params_names = [nm for nm in self.x_names if nm != 'z_pk']

        pass

    def get(self, k, z, params, nonlinear=False):
        """
        Main method to get the growth rate f(k, z) = dln P(k, z)/dln a
        at some z and k. Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters.

        NOTE: k is in units of h/Mpc.
        NOTE: If k and z remains unchanged, i.e. we are varying only
        the parameters, the computation is much faster, since there
        is not need to interpolate the reference spectra (for the
        spectra we emulate the ratio) again.
        """

        out, _ = self._get_pk_or_fk_common(k, z, params, nonlinear=nonlinear)

        return out

    def get_from_class(
            self, k, z, params, nonlinear=False, precision=0, verbose=False):
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
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        k = self._to_numpy_array(k)
        z = self._to_numpy_array(z)

        # Get additional Class arguments needed to run smoothly
        class_args = {n: self.class_args[n] for n in self.class_args
                      if n not in self.class_high_prec}
        class_args['output'] = 'tCl, pCl, lCl, mPk, dTk'
        class_args['P_k_max_h/Mpc'] = k.max()
        class_args['z_max_pk'] = max(z.max(), 0.1)

        # Prepare parameters list
        params = self._get_input_params_class(
            params, precision, class_args, verbose=verbose)

        # Compute
        self.cosmo = classy.Class()
        self.cosmo.set(params)
        self.cosmo.compute()

        # Get correct spectrum
        if self.name.endswith('_cb'):
            pk_array, k_array, z_array = self.cosmo.get_pk_and_k_and_z(
                nonlinear=nonlinear,
                only_clustering_species=True,
                h_units=False)
        elif self.name.endswith('_m'):
            pk_array, k_array, z_array = self.cosmo.get_pk_and_k_and_z(
                nonlinear=nonlinear,
                only_clustering_species=False,
                h_units=False)
        elif self.name.endswith('_weyl'):
            pk_array, k_array, z_array = self.cosmo.get_Weyl_pk_and_k_and_z(
                nonlinear=nonlinear,
                h_units=False)

        # Flip z_array (for the interpolation it has to be increasing)
        z_array = np.flip(z_array)
        pk_array = np.flip(pk_array, axis=1)

        k_array /= self.cosmo.h()

        # Evaluate pk at the requested range
        pk = interp.make_splrep(k_array, pk_array, s=0)(k)
        pk_at_z = interp.make_splrep(z_array, pk.T, s=0)(z).T
        dpkdz = interp.make_splrep(z_array, pk.T, s=0).derivative()(z).T

        out = -0.5 * (1+z) * dpkdz/pk_at_z
        return out


# ------------------- Cell ---------------------------------------------------#

class Cell(Spectrum):

    def __init__(self, **kwargs):
        Spectrum.__init__(self, **kwargs)

        # Input parameters (primordial power spectrum)
        self.input_params_names = self.x_names.copy()

        # Add ref_ell to ref dictionary
        self.ref['ell'] = kwargs['ref_ell']

        # Store min and max
        self.ell_min = np.min(self.ref['ell'])
        self.ell_max = np.max(self.ref['ell'])

        # Placeholder for stored spectra. This is to avoid
        # interpolating multiple times the reference spectra
        # if k and z arrays do not change.
        self.stored = {
            'ell': None,
            'ell_indices': None,
            'ref_spectrum': None,
        }

        pass

    def _check_ell_values(self, ell):
        """
        Check that ell is within the emulator range.
        """
        if ell.min() < self.ell_min or ell.max() > self.ell_max:
            raise Exception(
                'ell = [{} - {}] out of range [{} - {}]'.format(
                    ell.min(), ell.max(), self.ell_min, self.ell_max))
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

        # Check ell
        ell = self._to_numpy_array(ell)
        # 1) is the same as stored?
        same_ell = self._is_same_array(ell, self.stored['ell'])
        # 2) is inside the emulated ranges?
        if not same_ell:
            self._check_ell_values(ell)

        # If ell range changed, re-cut reference
        if not same_ell:
            self.stored['ell'] = ell
            self.stored['ell_indices'] = np.where(
                np.isin(self.ref['ell'], ell))[0]
            # This is done only if the emulator emulates the ratio
            # NOTE: For the Cells we could have just extracted the elements,
            # but it was quicker to just copy/past from Pk

            if self.ref['spectrum'] is not None:
                self.ref_spectrum_stored = self.ref['spectrum'][
                    self.stored['ell_indices']]

        # Prepare parameters list
        self.x_values = params.get_values_emu()

        # Evaluate emulator
        out_emu = self._eval_emu(self.x_values)[self.stored['ell_indices']]

        # Multiply by reference
        if self.ref['spectrum'] is not None:
            out = out_emu * self.ref_spectrum_stored
        else:
            out = out_emu

        return out

    def get_from_class(self, ell, params, precision=0, verbose=False):
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

        # Get additional Class arguments needed to run smoothly
        class_args = {n: self.class_args[n] for n in self.class_args
                      if n not in self.class_high_prec}
        class_args['output'] = 'tCl, pCl, lCl, mPk, dTk'
        class_args['l_max_scalars'] = ell.max()
        class_args['lensing'] = 'yes'

        # Prepare parameters list
        params = self._get_input_params_class(
            params, precision, class_args, verbose=verbose)

        # Compute
        self.cosmo = classy.Class()
        self.cosmo.set(params)
        self.cosmo.compute()

        # Get Cells
        cl_type = self.name.split('_')[1].lower()
        try:
            out = self.cosmo.lensed_cl(lmax=ell.max())[cl_type]
        except KeyError:
            out = self.cosmo.raw_cl(lmax=ell.max())[cl_type]

        # Mask Cells
        ell_out = np.arange(ell.max()+1)
        idx = np.where(np.isin(ell_out, ell))[0]
        out = out[idx]

        # Normalize Cells
        out *= ell*(ell+1.)/2./np.pi

        return out
