from contextlib import contextmanager

import hiclassy
import numpy as np
import scipy.interpolate as interp
from ._tensorflow_config import tf
from .pca import PCA
from .scalers import Scaler


# ------------------- Spectrum -----------------------------------------------#

class Spectrum(object):
    """Base spectrum emulator containing shared logic for Pk/Fk/Cl."""

    def __init__(self, **kwargs):
        """Rehydrate a spectrum emulator from serialized metadata.

        Args:
            **kwargs: Keyword arguments produced by the training pipeline.
                Required keys include ``name``, ``x_names``, ``x_ranges``,
                scaler/PCA configs, the trained ``model``, reference spectra,
                and CLASS arguments. Newer emulator bundles may also include
                ``x_ranges_thin``, ``x_ranges_std``, and ``x_ranges_ext``.
        """
        # Name of the spectrum (str)
        self.name = kwargs['name']
        self._is_pk = None
        self._is_cl = None
        # Parameters names needed by the emulator (list of str)
        self.x_names = kwargs['x_names']
        # Placeholder for values needed by the emulator (list of floats)
        self.x_values = [None for _ in self.x_names]
        # Parameter ranges used by the current validation path. For the
        # shipped bundles this matches the widest stored region.
        self.x_ranges = kwargs['x_ranges']
        # Named trust regions stored during training. They are kept separate
        # from x_ranges so existing validation behavior remains unchanged.
        self.x_ranges_by_region = {
            'thin': kwargs.get('x_ranges_thin', self.x_ranges),
            'std': kwargs.get('x_ranges_std', self.x_ranges),
            'ext': kwargs.get('x_ranges_ext', self.x_ranges),
        }
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
        # addition, it contains the parameters that are used by external
        # routines (e.g. ln_A_s_1e10, n_s to calculate the primordial
        # power spectrum).
        self.input_params_names = []

        # Placeholder for k, z, ell ranges
        self.k_min = None
        self.k_max = None
        self.z_min = None
        self.z_max = None
        self.ell_min = None
        self.ell_max = None

        pass

    @contextmanager
    def _use_class(self, params, requirements=None):
        """Yield a shared cached HiCLASS instance when one is configured."""
        cache = getattr(self, '_class_cache', None)
        if cache is not None:
            with cache.use(params, requirements=requirements) as cosmo:
                self.cosmo = cosmo
                yield cosmo
            return

        # Spectrum objects constructed outside HiFast retain the historical
        # standalone behavior.
        self.cosmo = hiclassy.HiClass()
        self.cosmo.set(params)
        self.cosmo.compute()
        yield self.cosmo

    def _to_numpy_array(self, x):
        """Convert scalars/lists into numpy arrays.

        Args:
            x (int | float | list | numpy.ndarray): Input values.

        Returns:
            numpy.ndarray: Array with float values.
        """
        if isinstance(x, float):
            x = np.array([x])
        elif isinstance(x, int):
            x = np.array([float(x)])
        elif isinstance(x, list):
            x = np.array(x)
        return x

    def _is_same_array(self, x, x_ref):
        """Return ``True`` when ``x`` matches the cached reference array.

        Args:
            x (numpy.ndarray): Current values.
            x_ref (numpy.ndarray | None): Cached values.

        Returns:
            bool: ``True`` when both arrays are equal.
        """
        if x_ref is None:
            return False
        else:
            return np.array_equal(x, x_ref)

    @staticmethod
    def choose_one(**kwargs):
        """Instantiate the appropriate spectrum subclass.

        Args:
            **kwargs: Serialized spectrum metadata including ``name``.

        Returns:
            Spectrum: Concrete `Pk`, `Fk`, or `Cell` instance.

        Raises:
            ValueError: If the spectrum name does not match known prefixes.
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

    def _eval_emu_batch(self, x):
        """Evaluate the emulator for a two-dimensional batch of inputs."""
        x = np.asarray(x)

        # Scale x
        if self.x_scaler is None:
            x_scaled = x
        else:
            x_scaled = self.x_scaler.transform(x)

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
            y = self.y_scaler.inverse_transform(y_scaled)

        return np.asarray(y)

    @staticmethod
    def _tf_scaler_transform(scaler, x):
        """Apply one of the package scalers without leaving TensorFlow.

        The regular scaler methods use scikit-learn/NumPy and therefore
        sever the graph needed by :class:`tf.GradientTape`.
        """
        if scaler is None:
            return x
        name = scaler.name
        dtype = x.dtype

        if name is None or name == 'None':
            return x
        if name == 'MinMaxCommonScaler':
            low = tf.cast(scaler.glob_min_, dtype)
            high = tf.cast(scaler.glob_max_, dtype)
            if scaler.glob_min_ == 0. and scaler.glob_max_ == 0.:
                return x
            if scaler.glob_min_ == scaler.glob_max_:
                return x / high
            return (x - low) / (high - low)

        skl = scaler.skl_scaler
        if name in ('StandardScaler', 'LogStandardScaler',
                    'MinusLogStandardScaler'):
            mean = tf.cast(skl.mean_, dtype)
            scale = tf.cast(skl.scale_, dtype)
            if name == 'LogStandardScaler':
                x = tf.math.log(x)
            elif name == 'MinusLogStandardScaler':
                x = tf.math.log(-x)
            return (x - mean) / scale

        offset = tf.cast(skl.min_, dtype)
        scale = tf.cast(skl.scale_, dtype)
        x = x * scale + offset
        if name == 'MinMaxPlus1Scaler':
            return x + tf.cast(1., dtype)
        if name == 'ExpMinMaxScaler':
            return tf.math.exp(x)
        if name == 'MinMaxScaler':
            return x
        raise ValueError('Scaler {} not recognized!'.format(name))

    @staticmethod
    def _tf_scaler_inverse_transform(scaler, x):
        """Invert one of the package scalers with TensorFlow operations."""
        if scaler is None:
            return x
        name = scaler.name
        dtype = x.dtype

        if name is None or name == 'None':
            return x
        if name == 'MinMaxCommonScaler':
            low = tf.cast(scaler.glob_min_, dtype)
            high = tf.cast(scaler.glob_max_, dtype)
            if scaler.glob_min_ == 0. and scaler.glob_max_ == 0.:
                return x
            if scaler.glob_min_ == scaler.glob_max_:
                return x * high
            return x * (high - low) + low

        skl = scaler.skl_scaler
        if name in ('StandardScaler', 'LogStandardScaler',
                    'MinusLogStandardScaler'):
            mean = tf.cast(skl.mean_, dtype)
            scale = tf.cast(skl.scale_, dtype)
            x = x * scale + mean
            if name == 'LogStandardScaler':
                return tf.math.exp(x)
            if name == 'MinusLogStandardScaler':
                return -tf.math.exp(x)
            return x

        offset = tf.cast(skl.min_, dtype)
        scale = tf.cast(skl.scale_, dtype)
        if name == 'MinMaxPlus1Scaler':
            x = x - tf.cast(1., dtype)
        elif name == 'ExpMinMaxScaler':
            x = tf.math.log(x)
        if name in ('MinMaxScaler', 'MinMaxPlus1Scaler',
                    'ExpMinMaxScaler'):
            return (x - offset) / scale
        raise ValueError('Scaler {} not recognized!'.format(name))

    @staticmethod
    def _tf_pca_transform(pca, x):
        """Apply a restored scikit-learn PCA as TensorFlow operations."""
        if pca is None:
            return x
        dtype = x.dtype
        mean = tf.cast(pca.pca.mean_, dtype)
        components = tf.cast(pca.pca.components_, dtype)
        return tf.linalg.matmul(x - mean, components, transpose_b=True)

    @staticmethod
    def _tf_pca_inverse_transform(pca, x):
        """Invert a restored PCA without detaching the gradient graph."""
        if pca is None:
            return x
        dtype = x.dtype
        mean = tf.cast(pca.pca.mean_, dtype)
        components = tf.cast(pca.pca.components_, dtype)
        return tf.linalg.matmul(x, components) + mean

    def _eval_emu_and_dz(self, params, z):
        """Evaluate a batch and differentiate each row with respect to z."""
        dtype = tf.as_dtype(self.model.compute_dtype)
        z_tf = tf.convert_to_tensor(z, dtype=dtype)
        values = []
        for name in self.x_names:
            if name == 'z_pk':
                values.append(z_tf)
            else:
                values.append(tf.convert_to_tensor(
                    [row[name] for row in params], dtype=dtype))

        with tf.autodiff.ForwardAccumulator(
                primals=z_tf,
                tangents=tf.ones_like(z_tf)) as accumulator:
            x = tf.stack(values, axis=1)
            x = self._tf_scaler_transform(self.x_scaler, x)
            x = self._tf_pca_transform(self.x_pca, x)
            y = self.model(x, training=False)
            y = self._tf_pca_inverse_transform(self.y_pca, y)
            y = self._tf_scaler_inverse_transform(self.y_scaler, y)

        dy_dz = accumulator.jvp(y)
        if dy_dz is None:
            raise RuntimeError(
                'The loaded emulator is not differentiable with respect '
                'to z_pk')
        return y.numpy(), dy_dz.numpy()

    def _get_input_params_class(
            self, params, precision, class_args, verbose=False):
        """Prepare input parameters for a CLASS run.

        Args:
            params (dict[str, float]): Cosmological parameters in hi_fast
                nomenclature.
            precision (int | dict[str, float]): Either 0/1/2 for preset
                precision configs or a dict of CLASS precision overrides.
            class_args (dict[str, float]): Baseline CLASS arguments gathered
                from training metadata.
            verbose (bool): When True, request verbose CLASS output.

        Returns:
            dict[str, float]: Combined dictionary ready to pass to CLASS.

        Raises:
            Exception: If ``precision`` is neither {0,1,2} nor a dict.
        """

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
            # TODO: implement sigma8_cb and S8_cb
            # (in Class all is defined wrt _m)
            # hi_fast_name, class_name
            ('sigma8_m', 'sigma8'),
            ('S8_m', 'S8'),
        ]

        for hi_fast_name, class_name in renaming_rules:
            if hi_fast_name in params_in:
                params_in[class_name] = params_in.pop(hi_fast_name)

        # Remove cosmo parameters from class_args if they are in params_in
        if 'sigma8' in params_in and 'ln_A_s_1e10' in class_args:
            class_args.pop('ln_A_s_1e10')
        if 'A_s' in params_in and 'ln_A_s_1e10' in class_args:
            class_args.pop('ln_A_s_1e10')
        if 'S8' in params_in and 'ln_A_s_1e10' in class_args:
            class_args.pop('ln_A_s_1e10')

        all_params = verb_params | class_args | prec | params_in

        return all_params


# ------------------- Pk -----------------------------------------------------#

class Pk(Spectrum):
    """Matter or Weyl-power-spectrum emulator."""

    def __init__(self, **kwargs):
        Spectrum.__init__(self, **kwargs)

        # Spectrum type flags
        self._is_pk = True
        self._is_cl = False

        # Input parameters (primordial power spectrum)
        self.input_params_names = [nm for nm in self.x_names if nm != 'z_pk']
        self.input_params_names += ['ln_A_s_1e10', 'n_s']

        # Add ref_z and ref_k to ref dictionary
        self.ref['z'] = kwargs['ref_z']
        self.ref['k'] = kwargs['ref_k']
        self.ref['spectrum_z_spline'] = None
        self.ref['spectrum_dz_spline'] = None
        if self.ref['spectrum'] is not None:
            self.ref['spectrum_z_spline'] = interp.make_splrep(
                self.ref['z'], self.ref['spectrum'].T, s=0)
            self.ref['spectrum_dz_spline'] = (
                self.ref['spectrum_z_spline'].derivative())

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

    def _store_reference_spectrum(self, k, z):
        """Cache the reference spectrum on a requested k-z grid."""
        same_k = self._is_same_array(k, self.stored['k'])
        same_z = self._is_same_array(z, self.stored['z'])
        if same_k and same_z:
            return

        self.stored['k'] = k
        self.stored['z'] = z
        if self.ref['spectrum_z_spline'] is None:
            self.stored['ref_spectrum'] = None
            return

        ref_at_z = self.ref['spectrum_z_spline'](z)
        self.stored['ref_spectrum'] = interp.make_splrep(
            self.ref['k'], ref_at_z.T, s=0)(k)

    def _check_k_values(self, k):
        """Ensure wavenumbers stay inside the training range.

        Args:
            k (numpy.ndarray): Wavenumbers in h/Mpc.

        Raises:
            Exception: If any value falls outside ``[k_min, k_max]``.
        """
        if k.min() < self.k_min or k.max() > self.k_max:
            raise Exception(
                'k (h/Mpc) = [{} - {}] out of range [{} - {}]'.format(
                    k.min(), k.max(), self.k_min, self.k_max))
        return

    def _check_z_values(self, z):
        """Ensure redshifts stay inside the training range.

        Args:
            z (numpy.ndarray): Redshifts.

        Raises:
            Exception: If any value falls outside ``[z_min, z_max]``.
        """
        if z.min() < self.z_min or z.max() > self.z_max:
            raise Exception(
                'z = [{} - {}] out of range [{} - {}]'.format(
                    z.min(), z.max(), self.z_min, self.z_max))
        return

    def _get_values_emu(self, params, **kwargs):
        """Assemble emulator inputs for a specific redshift.

        Args:
            params (dict[str, float]): Cosmological parameters.
            **kwargs: Expected to contain ``z_pk`` for the target slice.

        Returns:
            numpy.ndarray: Ordered emulator inputs.
        """
        vals = [
            params[p] if p != 'z_pk' else None for p in self.x_names]
        vals[self.x_names.index('z_pk')] = kwargs['z_pk']
        return np.array(vals)

    def _sigma_R_integral(self, k, pk, R):
        """Compute ``sigma_R`` via the standard top-hat integral.

        Args:
            k (numpy.ndarray): Wavenumbers in h/Mpc.
            pk (numpy.ndarray): Power spectrum in ``(Mpc/h)^3``.
            R (float): Smoothing scale in Mpc/h.

        Returns:
            float: ``sigma_R`` value.
        """

        # Window function
        x = k * R
        W = 3. * (np.sin(x) - x * np.cos(x)) / x**3.

        integrand = k**2. * pk * W**2.

        sigma_R = np.sqrt(
            np.trapezoid(integrand, x=k) / (2. * np.pi**2.))

        return sigma_R

    def _get_primordial_pk(self, ln_A_s_1e10, n_s, k_pivot, k):
        """Return the primordial power spectrum evaluated at ``k``.

        Args:
            ln_A_s_1e10 (float): Unnormalized log amplitude.
            n_s (float): Scalar spectral index.
            k_pivot (float): Pivot scale in h/Mpc.
            k (numpy.ndarray): Target wavenumbers in h/Mpc.

        Returns:
            numpy.ndarray: Primordial spectrum values.
        """

        log_P_1e10 = ln_A_s_1e10 + (n_s-1.)*np.log(k/k_pivot)
        P_primordial = np.exp(log_P_1e10) / 1e10

        return P_primordial

    def _get_weyl_pk(self, k, z, n_k, n_z, n_mu):
        """Return the nonlinear Weyl power spectrum from CLASS.

        Args:
            k (numpy.ndarray): Wavenumbers in 1/Mpc with shape
                ``(n_k, n_z, n_mu)``.
            z (numpy.ndarray): Redshift samples.
            n_k (int): Number of k values.
            n_z (int): Number of z values.
            n_mu (int): Number of mu values.

        Returns:
            numpy.ndarray: Weyl power spectrum evaluated at ``k`` and ``z``.
        """

        # TODO: for now we just call the linear one
        pk = self._get_weyl_pk_lin(k, z, n_k, n_z, n_mu)

        return pk

    def _get_weyl_pk_lin(self, k, z, n_k, n_z, n_mu):
        """Return the linear Weyl spectrum from CLASS.

        Args:
            k (numpy.ndarray): Wavenumbers in 1/Mpc with shape
                ``(n_k, n_z, n_mu)``.
            z (numpy.ndarray): Redshift samples.
            n_k (int): Number of k samples.
            n_z (int): Number of redshift samples.
            n_mu (int): Number of angular samples.

        Returns:
            numpy.ndarray: Linear Weyl power spectrum.
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

    def get(self, k, z, params, nonlinear=False):
        """Evaluate the emulator power spectrum ``P(k, z)``.

        Args:
            k (float | list | numpy.ndarray): Wavenumbers in h/Mpc.
            z (float | list | numpy.ndarray): Redshifts.
            params (dict[str, float]): Cosmological parameters dictionary.
            nonlinear (bool): Placeholder flag (nonlinear not supported).

        Returns:
            numpy.ndarray: Power spectrum with units ``(Mpc/h)^3``.

        Raises:
            ValueError: If ``nonlinear`` is True.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        k = self._to_numpy_array(k)
        z = self._to_numpy_array(z)
        if isinstance(params, dict):
            params = [params] * len(z)
        if len(params) != len(z):
            raise ValueError('params and z must contain the same number of '
                             'cosmologies')
        self._check_k_values(k)
        self._check_z_values(z)
        self._store_reference_spectrum(k, z)

        x = [self._get_values_emu(row, z_pk=z_one)
             for row, z_one in zip(params, z)]
        out_emu = self._eval_emu_batch(x)
        if not self._is_same_array(k, self.ref['k']):
            out_emu = interp.make_splrep(
                self.ref['k'], out_emu.T, s=0)(k).T
        if self.stored['ref_spectrum'] is not None:
            out = out_emu * self.stored['ref_spectrum'].T
        else:
            out = out_emu

        # Adjust shape with primordial Pk
        ref_primordial = self._get_primordial_pk(
            self.ref['params']['ln_A_s_1e10'],
            self.ref['params']['n_s'],
            self.ref['params']['k_pivot'],
            k*self.ref['params']['h'])
        for index, row in enumerate(params):
            primordial = self._get_primordial_pk(
                row['ln_A_s_1e10'], row['n_s'],
                self.ref['params']['k_pivot'], k*row['h'])
            out[index] *= primordial / ref_primordial

        return out

    def get_from_class(
            self, k, z, params, nonlinear=False, precision=0, verbose=False):
        """Compute ``P(k, z)`` or ``f(k, z)`` directly with CLASS.

        Args:
            k (float | list | numpy.ndarray): Wavenumbers in h/Mpc.
            z (float | list | numpy.ndarray): Redshifts.
            params (dict[str, float]): Cosmological parameters.
            nonlinear (bool): Placeholder flag (nonlinear not yet supported).
            precision (int | dict[str, float]): CLASS precision preset or
                overrides.
            verbose (bool): When True, enable verbose CLASS logs.

        Returns:
            numpy.ndarray: Spectrum values in ``(Mpc/h)^3``.
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

        requirements = {
            'output': 'mPk, dTk',
            'P_k_max_h/Mpc': k.max(),
            'z_max_pk': max(z.max(), 0.1),
        }
        with self._use_class(params, requirements=requirements) as cosmo:
            # Get correct spectrum
            if nonlinear is True and self.name.endswith('_cb'):
                fun = cosmo.get_pk_cb
            elif nonlinear is True and self.name.endswith('_m'):
                fun = cosmo.get_pk
            elif nonlinear is True and self.name.endswith('_weyl'):
                fun = self._get_weyl_pk
            elif nonlinear is False and self.name.endswith('_cb'):
                fun = cosmo.get_pk_cb_lin
            elif nonlinear is False and self.name.endswith('_m'):
                fun = cosmo.get_pk_lin
            elif nonlinear is False and self.name.endswith('_weyl'):
                fun = self._get_weyl_pk_lin

            # convert k in units of 1/Mpc
            n_mu = 1
            n_z = len(z)
            n_k = len(k)
            k_3D = np.broadcast_to(
                k[:, np.newaxis, np.newaxis],
                (n_k, n_z, n_mu)) * cosmo.h()
            out = fun(k_3D, z, n_k, n_z, n_mu) * cosmo.h()**3.
            out = out[:, :, 0]

        return out

    def get_fk_from_class(
            self, k, z, params, nonlinear=False, precision=0,
            verbose=False):
        """Compute ``f(k, z)`` with HiCLASS using this P(k) metadata.

        This supports HiCLASS fallback even when a bundle has no dedicated
        growth-rate emulator.
        """
        return Fk.get_from_class(
            self, k, z, params, nonlinear=nonlinear,
            precision=precision, verbose=verbose)

    def get_fk(self, k, z, params, nonlinear=False):
        """Evaluate f(k,z) from P(k,z) for independent cosmology rows.

        ``z`` and ``params`` must have the same length. A single parameter
        dictionary and scalar redshift are accepted as a one-row batch. The
        returned shape is ``(n_cosmologies, n_k)``.
        """
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        k = self._to_numpy_array(k)
        z = self._to_numpy_array(z)
        if isinstance(params, dict):
            params = [params]
        if len(params) != len(z):
            raise ValueError('params and z must contain the same number of '
                             'cosmologies')
        self._check_k_values(k)
        self._check_z_values(z)

        emu, demu_dz = self._eval_emu_and_dz(params, z)
        if self.ref['spectrum_z_spline'] is not None:
            ref = np.asarray(self.ref['spectrum_z_spline'](z))
            dref_dz = np.asarray(self.ref['spectrum_dz_spline'](z))
            if ref.shape != emu.shape:
                ref = ref.T
                dref_dz = dref_dz.T
            pk = emu * ref
            dpk_dz = demu_dz * ref + emu * dref_dz
        else:
            pk = emu
            dpk_dz = demu_dz

        if self._is_same_array(k, self.ref['k']):
            pk_at_k = pk
            dpk_dz_at_k = dpk_dz
        else:
            pk_at_k = interp.make_splrep(
                self.ref['k'], pk.T, s=0)(k).T
            dpk_dz_at_k = interp.make_splrep(
                self.ref['k'], dpk_dz.T, s=0)(k).T

        return (-0.5 * (1. + z[:, np.newaxis])
                * dpk_dz_at_k / pk_at_k)

    def get_sigma_R(self, R, z, params, nonlinear=False):
        """Return ``sigma_R`` evaluated at smoothing scale ``R``.

        Args:
            R (float): Smoothing scale in Mpc/h.
            z (float): Redshift.
            params (dict[str, float]): Cosmological parameters.
            nonlinear (bool): Placeholder flag (nonlinear not supported).

        Returns:
            float: ``sigma_R`` value.
        """

        # Use the reference k array
        k = self.ref['k']

        # Get P(k, z)
        pk = self.get(k, z, params, nonlinear=nonlinear)[0]

        # Compute sigma_R
        sigma_R = self._sigma_R_integral(k, pk, R)

        return sigma_R

    def get_sigma8_from_params(self, params):
        """Return ``sigma_8`` for the provided parameters.

        Args:
            params (dict[str, float]): Cosmological parameters.

        Returns:
            float: Corresponding ``sigma_8`` value.
        """

        sigma8 = self.get_sigma_R(8., 0.0, params)
        return sigma8

    def get_S8_from_params(self, params):
        """Return ``S_8 = sigma_8 * sqrt(Omega_m/0.3)``.

        Args:
            params (dict[str, float]): Cosmological parameters.

        Returns:
            float: Corresponding ``S_8`` value.
        """

        sigma8 = self.get_sigma_R(8., 0.0, params)
        return sigma8 * np.sqrt(params['Omega_m']/0.3)


# ------------------- Pk -----------------------------------------------------#

class Fk(Pk):
    """Growth-rate emulator using the same infrastructure as ``Pk``."""

    def __init__(self, **kwargs):
        Pk.__init__(self, **kwargs)

        # Input parameters
        self.input_params_names = [nm for nm in self.x_names if nm != 'z_pk']

        pass

    def get(self, k, z, params, nonlinear=False):
        """Evaluate the emulator growth rate ``f(k, z)``.

        Args:
            k (float | list | numpy.ndarray): Wavenumbers in h/Mpc.
            z (float | list | numpy.ndarray): Redshifts.
            params (dict[str, float]): Cosmological parameters.
            nonlinear (bool): Placeholder flag (nonlinear not supported).

        Returns:
            numpy.ndarray: Growth rate values.
        """

        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')
        k = self._to_numpy_array(k)
        z = self._to_numpy_array(z)
        if isinstance(params, dict):
            params = [params] * len(z)
        if len(params) != len(z):
            raise ValueError('params and z must contain the same number of '
                             'cosmologies')
        self._check_k_values(k)
        self._check_z_values(z)
        self._store_reference_spectrum(k, z)

        x = [self._get_values_emu(row, z_pk=z_one)
             for row, z_one in zip(params, z)]
        out = self._eval_emu_batch(x)
        if not self._is_same_array(k, self.ref['k']):
            out = interp.make_splrep(
                self.ref['k'], out.T, s=0)(k).T
        if self.stored['ref_spectrum'] is not None:
            out *= self.stored['ref_spectrum'].T
        return out

    def get_from_class(
            self, k, z, params, nonlinear=False, precision=0, verbose=False):
        """Compute ``f(k, z)`` by differentiating CLASS power spectra.

        Args:
            k (float | list | numpy.ndarray): Wavenumbers in h/Mpc.
            z (float | list | numpy.ndarray): Redshifts.
            params (dict[str, float]): Cosmological parameters.
            nonlinear (bool): Whether to use nonlinear spectra.
            precision (int | dict[str, float]): CLASS precision settings.
            verbose (bool): When True, enable verbose CLASS logs.

        Returns:
            numpy.ndarray: Growth-rate values derived from CLASS outputs.
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

        requirements = {
            'output': 'mPk, dTk',
            'P_k_max_h/Mpc': k.max(),
            'z_max_pk': max(z.max(), 0.1),
        }
        with self._use_class(params, requirements=requirements) as cosmo:
            # Get correct spectrum
            if self.name.endswith('_cb'):
                pk_array, k_array, z_array = cosmo.get_pk_and_k_and_z(
                    nonlinear=nonlinear,
                    only_clustering_species=True,
                    h_units=False)
            elif self.name.endswith('_m'):
                pk_array, k_array, z_array = cosmo.get_pk_and_k_and_z(
                    nonlinear=nonlinear,
                    only_clustering_species=False,
                    h_units=False)
            elif self.name.endswith('_weyl'):
                pk_array, k_array, z_array = (
                    cosmo.get_Weyl_pk_and_k_and_z(
                        nonlinear=nonlinear,
                        h_units=False))

            # Flip z_array (for interpolation it has to be increasing)
            z_array = np.flip(z_array)
            pk_array = np.flip(pk_array, axis=1)
            k_array /= cosmo.h()

            # Evaluate pk at the requested range
            pk = interp.make_splrep(k_array, pk_array, s=0)(k)
            pk_at_z = interp.make_splrep(z_array, pk.T, s=0)(z).T
            dpkdz = interp.make_splrep(
                z_array, pk.T, s=0).derivative()(z).T
            out = -0.5 * (1+z) * dpkdz/pk_at_z
        return out


# ------------------- Cell ---------------------------------------------------#

class Cell(Spectrum):
    """Angular CMB spectrum emulator."""

    def __init__(self, **kwargs):
        Spectrum.__init__(self, **kwargs)

        # Spectrum type flags
        self._is_pk = False
        self._is_cl = True

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
        """Ensure multipoles fall within the emulator range.

        Args:
            ell (numpy.ndarray): Multipoles.

        Raises:
            Exception: If ``ell`` extends beyond ``[ell_min, ell_max]``.
        """
        if ell.min() < self.ell_min or ell.max() > self.ell_max:
            raise Exception(
                'ell = [{} - {}] out of range [{} - {}]'.format(
                    ell.min(), ell.max(), self.ell_min, self.ell_max))
        return

    def _to_numpy_array(self, x):
        x = Spectrum._to_numpy_array(self, x)
        return x.astype(int)

    def _get_values_emu(self, params, **kwargs):
        """Pack emulator inputs for ``Cell`` spectra.

        Args:
            params (dict[str, float]): Cosmological parameters.

        Returns:
            numpy.ndarray: Ordered emulator inputs.
        """
        vals = [params[p] for p in self.x_names]
        return np.array(vals)

    def get(self, ell, params):
        """Evaluate the emulator angular spectrum ``C_ell``.

        Args:
            ell (int | list[int] | numpy.ndarray): Multipoles.
            params (dict[str, float]): Cosmological parameters.

        Returns:
            numpy.ndarray: Dimensionless
            ``\\ell(\\ell+1)C_\\ell/(2\\pi)``.
        """

        if isinstance(params, dict):
            params = [params]

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
                self.stored['ref_spectrum'] = self.ref['spectrum'][
                    self.stored['ell_indices']]

        # Prepare and evaluate all cosmologies in one model call.
        x = [self._get_values_emu(row) for row in params]
        out_emu = self._eval_emu_batch(x)[:, self.stored['ell_indices']]

        # Multiply by reference
        if self.ref['spectrum'] is not None:
            out = out_emu * self.stored['ref_spectrum'][np.newaxis, :]
        else:
            out = out_emu

        return out

    def get_from_class(self, ell, params, precision=0, verbose=False):
        """Compute ``C_ell`` directly with CLASS.

        Args:
            ell (int | list[int] | numpy.ndarray): Multipoles.
            params (dict[str, float]): Cosmological parameters.
            precision (int | dict[str, float]): CLASS precision settings.
            verbose (bool): When True, enable verbose CLASS logs.

        Returns:
            numpy.ndarray: Dimensionless angular spectra.
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

        requirements = {
            'output': 'tCl, pCl, lCl',
            'l_max_scalars': ell.max(),
        }
        with self._use_class(params, requirements=requirements) as cosmo:
            # Get Cells
            cl_type = self.name.split('_')[1].lower()
            if self.name.endswith('_lensed'):
                out = cosmo.lensed_cl(lmax=ell.max())[cl_type]
            else:
                out = cosmo.raw_cl(lmax=ell.max())[cl_type]

        # Mask Cells
        ell_out = np.arange(ell.max()+1)
        idx = np.where(np.isin(ell_out, ell))[0]
        out = out[idx]

        # Normalize Cells
        out *= ell*(ell+1.)/2./np.pi

        return out
