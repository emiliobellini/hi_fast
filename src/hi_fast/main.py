import numpy as np

from . import io as io
from . import spectra as sp
from .params import Params


class HiFast(object):
    """High-level interface for loading spectra emulators and producing
    cosmological observables."""

    def __init__(self, name, root='emu', timeit=False, verbose=False):
        """Instantiate the hi_fast interface and preload all requested
        emulators.

        Args:
            name (str | None): Emulator family to load (e.g. ``lcdm``).
                When ``None``, spectra must be loaded manually before
                evaluation.
            root (str): Directory containing the emulator bundles.
            timeit (bool): When True, log how long the loading step takes.
            verbose (bool): When True, print extra information while
                loading.
        """
        # Load spectra emulators
        self._spectra = self._load(
            name, root=root, timeit=timeit, verbose=verbose)
        # Init parameters handlers
        self._params = {spec.name: Params(spec, self._spectra)
                        for spec in self._spectra.values()}

        pass

    @staticmethod
    def _batch_ranges(length, batch_size):
        """Yield half-open row ranges for an optional internal batch size."""
        if batch_size is None:
            batch_size = length
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer or None')
        for first in range(0, length, batch_size):
            yield first, min(first + batch_size, length)

    @io.timeit
    def _load(self, name, root, timeit=False, verbose=False):
        """Load every spectrum emulator shipped inside a bundle.

        Args:
            name (str): Bundle name to load (e.g. ``lcdm``).
            root (str): Directory that contains the bundle directories or
                files.
            timeit (bool): Unused placeholder to keep ``@io.timeit``
                signature consistent.
            verbose (bool): When True, report which model was loaded.

        Returns:
            dict[str, sp.Spectrum]: Mapping from spectrum name to the
            instantiated emulator object.
        """
        # Initialize spectra dictionary
        spectra = {}
        # Load emulators as dictionary
        for file in io.Folder(name, root=root).list_files():
            # Read content emu
            emufile = io.EmuFile(file)

            # Check that it is not a dictionary file
            if emufile._is_dict_file() is False:
                continue
            # Load content
            content = emufile.load()
            # Store content
            spectra[content['name']] = sp.Spectrum.choose_one(**content)
        if verbose:
            io.info('Loaded emulators for {} model'.format(name))
        return spectra

    @io.timeit
    def get_pk(
            self,
            k,
            z,
            params,
            name='m',
            nonlinear=False,
            check_params_names=True,
            check_params_values=True,
            batch_size=None,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Evaluate the emulator power spectrum ``P(k, z)``.

        Args:
            k (float | sequence[float] | numpy.ndarray): Wavenumbers in
                units of h/Mpc.
            z (float | sequence[float] | numpy.ndarray): Redshifts to
                evaluate.
            params (dict[str, float]): Cosmological parameters. Each
                spectrum defines which keys are required and how they are
                converted.
            name (str): Spectrum identifier; ``m`` (total matter), ``cb``
                (CDM plus baryons), or ``weyl`` (Weyl potential).
            nonlinear (bool): Placeholder flag; nonlinear emulation is not
                implemented yet.
            check_params_names (bool): When True, ensure the provided
                dictionary contains exactly one representative for every
                required parameter.
            check_params_values (bool): When True, validate the converted
                parameters lie inside the emulator training domain.
            squeeze (bool): When True, drop singleton dimensions in the
                result for convenience.
            verbose (bool): When True, echo both provided and converted
                parameters.
            timeit (bool): When True, report evaluation time (handled by the
                decorator).

        Returns:
            numpy.ndarray or float: Power spectrum values in units of
            ``(Mpc/h)^3`` matching the broadcast shape of ``k`` and ``z``.

        Raises:
            ValueError: If ``nonlinear`` is True.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct spectrum
        spectrum = self._spectra['pk_{}'.format(name)]

        if isinstance(params, dict):
            params = [params]
        if not hasattr(z, '__len__'):
            z = [z]
        if len(params) == 1 and len(z) > 1:
            params = params * len(z)
        if len(params) != len(z):
            raise ValueError('params and z must contain the same number of '
                             'cosmologies')
        outputs = []
        for first, last in self._batch_ranges(len(params), batch_size):
            converted = [self._params[spectrum.name].get(
                row,
                check_params_names=check_params_names,
                check_params_values=check_params_values,
                verbose=verbose) for row in params[first:last]]
            outputs.append(spectrum.get(k, z[first:last], converted))
        out = np.concatenate(outputs, axis=0)

        # Squeeze dimensions
        if squeeze:
            return out.squeeze()

        return out

    @io.timeit
    def get_pk_old(
            self, k, z, params, name='m', nonlinear=False,
            check_params_names=True, check_params_values=True,
            squeeze=False, verbose=False, timeit=False):
        """Evaluate P(k,z) with the pre-batch implementation."""
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')
        spectrum = self._spectra['pk_{}'.format(name)]
        params = self._params[spectrum.name].get(
            params, check_params_names=check_params_names,
            check_params_values=check_params_values, verbose=verbose)
        out = spectrum.get_old(k, z, params)
        return out.squeeze() if squeeze else out

    @io.timeit
    def get_fk_old(
            self,
            k,
            z,
            params,
            name='m',
            get_from_pk=False,
            nonlinear=False,
            check_params_names=True,
            check_params_values=True,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Evaluate the emulator growth rate
        ``f(k, z) = d ln P(k, z)/d ln a``.

        Args:
            k (float | sequence[float] | numpy.ndarray): Wavenumbers in
                units of h/Mpc.
            z (float | sequence[float] | numpy.ndarray): Redshifts to
                evaluate.
            params (dict[str, float]): Cosmological parameters.
            name (str): Spectrum identifier; ``m``, ``cb``, or ``weyl``.
            get_from_pk (bool): When True, compute the growth rate from
                the power spectrum.
            nonlinear (bool): Placeholder flag; nonlinear emulation is not
                implemented yet.
            check_params_names (bool): When True, validate coverage of all
                required parameters.
            check_params_values (bool): When True, ensure parameters are in
                range.
            squeeze (bool): When True, drop singleton dimensions in the
                returned array.
            verbose (bool): When True, echo provided and converted
                parameters.
            timeit (bool): When True, report evaluation time.

        The returned growth rate is dimensionless.

        Returns:
            numpy.ndarray or float: Growth-rate values with the same shape
            broadcasting rules as ``get_pk``.

        Raises:
            ValueError: If ``nonlinear`` is True.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Check if emulator is available; if not, compute from Pk
        if get_from_pk is False and 'fk_{}'.format(name) not in self._spectra:
            get_from_pk = True
            io.warning('No emulator for fk_{} found; computing from Pk instead'
                       ''.format(name))

        # Select correct spectrum
        if get_from_pk is True:
            spectrum = self._spectra['pk_{}'.format(name)]
            # Add fake parameters that are used in the Power Spectrum
            # but not in the growth rate (fixed them to reference to
            # avoid numerical issues)
            params['A_s'] = spectrum.ref['params']['ln_A_s_1e10']
            params['n_s'] = spectrum.ref['params']['n_s']
        else:
            spectrum = self._spectra['fk_{}'.format(name)]

        # Get parameters
        params = self._params[spectrum.name].get(
            params,
            check_params_names=check_params_names,
            check_params_values=check_params_values,
            verbose=verbose)

        # Get output
        if get_from_pk is True:
            out = spectrum.get_fk_old(k, z, params)
        else:
            out = spectrum.get_old(k, z, params)

        # Squeeze dimensions
        if squeeze:
            if out.shape == (1, 1):
                return out[0, 0]
            elif out.shape[0] == 1:
                return out[0]
            elif out.shape[1] == 1:
                return out[:, 0]

        return out

    @io.timeit
    def get_fk(
            self,
            k,
            z,
            params,
            name='m',
            get_from_pk=False,
            nonlinear=False,
            check_params_names=True,
            check_params_values=True,
            batch_size=None,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Evaluate the growth rate using the new implementation.

        This is initially a copy of :meth:`get_fk_old`.  The two entry points
        are intentionally independent so changes to the new P(k,z)-derivative
        path can be benchmarked against the preserved implementation.
        """
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        if get_from_pk is False and 'fk_{}'.format(name) not in self._spectra:
            get_from_pk = True
            io.warning('No emulator for fk_{} found; computing from Pk instead'
                       ''.format(name))

        if isinstance(params, dict):
            params = [params]
        if not hasattr(z, '__len__'):
            z = [z]
        if len(params) == 1 and len(z) > 1:
            params = params * len(z)
        if len(params) != len(z):
            raise ValueError('params and z must contain the same number of '
                             'cosmologies')

        if get_from_pk is True:
            spectrum = self._spectra['pk_{}'.format(name)]
            for row in params:
                row['A_s'] = spectrum.ref['params']['ln_A_s_1e10']
                row['n_s'] = spectrum.ref['params']['n_s']
        else:
            spectrum = self._spectra['fk_{}'.format(name)]

        outputs = []
        for first, last in self._batch_ranges(len(params), batch_size):
            converted = [self._params[spectrum.name].get(
                row,
                check_params_names=check_params_names,
                check_params_values=check_params_values,
                verbose=verbose) for row in params[first:last]]
            if get_from_pk is True:
                outputs.append(spectrum.get_fk(
                    k, z[first:last], converted))
            else:
                outputs.append(spectrum.get(
                    k, z[first:last], converted))
        out = np.concatenate(outputs, axis=0)

        if squeeze:
            return out.squeeze()

        return out

    @io.timeit
    def get_cell(
            self,
            ell,
            params,
            name='TT',
            check_params_names=True,
            check_params_values=True,
            batch_size=None,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Return dimensionless CMB angular spectra ``C_ell``.

        Emulators store and predict ``\\ell(\\ell+1)C_\\ell / (2\\pi)``.

        Args:
            ell (int | sequence[int] | numpy.ndarray): Multipoles to
                evaluate.
            params (dict[str, float]): Cosmological parameters dictionary.
            name (str): Cl spectrum identifier (``TT``, ``TE``, ``EE``,
                ``Tp``, ``pp``, ``BB``). Lensed versions are used when
                available.
            check_params_names (bool): Validate parameter coverage.
            check_params_values (bool): Validate converted parameter ranges.
            squeeze (bool): Return scalars for 1-element outputs.
            verbose (bool): When True, print provided and derived
                parameters.
            timeit (bool): When True, measure call duration.

        Returns:
            numpy.ndarray or float: Dimensionless angular spectrum values.
        """

        # Select correct spectrum
        try:
            spectrum = self._spectra['cl_{}_lensed'.format(name)]
        except KeyError:
            spectrum = self._spectra['cl_{}'.format(name)]

        if isinstance(params, dict):
            params = [params]
        outputs = []
        for first, last in self._batch_ranges(len(params), batch_size):
            converted = [self._params[spectrum.name].get(
                row,
                check_params_names=check_params_names,
                check_params_values=check_params_values,
                verbose=verbose) for row in params[first:last]]
            outputs.append(spectrum.get(ell, converted))
        out = np.concatenate(outputs, axis=0)

        # Squeeze dimensions
        if squeeze:
            return out.squeeze()

        return out

    @io.timeit
    def get_cell_old(
            self, ell, params, name='TT', check_params_names=True,
            check_params_values=True, squeeze=False, verbose=False,
            timeit=False):
        """Evaluate one CMB cosmology with the pre-batch implementation."""
        try:
            spectrum = self._spectra['cl_{}_lensed'.format(name)]
        except KeyError:
            spectrum = self._spectra['cl_{}'.format(name)]
        params = self._params[spectrum.name].get(
            params, check_params_names=check_params_names,
            check_params_values=check_params_values, verbose=verbose)
        out = spectrum.get_old(ell, params)
        return out.squeeze() if squeeze else out

    @io.timeit
    def get_pk_from_class(
            self,
            k,
            z,
            params,
            name='m',
            precision=0,
            nonlinear=False,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Call CLASS to compute ``P(k, z)`` instead of using the emulator.

        Args:
            k (float | sequence[float] | numpy.ndarray): Wavenumbers in
                units of h/Mpc.
            z (float | sequence[float] | numpy.ndarray): Redshifts.
            params (dict[str, float]): Cosmological parameters to forward to
                CLASS.
            name (str): Spectrum identifier (``m``, ``cb``, ``weyl``).
            precision (int | dict[str, float]): Either 0/1/2 to select preset
                CLASS precision blocks or a dict of CLASS precision
                settings.
            nonlinear (bool): Placeholder flag; nonlinear mode is not
                available.
            squeeze (bool): Whether to drop singleton dimensions in the
                response.
            verbose (bool): When True, pass verbose flag to the spectrum
                wrapper.
            timeit (bool): When True, measure call duration.

        Returns:
            numpy.ndarray or float: Power spectrum values with units
            ``(Mpc/h)^3``.

        Raises:
            ValueError: If ``nonlinear`` is True.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct spectrum
        spectrum = self._spectra['pk_{}'.format(name)]

        # Get output
        out = spectrum.get_from_class(
            k, z, params, precision=precision, verbose=verbose)

        # Squeeze dimensions
        if squeeze:
            if out.shape == (1, 1):
                return out[0, 0]
            elif out.shape[0] == 1:
                return out[0]
            elif out.shape[1] == 1:
                return out[:, 0]

        return out

    @io.timeit
    def get_fk_from_class(
            self,
            k,
            z,
            params,
            name='m',
            precision=0,
            nonlinear=False,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Call CLASS to compute the growth rate ``f(k, z)``.

        Args:
            k (float | sequence[float] | numpy.ndarray): Wavenumbers in
                units of h/Mpc.
            z (float | sequence[float] | numpy.ndarray): Redshifts.
            params (dict[str, float]): Cosmological parameters forwarded to
                CLASS.
            name (str): Spectrum identifier (``m``, ``cb``, ``weyl``).
            precision (int | dict[str, float]): CLASS precision settings or
                preset index.
            nonlinear (bool): Placeholder flag; nonlinear mode is not
                available.
            squeeze (bool): Whether to drop singleton dimensions in the
                response.
            verbose (bool): When True, pass verbose flag to the spectrum
                wrapper.
            timeit (bool): When True, measure call duration.

        The returned quantity is dimensionless.

        Returns:
            numpy.ndarray or float: Growth-rate values.

        Raises:
            ValueError: If ``nonlinear`` is True.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct spectrum
        spectrum = self._spectra['fk_{}'.format(name)]

        # Get output
        out = spectrum.get_from_class(
            k, z, params, precision=precision, verbose=verbose)

        # Squeeze dimensions
        if squeeze:
            if out.shape == (1, 1):
                return out[0, 0]
            elif out.shape[0] == 1:
                return out[0]
            elif out.shape[1] == 1:
                return out[:, 0]

        return out

    @io.timeit
    def get_cell_from_class(
            self,
            ell,
            params,
            name='TT',
            precision=0,
            squeeze=False,
            verbose=False,
            timeit=False):
        """Call CLASS to compute CMB angular spectra ``C_ell``.

        Args:
            ell (int | sequence[int] | numpy.ndarray): Multipoles.
            params (dict[str, float]): Cosmological parameters to forward to
                CLASS.
            name (str): Cl spectrum identifier (``TT``, ``TE``, ``EE``,
                ``Tp``, ``pp``, ``BB``). Lensed spectra are selected when
                present.
            precision (int | dict[str, float]): Same semantics as
                :meth:`get_pk_from_class`.
            squeeze (bool): Return scalars for 1-element outputs.
            verbose (bool): Forwarded to the spectrum wrapper.
            timeit (bool): When True, measure call duration.

        Returns:
            numpy.ndarray or float: Dimensionless
            ``\\ell(\\ell+1)C_\\ell/(2\\pi)`` values.
        """

        # Select correct spectrum
        try:
            spectrum = self._spectra['cl_{}_lensed'.format(name)]
        except KeyError:
            spectrum = self._spectra['cl_{}'.format(name)]

        # Get output
        out = spectrum.get_from_class(
            ell, params, precision=precision, verbose=verbose)

        # Squeeze dimensions
        if squeeze and out.shape == (1,):
            return out[0]

        return out

    def print_info(self, name=None):
        """
        Print summary info for each spectrum emulator.
        Args:
            name (str | None): When provided, print info only for the named
                spectrum.
        """
        io._print_info(self._spectra, self._params, name=name)
        return
