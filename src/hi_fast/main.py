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

    @staticmethod
    def _grid_batch(params, redshifts, first, last):
        """Return one flattened slice of a cosmology-redshift grid."""
        indices = np.arange(first, last)
        n_redshifts = len(redshifts)
        param_indices = indices // n_redshifts
        redshift_indices = indices % n_redshifts
        batch_params = [params[index] for index in param_indices]
        return batch_params, redshifts[redshift_indices]

    @staticmethod
    def _normalize_params(params, names):
        """Return parameter dictionaries from mappings or ordered arrays.

        Array columns must follow ``names`` exactly. A one-dimensional array
        represents one cosmology and a two-dimensional array represents a
        batch with one cosmology per row.
        """
        if isinstance(params, dict):
            return [params.copy()]
        if (isinstance(params, (list, tuple)) and params
                and all(isinstance(row, dict) for row in params)):
            return [row.copy() for row in params]

        values = np.asarray(params)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2:
            raise ValueError('params must be a dictionary, a sequence of '
                             'dictionaries, or a one/two-dimensional array')
        if values.shape[1] != len(names):
            raise ValueError(
                'Expected {} parameter columns in this order: {}; got {}'
                .format(len(names), names, values.shape[1]))
        return [dict(zip(names, row)) for row in values]

    def get_params_names(self, spectrum):
        """Return the ordered parameter names expected by array input.

        Args:
            spectrum (str): Full observable name, such as ``pk_m``, ``fk_cb``,
                or ``cl_TT_lensed``.

        Returns:
            list[str]: Array-column names in the exact order accepted by the
            corresponding ``get_*`` method. ``z_pk`` is omitted because
            redshift is passed separately.

        Raises:
            ValueError: If the requested observable is unavailable.
        """
        if spectrum in self._spectra:
            selected = self._spectra[spectrum]
            names = selected.input_params_names
        elif spectrum.startswith('fk_'):
            pk_name = 'pk_{}'.format(spectrum.removeprefix('fk_'))
            if pk_name not in self._spectra:
                raise ValueError('Spectrum {} is not available'.format(
                    spectrum))
            names = self._spectra[pk_name].input_params_names
            # Growth rates derived from P(k,z) fix primordial shape and
            # amplitude to their reference values.
            names = [name for name in names
                     if name not in ('ln_A_s_1e10', 'n_s')]
        else:
            raise ValueError('Spectrum {} is not available. Choose from {}'
                             .format(spectrum,
                                     sorted(self._spectra.keys())))
        return [name for name in names if name != 'z_pk']

    def _get_cell_spectrum(self, name):
        """Resolve a public CMB selector to an available emulator."""
        lensed_name = 'cl_{}_lensed'.format(name)
        raw_name = 'cl_{}'.format(name)
        if lensed_name in self._spectra:
            return self._spectra[lensed_name]
        if raw_name in self._spectra:
            return self._spectra[raw_name]

        available = sorted({
            key.removeprefix('cl_').removesuffix('_lensed')
            for key in self._spectra if key.startswith('cl_')
        })
        raise ValueError(
            'CMB spectrum {!r} is not available. Choose from {}'
            .format(name, available))

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
        if verbose:
            io.title('HiFast: loading {} emulators'.format(name))
        # Initialize spectra dictionary
        spectra = {}
        # Load emulators as dictionary
        for file in io.Folder(name, root=root).list_files():
            if not file.endswith('.joblib'):
                continue
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
            io.info('Loaded {} emulators for {} ----> {}'.format(
                len(spectra), name, ', '.join(sorted(spectra.keys()))))
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
            timeit=False,
            paired=False):
        """Evaluate the emulator power spectrum ``P(k, z)``.

        Args:
            k (float | sequence[float] | numpy.ndarray): Wavenumbers in
                units of h/Mpc.
            z (float | sequence[float] | numpy.ndarray): Redshifts to
                evaluate.
            params (dict[str, float] | sequence[dict[str, float]] |
                numpy.ndarray): One cosmology, a sequence of cosmology
                dictionaries, or an array with shape ``(n_params,)`` or
                ``(n_cosmologies, n_params)``. Array columns must follow
                ``spectrum.input_params_names`` with ``z_pk`` excluded.
            name (str): Spectrum identifier; ``m`` (total matter), ``cb``
                (CDM plus baryons), or ``weyl`` (Weyl potential).
            nonlinear (bool): Placeholder flag; nonlinear emulation is not
                implemented yet.
            check_params_names (bool): When True, ensure the provided
                dictionary contains exactly one representative for every
                required parameter.
            check_params_values (bool): When True, validate the converted
                parameters lie inside the emulator training domain.
            batch_size (int | None): Maximum cosmology-redshift pairs
                evaluated in one emulator call. ``None`` evaluates the
                complete Cartesian product in one model call.
            squeeze (bool): When True, drop singleton dimensions in the
                result for convenience.
            verbose (bool): When True, echo both provided and converted
                parameters.
            timeit (bool): When True, report evaluation time (handled by the
                decorator).
            paired (bool): When True, evaluate ``params[i]`` only at
                ``z[i]`` instead of evaluating the Cartesian product. The
                two inputs must then have equal lengths.

        Returns:
            numpy.ndarray or float: Power spectra in units of ``(Mpc/h)^3``
            with shape ``(n_cosmologies, n_redshifts, n_k)`` by default, or
            ``(n_pairs, n_k)`` when ``paired=True``, before optional
            squeezing.

        Raises:
            ValueError: If ``nonlinear`` is True.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct spectrum
        spectrum = self._spectra['pk_{}'.format(name)]

        param_names = [name for name in spectrum.input_params_names
                       if name != 'z_pk']
        params = self._normalize_params(params, param_names)
        z = np.atleast_1d(z)
        if z.ndim != 1 or not len(z):
            raise ValueError('z must be a scalar or a non-empty 1D array')
        converted = [self._params[spectrum.name].get(
            row,
            check_params_names=check_params_names,
            check_params_values=check_params_values,
            verbose=verbose) for row in params]

        n_cosmologies = len(converted)
        n_redshifts = len(z)
        outputs = []
        if paired and n_cosmologies != n_redshifts:
            raise ValueError('params and z must have equal lengths when '
                             'paired=True')
        total = n_cosmologies if paired else n_cosmologies * n_redshifts
        for first, last in self._batch_ranges(total, batch_size):
            if paired:
                batch_params = converted[first:last]
                batch_z = z[first:last]
            else:
                batch_params, batch_z = self._grid_batch(
                    converted, z, first, last)
            outputs.append(spectrum.get(k, batch_z, batch_params))
        out = np.concatenate(outputs, axis=0)
        if not paired:
            out = out.reshape(
                n_cosmologies, n_redshifts, np.atleast_1d(k).size)

        # Squeeze dimensions
        if squeeze:
            return out.squeeze()

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
            timeit=False,
            paired=False):
        """Evaluate growth rates for one cosmology or a parameter batch.

        ``params`` accepts a dictionary, a sequence of dictionaries, or a
        one/two-dimensional array. Array columns follow the direct growth-rate
        emulator's ``input_params_names`` order, excluding ``z_pk``. Every
        cosmology is evaluated at every value in ``z``. With ``paired=True``,
        only ``params[i]`` at ``z[i]`` is evaluated and their lengths must be
        equal. ``batch_size`` limits the number of cosmology-redshift pairs
        per model call. The unsqueezed output shape is
        ``(n_cosmologies, n_redshifts, n_k)`` by default and
        ``(n_pairs, n_k)`` in paired mode.
        """
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        if get_from_pk is False and 'fk_{}'.format(name) not in self._spectra:
            get_from_pk = True
            io.warning('No emulator for fk_{} found; computing from Pk instead'
                       ''.format(name))

        if 'fk_{}'.format(name) in self._spectra:
            input_spectrum = self._spectra['fk_{}'.format(name)]
            param_names = [key for key in input_spectrum.input_params_names
                           if key != 'z_pk']
        else:
            input_spectrum = self._spectra['pk_{}'.format(name)]
            param_names = [key for key in input_spectrum.input_params_names
                           if key not in ('z_pk', 'ln_A_s_1e10', 'n_s')]
        params = self._normalize_params(params, param_names)
        z = np.atleast_1d(z)
        if z.ndim != 1 or not len(z):
            raise ValueError('z must be a scalar or a non-empty 1D array')

        if get_from_pk is True:
            spectrum = self._spectra['pk_{}'.format(name)]
            for row in params:
                row['A_s'] = spectrum.ref['params']['ln_A_s_1e10']
                row['n_s'] = spectrum.ref['params']['n_s']
        else:
            spectrum = self._spectra['fk_{}'.format(name)]

        converted = [self._params[spectrum.name].get(
            row,
            check_params_names=check_params_names,
            check_params_values=check_params_values,
            verbose=verbose) for row in params]
        n_cosmologies = len(converted)
        n_redshifts = len(z)
        outputs = []
        if paired and n_cosmologies != n_redshifts:
            raise ValueError('params and z must have equal lengths when '
                             'paired=True')
        total = n_cosmologies if paired else n_cosmologies * n_redshifts
        for first, last in self._batch_ranges(total, batch_size):
            if paired:
                batch_params = converted[first:last]
                batch_z = z[first:last]
            else:
                batch_params, batch_z = self._grid_batch(
                    converted, z, first, last)
            if get_from_pk is True:
                outputs.append(spectrum.get_fk(
                    k, batch_z, batch_params))
            else:
                outputs.append(spectrum.get(
                    k, batch_z, batch_params))
        out = np.concatenate(outputs, axis=0)
        if not paired:
            out = out.reshape(
                n_cosmologies, n_redshifts, np.atleast_1d(k).size)

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
            params (dict[str, float] | sequence[dict[str, float]] |
                numpy.ndarray): One or more cosmologies. Array columns must
                follow ``spectrum.input_params_names``.
            name (str): Cl spectrum identifier (``TT``, ``TE``, ``EE``,
                ``Tp``, ``pp``, ``BB``). Lensed versions are used when
                available.
            check_params_names (bool): Validate parameter coverage.
            check_params_values (bool): Validate converted parameter ranges.
            batch_size (int | None): Maximum cosmologies evaluated in one
                emulator call. ``None`` evaluates the complete input batch.
            squeeze (bool): Return scalars for 1-element outputs.
            verbose (bool): When True, print provided and derived
                parameters.
            timeit (bool): When True, measure call duration.

        Returns:
            numpy.ndarray or float: Dimensionless angular spectra with shape
            ``(n_cosmologies, n_ell)`` before optional squeezing.
        """

        # Select correct spectrum
        spectrum = self._get_cell_spectrum(name)

        param_names = [key for key in spectrum.input_params_names
                       if key != 'z_pk']
        params = self._normalize_params(params, param_names)
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
            ``(Mpc/h)^3`` and shape ``(1, n_redshifts, n_k)`` before
            optional squeezing.

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
        n_k = np.atleast_1d(k).size
        n_z = np.atleast_1d(z).size
        out = np.asarray(out).reshape(n_k, n_z).T[np.newaxis, :, :]

        # Squeeze dimensions
        if squeeze:
            return out.squeeze()

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
            numpy.ndarray or float: Growth-rate values with shape
            ``(1, n_redshifts, n_k)`` before optional squeezing.

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
        n_k = np.atleast_1d(k).size
        n_z = np.atleast_1d(z).size
        out = np.asarray(out).reshape(n_k, n_z).T[np.newaxis, :, :]

        # Squeeze dimensions
        if squeeze:
            return out.squeeze()

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
            ``\\ell(\\ell+1)C_\\ell/(2\\pi)`` values with shape
            ``(1, n_ell)`` before optional squeezing.
        """

        # Select correct spectrum
        spectrum = self._get_cell_spectrum(name)

        # Get output
        out = spectrum.get_from_class(
            ell, params, precision=precision, verbose=verbose)
        out = np.asarray(out).reshape(1, np.atleast_1d(ell).size)

        # Squeeze dimensions
        if squeeze:
            return out.squeeze()

        return out

    def print_info(self, name=None, bounds=None, markdown=False, output=None):
        """
        Print summary info for each spectrum emulator.
        Args:
            name (str | None): When provided, print info only for the named
                spectrum.
            bounds (str | None): Optional trust region to display. Choose
                ``thin``, ``std``, or ``ext``. When omitted, all stored
                regions are shown.
            markdown (bool): When True, render Markdown instead of terminal
                tables.
            output (str | None): Optional file path used only with
                ``markdown=True``. When omitted, Markdown is printed.
        """
        return io._print_info(
            self._spectra,
            self._params,
            name=name,
            bounds=bounds,
            markdown=markdown,
            output=output)
