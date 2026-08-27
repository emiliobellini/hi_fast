"""Private request and extraction service for HiCLASS observables."""

import numpy as np

from ._class_cache import HiClassCache


class HiClassService:
    """Coordinate HiCLASS requests for a collection of spectra."""

    BACKGROUND_QUANTITIES = {
        'H': {
            'member': 'Hubble', 'at_z': True, 'units': 'km/s/Mpc'},
        'comoving_distance': {
            'member': 'comoving_distance', 'at_z': True, 'units': 'Mpc'},
        'angular_diameter_distance': {
            'member': 'angular_distance', 'at_z': True, 'units': 'Mpc'},
        'luminosity_distance': {
            'member': 'luminosity_distance', 'at_z': True, 'units': 'Mpc'},
        'growth_factor': {
            'member': 'scale_independent_growth_factor',
            'at_z': True,
            'units': 'dimensionless',
        },
        'growth_rate': {
            'member': 'scale_independent_growth_factor_f',
            'at_z': True,
            'units': 'dimensionless',
        },
        'age': {'member': 'age', 'at_z': False, 'units': 'Gyr'},
        'Omega_m': {
            'member': 'Omega_m', 'at_z': False, 'units': 'dimensionless'},
        'Omega_b': {
            'member': 'Omega_b', 'at_z': False, 'units': 'dimensionless'},
        'Omega_cdm': {
            'member': 'Omega_cdm', 'at_z': False,
            'units': 'dimensionless'},
        'Omega_k': {
            'member': 'Omega_k', 'at_z': False, 'units': 'dimensionless'},
        'Omega_r': {
            'member': 'Omega_r', 'at_z': False, 'units': 'dimensionless'},
        'Omega_g': {
            'member': 'Omega_g', 'at_z': False, 'units': 'dimensionless'},
        'Omega_nu': {
            'member': 'Omega_nu', 'at_z': False,
            'units': 'dimensionless'},
        'Omega_Lambda': {
            'member': 'Omega_Lambda', 'at_z': False,
            'units': 'dimensionless'},
    }

    def __init__(self, spectra, cache=None):
        self.spectra = spectra
        self.cache = HiClassCache() if cache is None else cache

    def clear(self):
        """Release the shared native HiCLASS computation."""
        self.cache.clear()

    def info(self):
        """Return diagnostics for the shared HiCLASS cache."""
        return self.cache.info()

    @classmethod
    def background_info(cls):
        """Return public background nomenclature from the registry."""
        return [{
            'name': name,
            'hiclassy': '{}(z)'.format(entry['member'])
            if entry['at_z'] else entry['member'],
            'input': 'z' if entry['at_z'] else '—',
            'units': entry['units'],
        } for name, entry in cls.BACKGROUND_QUANTITIES.items()]

    def get_cell_spectrum(self, name):
        """Resolve a public CMB selector to an available spectrum."""
        for spectrum_name in (
                'cl_{}_lensed'.format(name), 'cl_{}'.format(name)):
            if spectrum_name in self.spectra:
                return self.spectra[spectrum_name]
        available = sorted({
            key.removeprefix('cl_').removesuffix('_lensed')
            for key in self.spectra if key.startswith('cl_')
        })
        raise ValueError(
            'CMB spectrum {!r} is not available. Choose from {}'
            .format(name, available))

    @staticmethod
    def evaluate_pairs(
            spectrum, coordinates, params, cosmology_indices,
            coordinate_indices, selected, class_precision, verbose,
            growth=False):
        """Evaluate flattened coordinate pairs grouped by cosmology."""
        n_modes = np.atleast_1d(coordinates[0]).size
        out = np.empty((len(cosmology_indices), n_modes), dtype=float)
        selected_positions = np.flatnonzero(selected)
        for cosmology_index in np.unique(
                cosmology_indices[selected_positions]):
            positions = selected_positions[
                cosmology_indices[selected_positions] == cosmology_index]
            sample_indices = coordinate_indices[positions]
            sample_coordinates = coordinates[1][sample_indices]
            if growth and hasattr(spectrum, 'get_fk_from_class'):
                values = spectrum.get_fk_from_class(
                    coordinates[0], sample_coordinates,
                    params[cosmology_index], precision=class_precision,
                    verbose=verbose)
            else:
                values = spectrum.get_from_class(
                    coordinates[0], sample_coordinates,
                    params[cosmology_index], precision=class_precision,
                    verbose=verbose)
            out[positions] = np.asarray(values).reshape(
                n_modes, len(sample_coordinates)).T
        return out

    @staticmethod
    def format_kz_result(values, k, z, squeeze=False):
        """Convert native CLASS k-z orientation to the public shape."""
        n_k = np.atleast_1d(k).size
        n_z = np.atleast_1d(z).size
        result = np.asarray(values).reshape(n_k, n_z).T[np.newaxis, :, :]
        return result.squeeze() if squeeze else result

    def _background_request(self, params, precision=0, verbose=False):
        """Return minimal background parameters and a cache key."""
        spectrum = (self.spectra['pk_m'] if 'pk_m' in self.spectra
                    else next(iter(self.spectra.values())))
        class_args = {name: value
                      for name, value in spectrum.class_args.items()
                      if name not in spectrum.class_high_prec}
        key_params = spectrum._get_input_params_class(
            params, precision, class_args, verbose=verbose)
        unused = {
            'output', 'P_k_max_h/Mpc', 'P_k_max_1/Mpc', 'z_max_pk',
            'l_max_scalars', 'lensing', 'modes', 'k_pivot', 'YHe',
            'ln_A_s_1e10', 'A_s', 'n_s', 'tau_reio', 'sigma8', 'S8',
            'sigma8_m', 'S8_m',
        }
        unused.update(spectrum.class_high_prec)
        compute_params = {
            name: value for name, value in key_params.items()
            if name not in unused
            and (not name.endswith('_verbose')
                 or name in ('input_verbose', 'background_verbose'))
        }
        return compute_params, key_params

    @classmethod
    def _extract_background(cls, cosmo, z, quantities, squeeze):
        result = {}
        for quantity in quantities:
            entry = cls.BACKGROUND_QUANTITIES[quantity]
            member = getattr(cosmo, entry['member'])
            if entry['at_z']:
                value = np.asarray(member(z))
                if quantity == 'H':
                    value = value * 299792.458
                result[quantity] = value.squeeze() if squeeze else value
            else:
                result[quantity] = member
        return result

    def get_background(
            self, params, z, quantities, precision=0, squeeze=False,
            verbose=False):
        compute_params, key_params = self._background_request(
            params, precision=precision, verbose=verbose)
        with self.cache.use(
                compute_params, requirements={}, key_params=key_params
                ) as cosmo:
            return self._extract_background(
                cosmo, z, quantities, squeeze=squeeze)

    def get_background_table(self, params, precision=0, verbose=False):
        compute_params, key_params = self._background_request(
            params, precision=precision, verbose=verbose)
        return self.cache.get_background_table(
            compute_params, key_params=key_params)

    def parse_observables(self, observables):
        """Validate and flatten a structured multi-observable request."""
        if not isinstance(observables, dict) or not observables:
            raise ValueError('observables must be a non-empty dictionary')
        unknown_groups = set(observables) - {'pk', 'fk', 'cell'}
        if unknown_groups:
            raise ValueError('Unknown observable groups: {}'.format(
                sorted(unknown_groups)))

        requests = []
        expected = {'pk': {'k', 'z'}, 'fk': {'k', 'z'}, 'cell': {'ell'}}
        for kind, group in observables.items():
            if not isinstance(group, dict) or not group:
                raise ValueError(
                    'observables[{!r}] must be a non-empty dictionary'
                    .format(kind))
            for name, coordinates in group.items():
                if not isinstance(coordinates, dict):
                    raise ValueError(
                        'The {} {} request must be a dictionary'.format(
                            kind, name))
                if set(coordinates) != expected[kind]:
                    raise ValueError(
                        'The {} {} request requires exactly {}; got {}'
                        .format(kind, name, sorted(expected[kind]),
                                sorted(coordinates)))
                if kind == 'cell':
                    spectrum = self.get_cell_spectrum(name)
                    parsed = {'ell': np.atleast_1d(coordinates['ell'])}
                else:
                    spectrum_name = '{}_{}'.format(kind, name)
                    if kind == 'fk' and spectrum_name not in self.spectra:
                        spectrum_name = 'pk_{}'.format(name)
                    if spectrum_name not in self.spectra:
                        raise ValueError(
                            'Spectrum {}_{} is not available'.format(
                                kind, name))
                    spectrum = self.spectra[spectrum_name]
                    parsed = {
                        'k': np.atleast_1d(coordinates['k']),
                        'z': np.atleast_1d(coordinates['z']),
                    }
                if any(value.ndim != 1 or not value.size
                       for value in parsed.values()):
                    raise ValueError(
                        'Coordinates for {} {} must be non-empty scalars or '
                        'one-dimensional arrays'.format(kind, name))
                requests.append({
                    'kind': kind, 'name': name, 'spectrum': spectrum,
                    'coordinates': parsed,
                })
        return requests

    @staticmethod
    def common_params(params, precision, requests, verbose=False):
        """Build one compatible parameter dictionary for all requests."""
        common = None
        common_key = None
        outputs = set()
        limits = {key: None for key in HiClassCache._LIMIT_KEYS}
        for request in requests:
            spectrum = request['spectrum']
            coordinates = request['coordinates']
            if request['kind'] == 'cell':
                class_params, requirements = spectrum._get_class_request(
                    coordinates['ell'], params, precision=precision,
                    verbose=verbose)
            else:
                class_params, requirements = spectrum._get_class_request(
                    coordinates['k'], coordinates['z'], params,
                    precision=precision, verbose=verbose)
            base_key, _ = HiClassCache._request_parts(class_params)
            if common_key is None:
                common = class_params.copy()
                common.pop('output', None)
                for key in HiClassCache._LIMIT_KEYS:
                    common.pop(key, None)
                common_key = base_key
            elif base_key != common_key:
                raise ValueError(
                    'Requested observables require incompatible CLASS '
                    'settings')
            outputs.update(HiClassCache._outputs(
                requirements.get('output')))
            for key in HiClassCache._LIMIT_KEYS:
                value = HiClassCache._limit(requirements.get(key))
                if value is not None:
                    limits[key] = (value if limits[key] is None
                                   else max(limits[key], value))
        if outputs:
            common['output'] = ', '.join(sorted(outputs))
        for key, value in limits.items():
            if value is not None:
                common[key] = value
        return common

    def get_pk(self, k, z, params, name, precision, squeeze, verbose):
        spectrum = self.spectra['pk_{}'.format(name)]
        values = spectrum.get_from_class(
            k, z, params, precision=precision, verbose=verbose)
        return self.format_kz_result(values, k, z, squeeze=squeeze)

    def get_fk(self, k, z, params, name, precision, squeeze, verbose):
        fk_name = 'fk_{}'.format(name)
        if fk_name in self.spectra:
            spectrum = self.spectra[fk_name]
            values = spectrum.get_from_class(
                k, z, params, precision=precision, verbose=verbose)
        else:
            spectrum = self.spectra['pk_{}'.format(name)]
            values = spectrum.get_fk_from_class(
                k, z, params, precision=precision, verbose=verbose)
        return self.format_kz_result(values, k, z, squeeze=squeeze)

    def get_cell(self, ell, params, name, precision, squeeze, verbose):
        spectrum = self.get_cell_spectrum(name)
        values = spectrum.get_from_class(
            ell, params, precision=precision, verbose=verbose)
        result = np.asarray(values).reshape(1, np.atleast_1d(ell).size)
        return result.squeeze() if squeeze else result

    def get_many(
            self, params, observables, precision=0, squeeze=False,
            verbose=False):
        requests = self.parse_observables(observables)
        common = self.common_params(
            params, precision, requests, verbose=verbose)
        with self.cache.use(common):
            pass
        results = {kind: {} for kind in observables}
        for request in requests:
            kind = request['kind']
            name = request['name']
            coordinates = request['coordinates']
            if kind == 'cell':
                value = self.get_cell(
                    coordinates['ell'], params, name, precision, squeeze,
                    verbose)
            elif kind == 'pk':
                value = self.get_pk(
                    coordinates['k'], coordinates['z'], params, name,
                    precision, squeeze, verbose)
            else:
                value = self.get_fk(
                    coordinates['k'], coordinates['z'], params, name,
                    precision, squeeze, verbose)
            results[kind][name] = value
        return results
