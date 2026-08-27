import numpy as np
import scipy.optimize as opt
from . import io as io


class Params(object):
    """Handle parameter validation and conversions for a spectrum."""

    @io.timeit
    def __init__(self, spectrum, spectra, timeit=False):
        """Create a parameter handler for a given emulator.

        Args:
            spectrum (sp.Spectrum): Spectrum metadata describing the required
                inputs, emulator axes, and allowed ranges.
            spectra (dict[str, sp.Spectrum]): Mapping from spectrum names to
                spectrum instances, used for cross-spectrum conversions.
            timeit (bool): Included for the decorator signature; ignored.
        """
        # Name of the spectrum
        self._spectrum_name = spectrum.name
        # List of spectra classes
        self._spectra = spectra
        # List of required input parameters names
        self._required = spectrum.input_params_names
        # List of required emulator parameters names
        self._emu = spectrum.x_names
        # List of additional parameters (not used by the emulator)
        self._additional = [p for p in self._required if p not in self._emu]
        # For each of the emulator parameters, get its range
        self._ranges = {
            name: spectrum.x_ranges[spectrum.x_names.index(name)]
            for name in self._emu}
        self._ranges_by_region = {
            region: {
                name: ranges[spectrum.x_names.index(name)]
                for name in self._emu
            }
            for region, ranges in spectrum.x_ranges_by_region.items()
        }
        # For each of the required parameters, get possible derived ones
        standard_rules, shooting_rules = self._conversion_rules()
        self._derived = {}
        for name in self._required:
            self._derived[name] = [name]
            for der in standard_rules:
                if standard_rules[der]['base'] == name:
                    self._derived[name].append(der)
            for der in shooting_rules:
                if shooting_rules[der]['base'] == name:
                    self._derived[name].append(der)
            self._derived[name].sort()

    def _conversion_rules(self):
        """Return dictionaries describing parameter conversions.

        Returns:
            tuple[dict, dict]: ``(standard_rules, shooting_rules)`` where each
            entry maps a derived-parameter name to a dictionary containing the
            target base parameter and the conversion callable.
        """

        # Standard conversions rules. Here the ordering is important,
        # e,g, omega_m depends on h, which can be derived from H0.
        # Them it is necessary to first convert H0 to h, and then
        # omega_m to Omega_m.
        standard_rules = {
            'H0': {
                'base': 'h',
                'func': lambda H0, _: H0 / 100.
            },
            'A_s': {
                'base': 'ln_A_s_1e10',
                'func': lambda A_s, _: np.log(A_s * 1e10)
            },
            'omega_m': {
                'base': 'Omega_m',
                'func': lambda omega_m, current: omega_m / current['h']**2
            },
            'omega_b': {
                'base': 'Omega_b',
                'func': lambda omega_b, current: omega_b / current['h']**2
            },
        }

        shooting_rules = {
            'sigma8_cb': {
                'base': 'ln_A_s_1e10',
                'func': self._spectra['pk_cb'].get_sigma8_from_params,
                'guess': 3.
            },
            'sigma8_m': {
                'base': 'ln_A_s_1e10',
                'func': self._spectra['pk_m'].get_sigma8_from_params,
                'guess': 3.
            },
            'S8_cb': {
                'base': 'ln_A_s_1e10',
                'func': self._spectra['pk_cb'].get_S8_from_params,
                'guess': 3.
            },
            'S8_m': {
                'base': 'ln_A_s_1e10',
                'func': self._spectra['pk_m'].get_S8_from_params,
                'guess': 3.
            },
        }

        return standard_rules, shooting_rules

    def _print_verbose(self, in_params, out_params):
        """Print the provided and converted parameter values.

        Args:
            in_params (dict[str, float]): User-supplied parameters, including
                derived values.
            out_params (dict[str, float]): Parameters after conversion to the
                emulator basis.
        """
        io.info('Input parameters for {}:'.format(self._spectrum_name))
        for par in self._required:
            msg = ' - {}: {}'.format(par, out_params[par])
            standard_rules, shooting_rules = self._conversion_rules()
            for der in standard_rules:
                if der in in_params:
                    if standard_rules[der]['base'] == par:
                        msg += '  (from {} = {})'.format(der, in_params[der])
            for der in shooting_rules:
                if der in in_params:
                    if shooting_rules[der]['base'] == par:
                        msg += '  (from {} = {})'.format(der, in_params[der])
            print(msg)
        return

    def _shooting(self, params, names, targets, rules):
        """Apply a multidimensional shooting method for derived params.

        Args:
            params (dict[str, float]): Current parameter dictionary.
            names (list[str]): Derived parameters to solve for.
            targets (list[float]): Target values for each entry in ``names``.
            rules (dict): Shooting rule metadata (see ``_conversion_rules``).

        Returns:
            dict[str, float]: Updated parameters dictionary with solved base
            values.
        """

        out = params.copy()

        def solve_all(x, params, names, targets, rules):
            # Assign values
            for nname, name in enumerate(names):
                params[rules[name]['base']] = x[nname]
            # Compute values
            vals = []
            for name in names:
                func = rules[name]['func']
                val = func(params)
                vals.append(val - targets[names.index(name)])
            return np.array(vals)

        # Initial guess
        guess = [rules[name]['guess'] for name in names]

        # Find roots
        sol = opt.root(solve_all, x0=guess, args=(out, names, targets, rules))

        for name in names:
            out[rules[name]['base']] = sol.x[names.index(name)]

        return out

    def _check_input_param_names(self, params):
        """Verify that exactly one representative for each parameter is set.

        Args:
            params (dict[str, float]): User-provided parameters.

        Raises:
            Exception: If a required parameter is missing or provided more
            than once via different derived forms.
        """

        for par in self._required:
            common = list(set(self._derived[par]) & set(params.keys()))
            if len(common) == 0:
                raise Exception(
                    'One parameter between {} is required, but not provided!'
                    ''.format(self._derived[par]))
            if len(common) > 1:
                raise Exception(
                    'Only one parameter between {} is allowed, but got {}!'
                    ''.format(self._derived[par], common))
        return

    def _check_output_param_values(self, params, trusted_region='ext'):
        """Ensure converted parameters lie inside one trusted region.

        Args:
            params (dict[str, float]): Parameter dictionary after conversion.
                It may also contain emulator coordinates such as ``z_pk``.
            trusted_region (str): Named emulator region: ``thin``, ``std``,
                or ``ext``.

        Raises:
            ValueError: If the region is unknown or any emulator parameter
                lies outside it.
        """
        if trusted_region not in self._ranges_by_region:
            raise ValueError(
                'trusted_region must be one of {}; got {!r}'.format(
                    sorted(self._ranges_by_region), trusted_region))

        ranges = self._ranges_by_region[trusted_region]
        for par in params:
            if par not in self._emu:
                continue
            low, high = ranges[par]
            in_range = low <= params[par] <= high
            if not in_range:
                raise ValueError(
                    'Parameter {} = {} is outside the {} trusted region '
                    '[{} - {}]'.format(
                        par, params[par], trusted_region, low, high))
        return

    def is_in_bounds(self, params, trusted_region='ext'):
        """Return whether converted parameters lie in a trusted region."""
        try:
            self._check_output_param_values(
                params, trusted_region=trusted_region)
        except ValueError:
            return False
        return True

    def get(
            self,
            params,
            check_params_names=True,
            check_params_values=True,
            trusted_region='ext',
            verbose=False):
        """Convert user parameters into the emulator basis.

        Args:
            params (dict[str, float]): Input parameters, possibly including
                derived quantities (e.g., ``H0`` instead of ``h``).
            check_params_names (bool): When True, ensure exactly one
                representative per required parameter is provided.
            check_params_values (bool): When True, verify the converted
                parameters lie within the emulator training ranges.
            trusted_region (str): Named range used when validating values:
                ``thin``, ``std``, or ``ext``.
            verbose (bool): When True, print the provided and converted
                parameters.

        Returns:
            dict[str, float]: Parameters aligned with the emulator inputs.
        """

        out = params.copy()

        if check_params_names is True:
            self._check_input_param_names(params)

        # Convert parameters
        standard_rules, shooting_rules = self._conversion_rules()
        shoot_on_name = []
        shoot_on_val = []
        for par in params:
            if par in standard_rules:
                base = standard_rules[par]['base']
                func = standard_rules[par]['func']
                out[base] = func(out[par], out)
                out.pop(par)
            if par in shooting_rules:
                shoot_on_name.append(par)
                shoot_on_val.append(out[par])
                out.pop(par)
        if len(shoot_on_name) > 0:
            out = self._shooting(
                out, shoot_on_name, shoot_on_val, shooting_rules)

        if check_params_values is True:
            self._check_output_param_values(
                out, trusted_region=trusted_region)

        if verbose:
            self._print_verbose(params, out)

        return out
