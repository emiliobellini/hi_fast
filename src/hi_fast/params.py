import numpy as np
import scipy.optimize as opt
from . import io as io


class Params(object):

    @io.timeit
    def __init__(self, spectrum, spectra, timeit=False):
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

        pass

    def _conversion_rules(self):
        """
        Define conversion rules for cosmological parameters.
        Returns two dictionaries:
        - standard_rules: direct conversions
        - shooting_rules: conversions requiring shooting methods.
        Both dictionaries have the format:
        {'derived_param': {
            'base': 'base_param',
            'func': function_to_convert}}
        """

        standard_rules = {
            'H0': {
                'base': 'h',
                'func': lambda H0: H0 / 100.
            },
            'A_s': {
                'base': 'ln_A_s_1e10',
                'func': lambda A_s: np.log(A_s * 1e10)
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
        """
        Print the input parameters for all loaded emulators.
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

    def print_params(self):
        """
        Print parameters info for all or a given spectrum emulator.
        Arguments:
        - name (str, default: None): name of the spectrum emulator.
        """
        io.print_level(1, '--- Spectrum: {} ---'.format(self._spectrum_name))
        for par in self._derived:
            print('  - {}.  Can be derived from: {}'.format(
                par, self._derived[par]))
        return

    def print_ranges(self):
        """
        Print parameter ranges.
        """
        io.print_level(1, '--- Spectrum: {} ---'.format(self._spectrum_name))
        for par in self._required:
            if par in self._ranges:
                low, high = self._ranges[par]
                print('  - {}: [{}, {}]'.format(par, low, high))
            else:
                print('  - {}: unbounded'.format(par))
        return

    def _shooting(self, params, names, targets, rules):
        """
        Perform shooting method to derive parameters.
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
        """
        Check that the parameters passed are
        exactly those expected (not missing, not defined multiple times).
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

    def _check_output_param_values(self, params):
        """
        Check that the parameters after conversion are
        within the emulator range.
        """
        for par in params:
            if par not in self._emu:
                continue
            low, high = self._ranges[par]
            in_range = low <= params[par] <= high
            if not in_range:
                raise Exception(
                    'Parameter {} = {} out of range [{} - {}]'
                    ''.format(par, params[par], low, high))
        return

    def get(
            self,
            params,
            check_params_names=True,
            check_params_values=True,
            verbose=False):
        """
        Get parameters for the emulator. Arguments:
        - params (dict): input parameters (can be derived);
        - check_params_names (bool, default: True): check parameter names;
        - check_params_values (bool, default: True): check parameter values;
        - verbose (bool, default: False): print input and converted parameters.
        Returns:
        - out (dict): parameters for the emulator."""

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
                out[base] = func(out[par])
                out.pop(par)
            if par in shooting_rules:
                shoot_on_name.append(par)
                shoot_on_val.append(out[par])
                out.pop(par)
        if len(shoot_on_name) > 0:
            out = self._shooting(
                out, shoot_on_name, shoot_on_val, shooting_rules)

        if check_params_values is True:
            self._check_output_param_values(out)

        if verbose:
            self._print_verbose(params, out)

        return out
