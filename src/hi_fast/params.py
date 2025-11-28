import numpy as np


class Params(object):

    def __init__(
            self,
            spectrum,
            params,
            convert=True,
            add_defaults=True,
            check_names=True,
            check_values=True,
            raise_error_names=True,
            raise_error_values=True):

        self._spectrum = spectrum
        self._in = params

        if convert is True:
            self._out = self._convert_cosmo_params(self._in)

        if add_defaults is True:
            self._out = self.add_defaults(self._out)

        if check_names:
            self._check_param_names(
                self._out, raise_error=raise_error_names)

        if check_values:
            self._check_param_values(
                self._out, raise_error=raise_error_values)

        # Prepare emulator parameters
        if self._spectrum.name.startswith('cl_'):
            self.emu = [self._out[name] for name in self._spectrum.x_names]
        else:
            self.emu = [
                self._out[name] if (name != 'z_pk') else None
                for name in self._spectrum.x_names]
            self.idx_z_pk = self._spectrum.x_names.index('z_pk')
        pass

    def _conversion_rules(self):
        rules = [
            # base param, new param, conversion function
            ('h', 'H0', lambda H0: H0/100.),
            ('ln_A_s_1e10', 'A_s', lambda A_s: np.log(A_s*1e10)),
            ('ln_A_s_1e10', 'sigma_8',
             lambda A_s: self._spectrum.sigma8_from_As(A_s)),
        ]
        return rules

    def _convert_cosmo_params(self, params):
        """
        Convert cosmological parameters if needed.
        """
        out = params.copy()

        for base, new, func in self._conversion_rules():
            if base not in params and new in out:
                out[base] = func(out[new])
                out.pop(new)
        return out

    def _check_param_names(self, params, raise_error=True):
        """
        Check that the parameters passed are
        exactly those expected (not missing, not undefined).
        """
        # Expected parameters
        expected = self._spectrum.input_params_names
        expected.sort()
        # Given parameters
        given = list(params.keys())
        given.sort()

        if given != expected:
            if raise_error is True:
                raise Exception(
                    'I expected parameters {}, and I got {}!'
                    ''.format(expected, given))
                return False
        return True

    def _check_param_values(self, params, raise_error=True):
        """
        Check that the parameters are within the emulator range.
        """
        for par in params:
            if par not in self._spectrum.x_names:
                continue
            idx_par = self._spectrum.x_names.index(par)
            low, high = self._spectrum.x_ranges[idx_par]
            in_range = low <= params[par] <= high
            if not in_range:
                if raise_error is True:
                    raise Exception(
                        'Parameter {} = {} out of range [{} - {}]'
                        ''.format(par, params[par], low, high))
                else:
                    return False
        return True

    def get_values_emu(self, **kwargs):
        if self._spectrum.name.startswith('cl_'):
            return self.emu
        else:
            self.emu[self.idx_z_pk] = kwargs['z_pk']
        return self.emu

    def add_defaults(self, params):
        defaults = {
            'k_pivot': 0.05  # In 1/Mpc units
        }
        for name, val in defaults.items():
            if name in self._spectrum.input_params_names:
                if name not in params:
                    params[name] = val
        return params
