import numpy as np
from . import io as io


class Params(object):

    @io.timeit
    def __init__(
            self,
            params,
            spectrum,
            emus_list=None,
            convert=True,
            add_defaults=True,
            check_names=True,
            check_values=True,
            raise_error_names=True,
            raise_error_values=True,
            timeit=False):

        self._in = params
        self._spectrum = spectrum
        self._out = params.copy()

        if convert is True:
            self._out = self._convert_cosmo_params(
                self._out, emus_list=emus_list)

        if add_defaults is True:
            self._out = Params._add_defaults(
                self._out,
                required_params=self._spectrum.input_params_names)

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

    def _conversion_rules(self, params, emus_list=None):
        """
        Define conversion rules for cosmological parameters.
        """

        # NOTE: make sure that if a conversion rule depends on the value of
        # another parameter, the latter is already converted when needed.
        if emus_list is None:
            fs8 = None
        else:
            fs8 = emus_list['pk_cb'].get_As_from_sigma_8

        conversion_rules = [
            # base param, new param, conversion function
            ('h', 'H0', lambda H0: H0/100.),
            ('ln_A_s_1e10', 'A_s', lambda A_s: np.log(A_s*1e10)),
            ('ln_A_s_1e10', 'sigma8_cb', lambda sigma8_cb: fs8(
                sigma8_cb, params))
        ]

        return conversion_rules

    def _convert_cosmo_params(self, params, emus_list=None):
        """
        Convert cosmological parameters if needed.
        """

        out = params.copy()
        conversion_rules = self._conversion_rules(out, emus_list=emus_list)

        for base, new, func in conversion_rules:
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

    @staticmethod
    def _add_defaults(params, required_params):
        defaults = {
            'k_pivot': 0.05  # In 1/Mpc units
        }
        for name, val in defaults.items():
            if name in required_params:
                if name not in params:
                    params[name] = val
        return params
