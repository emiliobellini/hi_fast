from . import io as io
from . import spectra as sp
from .params import Params


class HiFast(object):
    """
    Main hi_fast class.
    """

    def __init__(self, name, root='emu', timeit=False, verbose=False):
        """
        Init hi_fast. Arguments:
        - name (str, default: None): name of the model to load. If None
          one has to load manually using the HiFast.load method;
        - root (str, default: emu): root where the emulators are stored;
        - timeit (bool, default: False): print loading time;
        - verbose (bool, default: False): verbosity.
        """
        # Load spectra emulators
        self._spectra = self._load(
            name, root=root, timeit=timeit, verbose=verbose)
        # Init parameters handlers
        self._params = {spec.name: Params(spec, self._spectra)
                        for spec in self._spectra.values()}

        pass

    @io.timeit
    def _load(self, name, root, timeit=False, verbose=False):
        """
        Load all emulators. Arguments:
        - name (str, default: None): name of the model to load. If None
          one has to load manually using the HiFast.load method;
        - root (str, default: emu): root where the emulators are stored;
        - timeit (bool, default: False): print execution time;
        - verbose (bool, default: False): verbosity.
        """
        # Initialize spectra dictionary
        spectra = {}
        # Load emulators as dictionary
        for file in io.Folder(name, root=root).list_files():
            # Read content emu
            emufile = io.EmuFile(file)

            # Check that it is a dictionary file
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
            squeeze=False,
            verbose=False,
            timeit=False):
        """
        Main method to get the power spectrum P(k, z) at some z and k.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters;
        - name (str, default: m): name of the spectrum. Options:
            - m: total matter;
            - cb: CDM+baryons;
            - weyl: Weyl potential.
        - nonlinear (bool, default: False);
        - check_params_names (bool, default: True): check parameter names;
        - check_params_values (bool, default: True): check parameter values;
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - verbose (bool, default: False): verbosity;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct spectrum
        spectrum = self._spectra['pk_{}'.format(name)]

        # Get parameters
        params = self._params[spectrum.name].get(
            params,
            check_params_names=check_params_names,
            check_params_values=check_params_values,
            verbose=verbose)

        # Get output
        out = spectrum.get(k, z, params)

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
            nonlinear=False,
            check_params_names=True,
            check_params_values=True,
            squeeze=False,
            verbose=False,
            timeit=False):
        """
        Main method to get the growth rate
        f(k, z) = dln P(k, z)/dln a at some z and k.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters;
        - name (str, default: m): name of the spectrum. Options:
            - m: total matter;
            - cb: CDM+baryons;
            - weyl: Weyl potential.
        - nonlinear (bool, default: False);
        - check_params_names (bool, default: True): check parameter names;
        - check_params_values (bool, default: True): check parameter values;
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - verbose (bool, default: False): verbosity;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc. f(k, z) is dimensionless.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct spectrum
        spectrum = self._spectra['fk_{}'.format(name)]

        # Get parameters
        params = self._params[spectrum.name].get(
            params,
            check_params_names=check_params_names,
            check_params_values=check_params_values,
            verbose=verbose)

        # Get output
        out = spectrum.get(k, z, params)

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
    def get_cell(
            self,
            ell,
            params,
            name='TT',
            check_params_names=True,
            check_params_values=True,
            squeeze=False,
            verbose=False,
            timeit=False):
        """
        Main method to get the Cell at some ell. As in Class,
        we emulate the dimensionless Cell using:

        ell*(ell+1.)/2./pi * Cl

        Arguments:
        - ell (float, list or array): single/list of ells;
        - params (dict): dictionary with the cosmo parameters;
        - name (str, default: m): name of the spectrum. Options:
            - TT;
            - TE;
            - EE;
            - Tp;
            - pp;
            - BB;
            where:
            - T: temperature
            - E, B: polarization
            - p: lensing potential
        - check_params_names (bool, default: True): check parameter names;
        - check_params_values (bool, default: True): check parameter values;
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - verbose (bool, default: False): verbosity;
        - timeit (bool, default: False): print execution time.
        """

        # Select correct spectrum
        try:
            spectrum = self._spectra['cl_{}_lensed'.format(name)]
        except KeyError:
            spectrum = self._spectra['cl_{}'.format(name)]

        # Get parameters
        params = self._params[spectrum.name].get(
            params,
            check_params_names=check_params_names,
            check_params_values=check_params_values,
            verbose=verbose)

        # Get output
        out = spectrum.get(ell, params)

        # Squeeze dimensions
        if squeeze and out.shape == (1,):
            return out[0]

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
        """
        Main method to get the power spectrum P(k, z) at some z and k
        from Class.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters;
        - name (str, default: m): name of the spectrum. Options:
            - m: total matter;
            - cb: CDM+baryons;
            - weyl: Weyl potential.
        - precision (0, 1, 2, or dict): for default precisions use:
            - 0: standard class precision;
            - 1: precision parameters used for this emulator;
            - 2: high precision parameters.
            Otherwise, it is possible to pass directly a
            dictionary of precision parameters.
        - nonlinear (bool, default: False);
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - verbose (bool, default: False): verbosity;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
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
        """
        Main method to get the growth rate
        f(k, z) = dln P(k, z)/dln a at some z and k from Class.
        Arguments:
        - k (float, list or array): single/list of wavenumbers;
        - z (float, list or array): single/list of redshift;
        - params (dict): dictionary with the cosmo parameters;
        - name (str, default: m): name of the spectrum. Options:
            - m: total matter;
            - cb: CDM+baryons;
            - weyl: Weyl potential.
        - precision (0, 1, 2, or dict): for default precisions use:
            - 0: standard class precision;
            - 1: precision parameters used for this emulator;
            - 2: high precision parameters.
            Otherwise, it is possible to pass directly a
            dictionary of precision parameters.
        - nonlinear (bool, default: False);
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - verbose (bool, default: False): verbosity;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc.
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
        """
        Main method to get the Cell at some ell from Class.
        As in Class, we emulate the dimensionless Cell using:

        ell*(ell+1.)/2./pi * Cl

        Arguments:
        - ell (float, list or array): single/list of ells;
        - params (dict): dictionary with the cosmo parameters;
        - name (str, default: m): name of the spectrum. Options:
            - TT;
            - TE;
            - EE;
            - Tp;
            - pp;
            - BB;
            where:
            - T: temperature
            - E, B: polarization
            - p: lensing potential
        - precision (0, 1, 2, or dict): for default precisions use:
            - 0: standard class precision;
            - 1: precision parameters used for this emulator;
            - 2: high precision parameters.
            Otherwise, it is possible to pass directly a
            dictionary of precision parameters;
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - verbose (bool, default: False): verbosity;
        - timeit (bool, default: False): print execution time.
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
