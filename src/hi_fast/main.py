import hi_fast.io as io
import hi_fast.spectra as sp


class HiFast(object):
    """
    Main hi_fast class.
    """

    def __init__(self, name=None, root='emu', timeit=False, verbose=False):
        """
        Init hi_fast. Arguments:
        - name (str, default: None): name of the model to load. If None
          one has to load manually using the HiFast.load method;
        - root (str, default: emu): root where the emulators are stored;
        - timeit (bool, default: False): print loading time;
        - verbose (bool, default: False): verbosity.
        """
        self.root = root
        self._emu = {}
        if name is not None:
            self._load(name, root=self.root, timeit=timeit, verbose=verbose)

        pass

    @io.timeit
    def _load(self, name, root=None, timeit=False, verbose=False):
        """
        Load all emulators. Arguments:
        - name (str, default: None): name of the model to load. If None
          one has to load manually using the HiFast.load method;
        - root (str, default: emu): root where the emulators are stored;
        - timeit (bool, default: False): print execution time;
        - verbose (bool, default: False): verbosity.
        """
        # Fix root
        if root is None:
            root = self.root
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
            self._emu[content['name']] = sp.Spectrum.choose_one(**content)
        if verbose:
            io.info('Loaded emulators for {} model'.format(name))
        return

    def print_cosmo_params(self):
        """
        Print the cosmological parameters required by HiFast.
        """

        ref_emu = self._emu[list(self._emu.keys())[0]]

        for spectrum in self._emu.values():
            # Check that all spectra have the same cosmological params
            if spectrum.cosmo_params != ref_emu.cosmo_params:
                raise ValueError(
                    'Different cosmological parameters for different '
                    'spectra emulators')

        io.info('Cosmological parameters:')
        for param in ref_emu.cosmo_params:
            if param in ref_emu.derived_cosmo_params:
                derived_params = list(ref_emu.derived_cosmo_params[param])
                io.print_level(1, '{} (can be derived from: {})'.format(
                    param, ', '.join(derived_params)))
            else:
                io.print_level(1, '{}'.format(param))
        return

    @io.timeit
    def get_pk(
            self,
            k,
            z,
            params,
            name='m',
            nonlinear=False,
            squeeze=False,
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
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct emu
        emu = self._emu['pk_{}'.format(name)]

        # Get output
        out = emu.get(k, z, params)

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
            squeeze=False,
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
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc. f(k, z) is dimensionless.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct emu
        emu = self._emu['fk_{}'.format(name)]

        # Get output
        out = emu.get(k, z, params)

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
            squeeze=False,
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
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - timeit (bool, default: False): print execution time.

        """

        # Select correct emu
        try:
            emu = self._emu['cl_{}_lensed'.format(name)]
        except KeyError:
            emu = self._emu['cl_{}'.format(name)]

        # Get output
        out = emu.get(ell, params)

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
          Eitherwise, it is possible to pass directly a
          dictionary of precision parameters.
        - nonlinear (bool, default: False);
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc. P(k, z) is in units of (Mpc/h)^3.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct emu
        emu = self._emu['pk_{}'.format(name)]

        # Get output
        out = emu.get_from_class(k, z, params, precision=precision)

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
          Eitherwise, it is possible to pass directly a
          dictionary of precision parameters.
        - nonlinear (bool, default: False);
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - timeit (bool, default: False): print execution time.

        NOTE: k is in units of h/Mpc.
        """

        # TODO: implement nonlinear
        if nonlinear:
            raise ValueError('Nonlinear Pk not yet implemented')

        # Select correct emu
        emu = self._emu['fk_{}'.format(name)]

        # Get output
        out = emu.get_from_class(k, z, params, precision=precision)

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
          Eitherwise, it is possible to pass directly a
          dictionary of precision parameters;
        - squeeze (bool, default: False): squeeze dimensions of output array;
        - timeit (bool, default: False): print execution time.

        """

        # Select correct emu
        try:
            emu = self._emu['cl_{}_lensed'.format(name)]
        except KeyError:
            emu = self._emu['cl_{}'.format(name)]

        # Get output
        out = emu.get_from_class(ell, params, precision=precision)

        # Squeeze dimensions
        if squeeze and out.shape == (1,):
            return out[0]

        return out
