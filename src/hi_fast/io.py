"""
.. module:: io

:Synopsis: Input/output related functions and classes.
:Author: Emilio Bellini

"""

import joblib
import os
import re
import time
from pickle import UnpicklingError
from ._tensorflow_config import keras
from tabulate import tabulate
from astropy.io import fits
from collections import OrderedDict


# ------------------- Folder -------------------------------------------------#

class Folder(object):
    """Utility wrapper around filesystem directories.

    The helper exposes convenience methods to list contents, create
    subfolders, and validate directory existence with consistent logging.
    """

    def __repr__(self):
        return self.path

    def __str__(self):
        return str(self.path)

    def __init__(self, path, root=None, should_exist=False):
        """Build a ``Folder`` instance pointing to ``path``.

        Args:
            path (str): Relative or absolute path to the directory.
            root (str | None): Optional base directory joined with ``path``.
            should_exist (bool): When True, raise if the directory is absent.
        """
        if root is None:
            self.path = os.path.abspath(path)
        else:
            self.path = os.path.abspath(os.path.join(root, path))
        # Check existence
        self.exists = os.path.isdir(self.path)
        if should_exist:
            self._exists_or_error()

    def _exists_or_error(self):
        """Raise ``IOError`` if the directory does not exist."""
        if not os.path.isdir(self.path):
            raise IOError('Folder {} does not exist!'.format(self.path))
        return

    def create(self, verbose=False):
        """Create the directory if missing.

        Args:
            verbose (bool): When True, print a message after creating.

        Returns:
            Folder: ``self`` for easy chaining.
        """
        if not self.exists:
            os.makedirs(self.path)
            self.exists = os.path.isdir(self.path)
            if verbose:
                print_level(1, 'Created folder {}'.format(self.path))
        self._exists_or_error()
        return self

    def list_files(self, patterns=None, unique=False):
        """Return files inside the folder, optionally filtered.

        Args:
            patterns (str | list[str] | None): Regular-expression patterns;
                a file is kept if it matches any entry. ``None`` keeps all
                files.
            unique (bool): When True, raise if zero or multiple matches are
                found.

        Returns:
            list[str]: Absolute paths of files satisfying the patterns.
        """
        if not self.exists:
            filtered = []
        else:
            # List all files in path
            for root, _, files in os.walk(self.path):
                if root == self.path:
                    all = [os.path.join(root, x) for x in files]
            # Filter with pattern
            if patterns:
                if isinstance(patterns, str):
                    patterns = [patterns]
                filtered = []
                for pattern in patterns:
                    filtered += [x for x in all if re.match(pattern, x)]
            else:
                filtered = all
        # Check uniqueness
        if unique:
            if len(filtered) == 0:
                raise Exception('No files matching patterns')
            elif len(filtered) > 1:
                raise Exception('Multiple files matching patterns')
        return filtered

    def list_subfolders(self, patterns=None, unique=False):
        """Return child directories, optionally filtered.

        Args:
            patterns (str | list[str] | None): Regular-expression patterns
                applied to subfolder paths. ``None`` keeps all.
            unique (bool): When True, raise if zero or multiple matches are
                found.

        Returns:
            list[str]: Absolute paths of matching subfolders.
        """
        if not self.exists:
            filtered = []
        else:
            # List all files in path
            for root, dirs, _ in os.walk(self.path):
                if root == self.path:
                    all = [os.path.join(root, x) for x in dirs]
            # Filter with pattern
            if patterns:
                if isinstance(patterns, str):
                    patterns = [patterns]
                filtered = []
                for pattern in patterns:
                    filtered += [x for x in all if re.match(pattern, x)]
            else:
                filtered = all
        # Check uniqueness
        if unique:
            if len(filtered) == 0:
                raise Exception('No subfolders matching patterns')
            elif len(filtered) > 1:
                raise Exception('Multiple subfolders matching patterns')
        return filtered

    def is_empty(self):
        """Check whether the directory contains any files.

        Returns:
            bool: ``True`` when the directory is missing or empty.
        """
        if not self.exists:
            return True
        if self.list_files():
            return False
        else:
            return True

    def subfolder(self, subpath, should_exist=False):
        """Return a ``Folder`` rooted at ``self/subpath``.

        Args:
            subpath (str): Relative sub-directory.
            should_exist (bool): When True, ensure the subfolder exists.

        Returns:
            Folder: Helper for the nested directory.
        """
        path = os.path.join(self.path, subpath)
        return Folder(path=path, should_exist=should_exist)

    def join(self, subpath):
        """Join the folder path with ``subpath``.

        Args:
            subpath (str): Relative fragment appended to ``self.path``.

        Returns:
            str: Absolute path of the combined location.
        """
        path = os.path.join(self.path, subpath)
        return path


# ------------------- Fits Files ---------------------------------------------#

class FitsFile(object):
    """
    Class to manage fits files.

    NOTE: the only fine tuned part here is how we manage headers.
    We accept dictionary as headers, manipulating them to make
    them acceptable as headers by astropy.fits. In particular:
    1) we flatten the dictionary with _flatten_dict
      (_unflatten_dict is used to reverse it). In the same function
      we split keys and values of the flattened dict, making both
      of them values (astropy accepts keys with less than 8 chars);
    2) we convert (nested) lists to strings with _delistify (_listify
      is used to reverse it), because astropy does not accept lists.
      For nested arrays or lists we accept up to 2 dimensions.

    """

    def __init__(self, fname, root=None):
        # Define path
        if root is None:
            self.path = os.path.abspath(fname)
        else:
            self.path = os.path.abspath(os.path.join(root, fname))
        # Check existence
        self.exists = os.path.isfile(self.path)
        # Check is fits
        is_fits = self.path.endswith('.fits') or self.path.endswith('.fits.gz')
        if not is_fits:
            raise ValueError('Expected .fits file, found {}'.format(self.path))
        return

    def _flatten_dict(self, nested_dict, delimiter='__'):
        split_dict = {}
        flat_dict = self._flatten_dict_recursive(
            nested_dict, delimiter=delimiter)
        # In astropy, keys can not be longer than 8 characters. We then create
        # a flat dict where both keys and values are values. This dictionary
        # will have keys starting with two delimiters for the keys of the
        # previous step dictionary, and keys starting with one delimiter for
        # the values of the previous dictionary. To fix the correspondence each
        # key ends with a different integer,
        for nkey, (key, val) in enumerate(flat_dict.items()):
            split_dict['{}{}{}'.format(delimiter, delimiter, nkey)] = key
            split_dict['{}{}'.format(delimiter, nkey)] = val
        return split_dict

    def _flatten_dict_recursive(
            self, nested_dict, parent_key='', delimiter='__'):
        """Flatten a nested dictionary, preserving key order."""
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{delimiter}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict_recursive(
                    value, new_key, delimiter).items())
            else:
                items.append((new_key, value))
        return OrderedDict(items)

    def _unflatten_dict(self, flat_dict, delimiter='__'):
        current_dict = {}
        # We first fix the correspondence between keys and values to get a list
        # of flattened keys and values (si discussion in _flatten_dict above).
        for key_flat in flat_dict.keys():
            if key_flat.startswith('{}{}'.format(delimiter, delimiter)):
                key = flat_dict[key_flat]
                val = flat_dict[key_flat[len(delimiter):]]
                current_dict[key] = val
        """
        Reconstruct a nested dictionary from flattened keys,
        preserving order.
        """
        result = OrderedDict()
        for key, value in current_dict.items():
            parts = key.split(delimiter)
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = OrderedDict()
                d = d[part]
            d[parts[-1]] = value
        return result

    def _delistify(self, flat_dict, delimiter='_*_'):
        # fits header do not accept lists as values.
        # here we convert them into strings, assuming
        # ndim <= 2.
        delimiter2 = 2*delimiter
        for key, val1 in flat_dict.items():
            if isinstance(val1, list):
                if all([isinstance(x, str) for x in val1]):
                    val1 = delimiter2.join(val1)
                    flat_dict[key] = val1
                if all([isinstance(x, list) for x in val1]):
                    tmp = []
                    for val2 in val1:
                        tmp.append(delimiter.join([str(x) for x in val2]))
                    flat_dict[key] = delimiter2.join(tmp)
        return flat_dict

    def _listify(self, flat_dict, delimiter='_*_'):
        # fits header do not accept lists as values.
        # here we convert back delistified lists,
        # assuming ndim <= 2.
        current_dict = {}
        delimiter2 = 2*delimiter
        for key, val1 in flat_dict.items():
            if isinstance(val1, str) and delimiter2 in val1:
                current_dict[key] = val1.split(delimiter2)
            else:
                current_dict[key] = val1
        for key, val1 in current_dict.items():
            if isinstance(val1, str) and delimiter in val1:
                current_dict[key] = val1.split(delimiter)
            elif isinstance(val1, list):
                for nval2, val2 in enumerate(val1):
                    if isinstance(val2, str) and delimiter in val2:
                        current_dict[key][nval2] = val2.split(delimiter)
                    else:
                        current_dict[key][nval2] = val2
        for key, val1 in current_dict.items():
            if isinstance(val1, str):
                current_dict[key] = self._floatify(val1)
            elif isinstance(val1, list):
                for nval2, val2 in enumerate(val1):
                    if isinstance(val2, str):
                        current_dict[key][nval2] = self._floatify(val2)
                    elif isinstance(val2, list):
                        current_dict[key][nval2] = [self._floatify(x)
                                                    for x in val2]
                    else:
                        current_dict[key][nval2] = val2
        return current_dict

    def _floatify(self, val_string):
        try:
            val = float(val_string)
        except ValueError:
            val = val_string
        return val

    def write(self, name=None, data=None, header=None, verbose=False):
        """
        Write an HDU to a fits file. If the file does not
        exists, it creates with the content stored in the
        PrimaryHDU, otherwise it appends and ImageHDU to it.
        Arguments:
        - name (str, default: None): name of the HDU;
        - data (array, default: None): data to be stored;
        - header (dict or fits.Header): header to be stored.
          If the input is a dictionary, it is manually
          converted so it can be accepted by astropy.fits
          as header (see NOTE at the beginning of the class);
        - verbose (bool, default: False): verbosity.
        """
        # Create parent folder
        try:
            Folder(os.path.dirname(self.path)).create()
        except FileExistsError:
            pass
        # We assume that header is either a dictionary, or a fits.Header
        if isinstance(header, dict):
            header = self._flatten_dict(header)
            header = self._delistify(header)
            header = fits.Header(header)
        # Create first HDU
        if not self.exists:
            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
            hdul.writeto(self.path)
            self.exists = True
        else:
            # Open the file and append
            with fits.open(self.path, mode='append') as hdul:
                hdul.append(fits.ImageHDU(data, name=name, header=header))
        if verbose:
            print_level(1, 'Appended {} to {}'.format(
                name.upper(), os.path.relpath(self.path)))
        return

    def update(self, name, data=None, header=None):
        """
        Update an HDU of a fits file. The HDU should already
        exists (otherwise use the .write method).
        Arguments:
        - name (str): name of the HDU;
        - data (array, default: None): data to be updated;
        - header (dict or fits.Header): header to be updated.
          If the input is a dictionary, it is manually
          converted so it can be accepted by astropy.fits
          as header (see NOTE at the beginning of the class).
        - verbose (bool, default: False): verbosity.
        """
        if isinstance(header, dict):
            header = self._flatten_dict(header)
            header = self._delistify(header)
            header = fits.Header(header)
        with fits.open(self.path, mode='update') as hdul:
            if data is not None:
                hdul[name].data = data
            if header is not None:
                hdul[name].header = header
        return

    def print_info(self):
        """
        Print on screen fits file info.
        """
        with fits.open(self.path) as hdul:
            print(hdul.info())
        return

    def get_header(self, name, unflat_dict=True):
        """
        Open a fits file and return the header of an HDU.
        Arguments:
        - name (str): name of the HDU;
        - unflat_dict (bool, default: True): if False return just the
          header, otherwise try to reconstruct the nested dictionary.
        Return:
        - header.
        """
        with fits.open(self.path) as fn:
            if unflat_dict:
                hd = self._listify(fn[name].header)
                hd = self._unflatten_dict(hd)
            else:
                hd = fn[name].header
        return hd

    def get_data(self, name):
        """
        Open a fits file and return the data of an HDU.
        Arguments:
        - name (str): name of the HDU.
        Return:
        - data array.
        """
        with fits.open(self.path) as fn:
            return fn[name].data

    def get_keys(self):
        """
        Return the list of HDU names stored in the fits file.
        Return:
        - list of HDU names.
        """
        with fits.open(self.path) as fn:
            names = [hdu.name for hdu in fn]
            return names


# ------------------- EmuFile ------------------------------------------------#

class EmuFile(object):
    """Helper for locating, loading, and validating emulator files."""

    def __repr__(self):
        return self.path

    def __str__(self):
        return str(self.path)

    def __init__(self, fname, root=None, should_exist=False):
        """Create an ``EmuFile`` pointing to ``fname``.

        Args:
            fname (str): Relative or absolute filename of the emulator
                metadata bundle.
            root (str | Folder | None): Optional directory prepended to
                ``fname``. Passing a ``Folder`` reuses its path attribute.
            should_exist (bool): When True, raise if the file is missing.
        """
        # Define path of the emulator file
        if root is None:
            self.path = fname
        elif isinstance(root, str):
            self.path = os.path.join(root, fname)
        elif isinstance(root, Folder):
            self.path = os.path.join(root.path, fname)
        else:
            raise ValueError(
                'Argument root not recognized. '
                'It can be a string or a Folder object!')
        self.path = os.path.abspath(self.path)

        # Check existence
        self.exists = os.path.isfile(self.path)
        if should_exist:
            self._exists_or_error()

    def _exists_or_error(self):
        """Raise ``IOError`` if the emulator file is absent."""
        if not os.path.isdir(self.path):
            raise IOError('Folder {} does not exist!'.format(self.path))
        return

    def _get_path(self, fname, root):
        """Resolve ``fname`` and ``root`` into an absolute path.

        Args:
            fname (str | EmuFile | None): Optional override for the target
                file. ``None`` uses ``self.path``.
            root (str | Folder | None): Optional base directory applied to
                the resolved ``fname``.

        Returns:
            str: Absolute filesystem path.
        """
        # Deal with fname
        if fname is None:
            path = self.path
        elif isinstance(fname, str):
            path = fname
        elif isinstance(fname, EmuFile):
            path = fname.path
        else:
            raise ValueError(
                'Argument fname not recognized. '
                'It can be a string or a EmuFile object!')
        # Add root
        if root is None:
            pass
        elif isinstance(root, str):
            path = os.path.join(root, path)
        elif isinstance(fname, Folder):
            path = os.path.join(root.path, path)
        else:
            raise ValueError(
                'Argument root not recognized. '
                'It can be a string or a Folder object!')
        return path

    def _is_dict_file(self):
        """Return ``True`` if the serialized payload is a dictionary."""
        try:
            content = joblib.load(self.path)
        except UnpicklingError:
            return False
        if isinstance(content, dict):
            return True
        else:
            return False

    def load(self, fname=None, root=None, verbose=False):
        """Load emulator metadata (and Keras model, if present).

        Args:
            fname (str | EmuFile | None): Optional file override.
            root (str | Folder | None): Optional base directory combined
                with ``fname``.
            verbose (bool): When True, log where the emulator was loaded
                from.

        Returns:
            dict: Emulator description pulled from disk.
        """
        # Get path
        path = self._get_path(fname, root)

        self.content = joblib.load(path)

        # Load keras model if any
        if 'model_path' in self.content.keys():
            model_path = os.path.join(
                os.path.dirname(path), self.content['model_path'])
            self.content['model'] = keras.models.load_model(
                model_path, compile=False)

        # Remove model_path from content
        self.content.pop('model_path', None)

        if verbose:
            info('Loaded emulator info from {}'.format(path))
        return self.content


# ------------------- Info ---------------------------------------------------#

def _print_info(spectra, params, name=None):
    """Print summary info for each spectrum emulator.
    Args:
        spectra (dict): Mapping from spectrum names to Spectrum objects.
        params (dict): Mapping from spectrum names to Params objects.
        name (str | None): When provided, print info only for the named
            spectrum.
    """
    if name is not None:
        spectra = {name: spectra[name]}
        params = {name: params[name]}
    info('HiFast emulator info:')

    for spec_name in spectra.keys():
        print('\n')
        print_level(1, 'Spectrum: {}'.format(spec_name))
        spec = spectra[spec_name]
        param = params[spec_name]

        headers = ['Parameter', 'Min', 'Max', 'Can be derived from']
        headers = [write_green(x) for x in headers]
        tab = []
        for p_name in param._required:
            if p_name in param._ranges:
                p_min, p_max = param._ranges[p_name]
            else:
                p_min, p_max = 'N/A', 'N/A'
            der = ', '.join([x for x in param._derived[p_name] if x != p_name])
            tab.append([write_blue(p_name), p_min, p_max, der])

        # Print k, z, ell ranges
        if spec.k_min is not None and spec.k_max is not None:
            k_min, k_max = spec.k_min, spec.k_max
            tab.append([write_magenta('k [h/Mpc]'), k_min, k_max, 'N/A'])
        if spec.z_min is not None and spec.z_max is not None:
            z_min, z_max = spec.z_min, spec.z_max
            tab.append([write_magenta('z'), z_min, z_max, 'N/A'])
        if spec.ell_min is not None and spec.ell_max is not None:
            ell_min, ell_max = spec.ell_min, spec.ell_max
            tab.append([write_magenta('ell'), ell_min, ell_max, 'N/A'])

        print(tabulate(tab, headers=headers, tablefmt='grid'))

    return


# ------------------- Scripts ------------------------------------------------#

def timeit(func):
    """Decorator logging execution time when ``timeit=True``.

    Args:
        func (Callable): Function to wrap.

    Returns:
        Callable: Wrapped callable preserving ``func``'s signature.
    """
    def wrapper_function(*args, **kwargs):
        try:
            dotimeit = kwargs['timeit']
        except KeyError:
            dotimeit = False
        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = True
        if verbose and dotimeit:
            start = time.time()
        result = func(*args,  **kwargs)
        if verbose and dotimeit:
            print_level(1, '{} executed in {:.2e} seconds'.format(
                func, time.time()-start))
        return result
    return wrapper_function


def write_red(msg):
    """Return ``msg`` wrapped in ANSI escape codes for bold red text."""
    return '\033[1;31m{}\033[00m'.format(msg)


def write_green(msg):
    """Return ``msg`` wrapped in ANSI escape codes for bold green text."""
    return '\033[1;32m{}\033[00m'.format(msg)


def write_blue(msg):
    """Return ``msg`` wrapped in ANSI escape codes for bold blue text."""
    return '\033[1;34m{}\033[00m'.format(msg)


def write_magenta(msg):
    """Return ``msg`` wrapped in ANSI escape codes for bold magenta text."""
    return '\033[1;35m{}\033[00m'.format(msg)


def warning(msg):
    """Print ``msg`` prefixed with a red ``[WARNING]`` tag."""
    prepend = write_red('[WARNING]')
    print('{} {}'.format(prepend, msg), flush=True)
    return


def info(msg):
    """Print ``msg`` prefixed with a green ``[info]`` tag."""
    prepend = write_green('[info]')
    print('{} {}'.format(prepend, msg), flush=True)
    return


def title(msg, width=72):
    """Print ``msg`` as a centered green section title.

    Args:
        msg (str): Title text.
        width (int): Minimum width of the surrounding border.
    """
    width = max(width, len(msg))
    border = '=' * width
    print('\n{}\n{}\n{}'.format(
        write_green(border),
        write_green(msg.center(width)),
        write_green(border)), flush=True)
    return


def print_level(num, msg, arrow=True):
    """Pretty-print messages with indentation levels.

    Args:
        num (int): Indentation level. Each level adds four dashes.
        msg (str): Message to print.
        arrow (bool): When True, prepend an arrow marker, otherwise indent
            with spaces only.
    """
    if num > 0:
        if arrow:
            prepend = write_green(num*'----' + '> ')
        else:
            prepend = (4*num+2)*' '
    else:
        prepend = ''
    print('{}{}'.format(prepend, msg), flush=True)
    return
