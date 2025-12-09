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
from tensorflow import keras
from tabulate import tabulate


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
            print_level(1, '{} executed in {} seconds'.format(
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
