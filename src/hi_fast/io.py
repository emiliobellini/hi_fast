"""
.. module:: io

:Synopsis: Input/output related functions and classes.
:Author: Emilio Bellini

"""

import joblib
import os
import re
import time


# ------------------- Folder -------------------------------------------------#

class Folder(object):
    """
    Generic class for folders.
    Here we implemented some ad-hoc method
    to ease some common task with folders.

    Arguments:
        - path (str): path to the folder
        - should_exist (bool, optional): check that the folder exists
    """

    def __repr__(self):
        return self.path

    def __str__(self):
        return str(self.path)

    def __init__(self, path, root=None, should_exist=False):
        if root is None:
            self.path = os.path.abspath(path)
        else:
            self.path = os.path.abspath(os.path.join(root, path))
        # Check existence
        self.exists = os.path.isdir(self.path)
        if should_exist:
            self._exists_or_error()

    def _exists_or_error(self):
        """
        Check if a folder exists and it is a proper
        directory, otherwise raise an error.
        """
        if not os.path.isdir(self.path):
            raise IOError('Folder {} does not exist!'.format(self.path))
        return

    def create(self, verbose=False):
        """
        Check if a folder exists, otherwise create it.

        Returns:
            - self: the same object
        """
        if not self.exists:
            os.makedirs(self.path)
            self.exists = os.path.isdir(self.path)
            if verbose:
                print_level(1, 'Created folder {}'.format(self.path))
        self._exists_or_error()
        return self

    def list_files(self, patterns=None, unique=False):
        """
        List all files matching any of the patterns (if specified).

        Arguments:
            - patterns (str or list of str, optional): regex patterns
                (file included if any of them is satisfied)
            - unique (bool, optional): check if there is more than
                one file satisfying the patterns

        Return:
            - list of files satisfying the pattern
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
        """
        List all subfolders matching any of the patterns (if specified).

        Arguments:
            - patterns (str or list of str, optional): regex patterns
                (subfolder included if any of them is satisfied)
            - unique (bool, optional): check if there is more than
                one subfolder satisfying the patterns

        Return:
            - list of files satisfying the pattern
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
        """
        Check if a folder is empty or not.

        Return:
            - True if folder is empty (or it does not exist), False otherwise
        """
        if not self.exists:
            return True
        if self.list_files():
            return False
        else:
            return True

    def subfolder(self, subpath, should_exist=False):
        """
        Define subfolder.

        Returns:
            - Folder class for the resulting path
        """
        path = os.path.join(self.path, subpath)
        return Folder(path=path, should_exist=should_exist)

    def join(self, subpath):
        """
        Join folder with subpath.

        Returns:
            - String with location of the resulting path
        """
        path = os.path.join(self.path, subpath)
        return path


# ------------------- Folder -------------------------------------------------#

class EmuFile(object):
    """
    Save and load emulator files.

    Arguments:
        - fname (str): file name of the emulator;
        - root (str or Folder class): root for the emualtor;
        - should_exist (bool, optional): check that the file exists
    """

    def __repr__(self):
        return self.path

    def __str__(self):
        return str(self.path)

    def __init__(self, fname, root=None, should_exist=False):
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
        """
        Check if the file exists, otherwise raise an error.
        """
        if not os.path.isdir(self.path):
            raise IOError('Folder {} does not exist!'.format(self.path))
        return

    def _get_path(self, fname, root):
        """
        Merge together in a unique string fname and root. Arguments:
        - fname (str or EmuFile object or None);
        - root (str or Folder object or None).
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

    def save(self, content, fname=None, root=None, verbose=False):
        """
        Save content to file.
        """
        # Get path
        path = self._get_path(fname, root)

        joblib.dump(content, path)

        if verbose:
            info('Saved emulator info at {}'.format(path))
        return

    def load(self, fname=None, root=None, verbose=False):
        """
        Load content from file.
        """
        # Get path
        path = self._get_path(fname, root)

        self.content = joblib.load(path)

        if verbose:
            info('Loaded emulator info from {}'.format(path))
        return self.content


# ------------------- Scripts ------------------------------------------------#

def timeit(func):
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
    return '\033[1;31m{}\033[00m'.format(msg)


def write_green(msg):
    return '\033[1;32m{}\033[00m'.format(msg)


def warning(msg):
    prepend = write_red('[WARNING]')
    print('{} {}'.format(prepend, msg), flush=True)
    return


def info(msg):
    prepend = write_green('[info]')
    print('{} {}'.format(prepend, msg), flush=True)
    return


def print_level(num, msg, arrow=True):
    if num > 0:
        if arrow:
            prepend = write_green(num*'----' + '> ')
        else:
            prepend = (4*num+2)*' '
    else:
        prepend = ''
    print('{}{}'.format(prepend, msg), flush=True)
    return
