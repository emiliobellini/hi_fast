"""Timing and terminal-formatting helpers."""

import time


def timeit(func):
    """Decorator logging execution time when ``timeit=True``.

    Args:
        func (Callable): Function to wrap.

    Returns:
        Callable: Wrapped callable preserving ``func``'s signature.
    """
    def wrapper_function(*args, **kwargs):
        """Call the wrapped function and optionally report elapsed time."""
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
