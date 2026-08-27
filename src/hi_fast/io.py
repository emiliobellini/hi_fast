"""Backward-compatible facade for HiFast input/output helpers.

Implementations live in focused private modules. Existing imports from
``hi_fast.io`` remain supported.
"""

from ._filesystem import (
    EmuFile,
    FitsFile,
    Folder,
    _load_validation_report,
)
from ._metadata import (
    _build_info_metadata,
    _print_info,
)
from ._terminal import (
    info,
    print_level,
    timeit,
    title,
    warning,
    write_blue,
    write_green,
    write_magenta,
    write_red,
)

__all__ = [
    'EmuFile',
    'FitsFile',
    'Folder',
    'info',
    'print_level',
    'timeit',
    'title',
    'warning',
    'write_blue',
    'write_green',
    'write_magenta',
    'write_red',
    '_build_info_metadata',
    '_load_validation_report',
    '_print_info',
]
