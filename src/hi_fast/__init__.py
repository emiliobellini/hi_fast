from . import _tensorflow_config  # noqa: F401
from .main import HiFast

__all__ = ['HiFast']

# TODO:
# - integrate background quantities from class
# - FUTURE: try to learn while sampling if an emulator output can
#   be trusted outside the emulator range
# - implement non linear
