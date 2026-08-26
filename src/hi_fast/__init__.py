from . import _tensorflow_config  # noqa: F401
from .main import HiFast

__all__ = ['HiFast']

# TODO:
# - integrate class so that computation is not repeated
#   when asking for different spectra
# - integrate background quantities from class
# - make easy for a sampler to switch from emulator and class
#   if outside the boundaries
# - FUTURE: try to learn while sampling if an emulator output can
#   be trusted outside the emulator range
# - implement non linear
