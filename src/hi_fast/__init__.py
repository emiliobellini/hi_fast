from .main import HiFast

# TODO:
# - integrate class so that computation is not repeated when asking for different spectra
# - integrate background quantities from class
# - make easy for a sampler to switch from emulator and class if outside the boundaries
# - FUTURE: try to learn while sampling if an emulator output can be trusted outside the emulator range
# - implement non linear
# - write readme for each emulator (name, parameters, ranges)
# - extra check dimensions of each spectrum (mainly Cl) with class
# - compare class and emu output to check dimensions emu

# Dimensions checked with class.
# Cl TT - OK
# Cl TE - OK
# Cl EE - OK
# Cl Tp
# Cl pp - OK
# Cl BB
# Pk m - OK
# Pk cb - OK
# Pk weyl
# f m
# f cb
# f weyl
