# HiFast

HiFast is a fast emulation interface for cosmological power spectra, growth
rates, and CMB angular spectra generated with HiCLASS. The project bundles
trained neural networks, preprocessing scalers, and PCA projections so these
observables can be evaluated without rerunning the Boltzmann solver.

## Features

- Predict linear matter spectra `P(k, z)` for total matter, CDM+baryons, and
  Weyl potentials.
- Provide growth rates `f(k, z) = d ln P / d ln a` via dedicated emulators.
- Deliver dimensionless CMB angular spectra `ℓ(ℓ+1)C_ℓ/2π` for T/E/B/p.
- Convert between commonly used cosmological parameters (e.g. `H0 ↔ h`,
  `sigma8_m`, `sigma8_cb`, `S8_m`, and `S8_cb`) and validate that requests
  stay inside training ranges.
- Call HiCLASS explicitly through matching `get_*_from_class` methods for
  validation or higher-accuracy calculations.

## Installation

HiFast supports Python 3.10–3.13. Its dependencies, including TensorFlow
2.18+, SciPy, scikit-learn, Astropy, joblib, and `hiclassy`, are declared in
`pyproject.toml`. To install the package and its runtime dependencies in
editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Emulator Assets

The neural-network weights and preprocessing metadata live under `emu/`.
Each subfolder corresponds to a cosmological model and ships with `.joblib`
metadata plus the matching `.keras` networks. The available bundles are
`lcdm`, `lcdm_k`, `lcdm_nu`, and `lcdm_nu_k`. Keep the `emu/` hierarchy intact
when relocating the project, and pass its location through the `root`
argument:

```
emu/
	lcdm/
		cl_TT_lensed.joblib
		cl_TT_lensed.keras
		...
```

## Quickstart

```python
from hi_fast import HiFast

# Load all spectra for the LCDM emulator bundle located under ./emu
hifast = HiFast(name="lcdm", root="emu", timeit=True)

params = {
    "Omega_m": 0.31,
    "Omega_b": 0.049,
    "h": 0.68,
    "n_s": 0.965,
    "ln_A_s_1e10": 3.044,
    "tau_reio": 0.054,
}

k = [0.01, 0.1]  # h/Mpc
z = [0.0, 1.0]

pk = hifast.get_pk(k, z, params, name="m", squeeze=True)
fk = hifast.get_fk(k, z, params, name="m", squeeze=True)
cl_tt = hifast.get_cell([2, 10, 50], params, name="TT")

print(pk.shape, fk.shape, cl_tt.shape)
# (2, 2) (2, 2) (1, 3)
```

Use `print_info()` to inspect the observables, required parameters, derived
parameter alternatives, and training ranges stored in a bundle:

```python
hifast.print_info()             # every loaded observable
hifast.print_info("pk_m")       # one observable
```

For CMB spectra, the public `name` argument is the short selector (`TT`, `TE`,
`EE`, `BB`, `Tp`, or `pp`). Complete names such as `cl_TT_lensed` identify
emulator bundles and FITS entries; they are also the names accepted by
`get_params_names`. HiFast prefers a lensed emulator and falls back to the
corresponding raw emulator when only the latter is available.

The emulator methods accept `check_params_names` and `check_params_values`
flags to enforce input validation and expose a `timeit` switch for profiling.
Inputs outside the emulator ranges raise an exception; HiFast does not switch
to HiCLASS automatically.

The emulator methods are batch-first. Parameter input can be one dictionary,
a sequence of dictionaries, or a NumPy array. A one-dimensional array is one
cosmology; a two-dimensional array contains one cosmology per row. Array
columns must follow the observable emulator's `input_params_names` order,
excluding `z_pk`, because redshift is passed separately. Dictionary input can
also use supported alternatives such as `H0`, `A_s`, `sigma8_m`, `sigma8_cb`,
`S8_m`, or `S8_cb`, when applicable.
For power spectra and growth rates, every cosmology is evaluated at every
redshift and the unsqueezed output has shape
`(n_cosmologies, n_redshifts, n_k)`. CMB spectra have shape
`(n_cosmologies, n_ell)`.

With `squeeze=True`, every dimension of length one is removed. Keep the
default `squeeze=False` when stable array ranks are important.

For paired datasets, pass `paired=True` to `get_pk` or `get_fk`. This evaluates
each cosmology only at the redshift with the same index, requires
`len(params) == len(z)`, and returns `(n_pairs, n_k)` before optional
squeezing:

```python
pk_pairs = hifast.get_pk(
    k, row_redshifts, parameter_rows, name="m", paired=True
)
```

Use `get_params_names` instead of accessing emulator internals to discover the
required array-column order:

```python
import numpy as np

names = hifast.get_params_names("pk_m")
parameter_rows = np.array([
    [first_cosmology[name] for name in names],
    [second_cosmology[name] for name in names],
])
```

Large inputs can be evaluated in bounded chunks without manually splitting
the arrays:

```python
pk = hifast.get_pk(
    k,
    redshifts,
    parameter_rows,
    name="m",
    batch_size=512,
)
```

For `get_pk` and `get_fk`, `batch_size` limits the number of flattened
cosmology-redshift pairs per model call in both Cartesian and paired modes.
For `get_cell`, it limits the number of cosmologies. Passing `None` evaluates
the complete input in one model call.

Growth rates can either use their dedicated emulator or be derived from the
power-spectrum emulator:

```python
fk_direct = hifast.get_fk(k, z, params, name="m")
fk_from_pk = hifast.get_fk(k, z, params, name="m", get_from_pk=True)
```

## Direct HiCLASS calculations

`get_pk_from_class`, `get_fk_from_class`, and `get_cell_from_class` provide
explicit HiCLASS calculations for one cosmology. Their `precision` argument
accepts presets `0`, `1`, and `2`, or a dictionary of precision overrides.
These calls are independent: HiFast currently does not cache and reuse one
HiCLASS computation across different observables.

Only linear power spectra and growth rates are currently supported;
`nonlinear=True` raises an exception.

HiFast suppresses TensorFlow's C++ startup diagnostics by default, including
harmless messages about unavailable CUDA drivers on CPU-only systems. This
suppression is limited to TensorFlow import; subsequent errors and warnings
remain visible. If the import itself fails, its captured diagnostics are
printed before the exception is propagated.

## Development Notes

- Core source lives under `src/hi_fast/`.
- Emulator metadata and weights reside in `emu/`.
- Scientific validation and plotting scripts are located directly in `tests/`.
- Performance scripts are located in `benchmarks/`.

Before submitting changes, run the automated tests and ensure docstrings stay
in sync with new behavior.

### Automated tests

Install the package with its test dependencies once per environment:

```bash
python -m pip install -e ".[test]"
```

Run the fast automated unit suite with:

```bash
python -m pytest
```

The pytest configuration collects only `tests/unit`. The files directly under
`tests/` are scientific validation and plotting scripts. Performance scripts
live under `benchmarks/`; run them explicitly when their FITS data and emulator
assets are available.

Useful commands include:

```bash
# More detail, including individual test names
python -m pytest -v

# Stop after the first failure
python -m pytest -x

# Run one test file
python -m pytest tests/unit/test_batch_api.py

# Include a source-coverage report
python -m pytest --cov=hi_fast --cov-report=term-missing
```

GitHub Actions runs the unit suite and coverage command automatically for each
push and pull request.

## License

Distributed under the terms of the GNU General Public License v3.0. See
`LICENSE` for details.
