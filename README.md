# hi_fast
hi_class emulator
# hi_fast

Fast emulation layer on top of [CLASS](https://lesgourg.github.io/class_public/class.html)
to evaluate cosmological power spectra, growth rates, and CMB angular
spectra without rerunning the Boltzmann solver. The project bundles the
trained neural emulators, preprocessing scalers, and PCA projections so you
can query spectra with just a handful of cosmological parameters.

## Features

- Predict linear matter spectra `P(k, z)` for total matter, CDM+baryons, and
	Weyl potentials.
- Provide growth rates `f(k, z) = d ln P / d ln a` via dedicated emulators.
- Deliver dimensionless CMB angular spectra `ℓ(ℓ+1)C_ℓ/2π` for T/E/B/p.
- Convert between commonly used cosmological parameters (e.g. `H0 ↔ h`,
	`σ8`, `S8`) and validate that requests stay inside training ranges.
- Fallback to CLASS on demand with preconfigured precision settings for
	validation or higher accuracy runs.

## Installation

hi_fast requires Python 3.8+ and depends on NumPy, SciPy, scikit-learn,
TensorFlow 2.15+, joblib, and CLASSy. To install in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

If you already have CLASS built locally, ensure the `classy` Python module is
discoverable (e.g., via `PYTHONPATH`).

## Emulator Assets

The neural-network weights and preprocessing metadata live under `emu/`.
Each subfolder corresponds to a cosmological model (e.g., `lcdm`) and ships
with `.joblib` files plus the matching `.keras` networks. When you package or
relocate the project, keep the `emu/` hierarchy intact so the loader can find
the assets:

```
emu/
	lcdm/
		cl_TT_lensed.joblib
		cl_TT_lensed.keras
		...
```

## Quickstart

```python
import numpy as np

from hi_fast import main

# Load all spectra for the LCDM emulator bundle located under ./emu
hifast = main.HiFast(name="lcdm", root="emu", timeit=True)

params = {
		"Omega_m": 0.31,
		"Omega_b": 0.049,
		"h": 0.68,
		"n_s": 0.965,
		"ln_A_s_1e10": 3.044,
}

k = [0.01, 0.1]  # h/Mpc
z = [0.0, 1.0]

pk = hifast.get_pk(k, z, params, name="m", squeeze=True)
fk = hifast.get_fk(k, z, params, name="m", squeeze=True)
cl_tt = hifast.get_cell([2, 10, 50], params, name="TT")

print(pk.shape, fk.shape, cl_tt)
```

For CMB spectra, the public `name` argument is the short selector (`TT`, `TE`,
`EE`, `BB`, `Tp`, or `pp`). Complete names such as `cl_TT_lensed` identify
emulator bundles and FITS entries; they are also the names accepted by
`get_params_names`. HiFast prefers a lensed emulator and falls back to the
corresponding raw emulator when only the latter is available.

All `get_*` methods accept `check_params_names` / `check_params_values`
flags to enforce input validation and expose a `timeit` switch for profiling.

The emulator methods are batch-first. Parameter input can be one dictionary,
a sequence of dictionaries, or a NumPy array. A one-dimensional array is one
cosmology; a two-dimensional array contains one cosmology per row. Array
columns must follow the observable emulator's `input_params_names` order,
excluding `z_pk`, because redshift is passed separately. Dictionary input can
still use supported derived parameter names such as `H0`, `sigma8`, or `S8`.
For power spectra and growth rates, every cosmology is evaluated at every
redshift and the unsqueezed output has shape
`(n_cosmologies, n_redshifts, n_k)`. CMB spectra have shape
`(n_cosmologies, n_ell)`.

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

HiFast suppresses TensorFlow's C++ startup diagnostics by default, including
harmless messages about unavailable CUDA drivers on CPU-only systems. This
suppression is limited to TensorFlow import; subsequent errors and warnings
remain visible. If the import itself fails, its captured diagnostics are
printed before the exception is propagated.

## Development Notes

- Core source lives under `src/hi_fast/`.
- Emulator metadata and weights reside in `emu/`.
- Examples and inspection scripts are located in `examples/`.
- Performance scripts are located in `benchmarks/`.

Before submitting changes, run the conversion tests and ensure docstrings stay
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

Distributed under the terms of the MIT license. See `LICENSE` for details.
