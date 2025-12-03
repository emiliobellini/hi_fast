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

All `get_*` methods accept `check_params_names` / `check_params_values`
flags to enforce input validation and expose a `timeit` switch for profiling.

## Development Notes

- Core source lives under `src/hi_fast/`.
- Emulator metadata and weights reside in `emu/`.
- Examples and inspection notebooks are located in `examples/` and `tests/`.

Before submitting changes, run the conversion tests and ensure docstrings stay
in sync with new behavior.

## License

Distributed under the terms of the MIT license. See `LICENSE` for details.
