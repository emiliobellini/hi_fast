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
- Select `thin`, `std`, or `ext` emulator trust regions and automatically
  route out-of-region samples to HiCLASS.

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
ells = [2, 10, 50]

pk = hifast.get_pk(k, z, params, name="m", squeeze=True)
fk = hifast.get_fk(k, z, params, name="m", squeeze=True)
cl_tt = hifast.get_cell(ells, params, name="TT", squeeze=True)

print(pk.shape, fk.shape, cl_tt.shape)
# (2, 2) (2, 2) (3,)
```

Use `print_info()` to inspect the observables and trust regions stored in a
bundle. With no arguments it prints a compact grouped summary; with one
observable name it prints the detailed parameter table:

```python
hifast.print_info()                       # every loaded observable
hifast.print_info("pk_m")                 # one observable
hifast.print_info("pk_m", bounds="std")   # one trust region
```

The same metadata can be rendered as Markdown, which is useful for generating
bundle-level emulator documentation:

```python
hifast.print_info(markdown=True)
hifast.print_info(markdown=True, output="emu/lcdm/README.md")
```

Each emulator stores three nested parameter regions: `thin`, `std`, and
`ext`. The `thin` region is the narrowest, `std` is the standard validation
region, and `ext` is the widest stored region. These ranges define where an
emulator is meant to be trusted; they are distinct from the `k` and `ell`
support of a spectrum.

The emulator-facing `get_pk`, `get_fk`, and `get_cell` methods accept a
`trusted_region` boundary policy. Its default,
`"ext"`, preserves the widest emulator domain. Choose `"thin"` or `"std"`
for a more conservative domain, or pass `None` to bypass the emulator and use
HiCLASS for the complete request:

```python
pk_conservative = hifast.get_pk(
    k, z, params, trusted_region="std"
)
pk_class = hifast.get_pk(
    k, z, params, trusted_region=None
)
```

By default, a request outside the selected region raises an informative
exception. Set `on_out_of_bounds="class"` to use HiCLASS only for the
out-of-region entries:

```python
pk = hifast.get_pk(
    k,
    z,
    parameter_rows,
    trusted_region="std",
    on_out_of_bounds="class",
    class_precision=1,
)
```

This policy covers cosmological parameters, redshift, and the emulator's
fixed `k` or `ell` support. In mixed batches, trusted entries remain batched
through the emulator while out-of-range redshifts are grouped into one
HiCLASS calculation per affected cosmology. The requested output order and
shape do not change. `class_precision` has the same meaning as the `precision`
argument of the explicit `get_*_from_class` methods.

When a bundle contains `validation.json`, `print_info()` and its generated
README also show held-out test accuracy. The original global train/test split
is reconstructed from the training fraction, random seed, finite-row filter,
and dataset ordering stored in each emulator joblib. Results are cumulative
by source region: `thin` contains its held-out rows, `std` combines the thin
and standard held-out rows, and `ext` combines all three source datasets.
Only held-out rows are reported; training rows are excluded.

For CMB spectra, the public `name` argument is the short selector (`TT`, `TE`,
`EE`, `BB`, `Tp`, or `pp`). Complete names such as `cl_TT_lensed` identify
emulator bundles and FITS entries; they are also the names accepted by
`get_params_names`. HiFast prefers a lensed emulator and falls back to the
corresponding raw emulator when only the latter is available.

The emulator methods accept `check_params_names` and `check_params_values`
flags and expose a `timeit` switch for profiling. Setting
`check_params_values=False` disables trusted-region checks for cosmological
parameter values; redshift and `k`/`ell` support are still enforced by the
boundary policy. Parameter-name checking remains active in all-HiCLASS mode
unless explicitly disabled.

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
HiFast privately retains the latest compatible HiCLASS computation and shares
it across these methods, background quantities, and automatic out-of-bounds
fallbacks. Requests reuse
the calculation when the cosmological and precision parameters are unchanged
and its computed outputs, wavenumber/redshift ranges, and multipole range
cover the new observable. A request requiring wider coverage upgrades the
calculation; changing cosmology or precision replaces it. This is automatic
and does not change the public API.

When several observables are known in advance, `get_from_class` combines their
requirements and computes all of them with one HiCLASS run:

```python
results = hifast.get_from_class(
    params,
    observables={
        "cell": {
            "TT": {"ell": [2, 10, 50]},
            "EE": {"ell": [2, 10, 50]},
        },
        "pk": {
            "m": {"k": [0.01, 0.1], "z": [0.0, 1.0]},
        },
    },
    precision=0,
)

cl_tt = results["cell"]["TT"]
cl_ee = results["cell"]["EE"]
pk_m = results["pk"]["m"]
```

The `pk`, `fk`, and `cell` groups accept the same short spectrum names as the
individual methods. The nested result dictionary mirrors the request. Set
`squeeze=True` to apply the usual singleton-dimension removal to each result.

Background quantities require no emulator and use a fast background-only
HiCLASS calculation:

```python
background = hifast.get_background(
    params,
    z=[0.0, 0.5, 1.0],
    quantities=[
        "H",
        "comoving_distance",
        "angular_diameter_distance",
        "growth_factor",
        "growth_rate",
        "age",
        "Omega_m",
    ],
)
```

Supported redshift-dependent quantities are `H`, `comoving_distance`,
`angular_diameter_distance`, `luminosity_distance`, `growth_factor`, and
`growth_rate`. Supported scalar quantities are `age`, `Omega_m`, `Omega_b`,
`Omega_cdm`, `Omega_k`, `Omega_r`, `Omega_g`, `Omega_nu`, and `Omega_Lambda`.
Distances are returned in Mpc, `H` in km/s/Mpc, and the age in Gyr. Omitting
`quantities` returns all supported values. A compatible existing spectral
HiCLASS calculation is reused; otherwise the background-only calculation
typically takes only a few hundredths of a second.

For access to every native column on CLASS's internal sampling, use:

```python
table = hifast.get_background_table(params)
z_class = table["z"]
h_over_c = table["H [1/Mpc]"]
```

The table's column names and units come directly from the installed HiCLASS
version. `print_info()` lists the stable HiFast background nomenclature next
to the corresponding HiCLASS method or property; generated emulator READMEs
include the same table. The materialized native table is cached with the
HiCLASS instance, because exporting all internal columns is more expensive
than copying the resulting arrays. It is invalidated automatically whenever
the underlying CLASS computation is replaced or upgraded.

For developers, `HiClassService.BACKGROUND_QUANTITIES` is the authoritative
background registry. Each entry records the HiCLASS method or property,
whether it requires `z`, and its units. A quantity requiring additional
arguments or CLASS modules also needs dedicated extraction and computation
requirements rather than only a registry entry.

Only linear power spectra and growth rates are currently supported;
`nonlinear=True` raises an exception.

HiFast suppresses TensorFlow's C++ startup diagnostics by default, including
harmless messages about unavailable CUDA drivers on CPU-only systems. This
suppression is limited to TensorFlow import; subsequent errors and warnings
remain visible. If the import itself fails, its captured diagnostics are
printed before the exception is propagated.

## Development Notes

- Core source lives under `src/hi_fast/`.
- Filesystem/FITS helpers, metadata rendering, and terminal formatting live in
  separate private modules; `hi_fast.io` remains a compatibility facade.
- Emulator metadata and weights reside in `emu/`.
- Scientific validation scripts are located directly in `tests/`.
- The held-out accuracy and histogram workflow is
  `scripts/validate_emulators.py`.
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
`tests/` are scientific validation scripts. Performance scripts live under
`benchmarks/`; run them explicitly when their FITS data and emulator assets
are available.

Generate a memory-bounded held-out validation report with:

```bash
python scripts/validate_emulators.py \
    ../../Data/emu_like/lcdm/sample \
    --model lcdm \
    --batch-size 512
```

The script first reconstructs the global test indices using only per-file
finite-row counts. It then loads thin, std, and ext sequentially, retaining
only the small RMS arrays needed for cumulative statistics. By default it
writes `emu/<model>/validation.json` and plots below
`validation_plots/<model>/`. Pass `--no-plots` for statistics only. Once the
report exists, regenerate bundle READMEs with
`scripts/generate_emulator_readmes.py`.

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
