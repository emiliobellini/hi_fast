"""Unit tests for the private shared HiCLASS computation cache."""

import numpy as np

import hi_fast._class_cache as cache_module
from hi_fast._class_cache import HiClassCache
from hi_fast._class_service import HiClassService
from hi_fast.main import HiFast
import hi_fast.spectra as spectra_module


class FakeHiClass:
    """Small HiClass replacement that records native lifecycle calls."""

    instances = []

    def __init__(self):
        self.params = None
        self.compute_calls = 0
        self.cleanup_calls = 0
        self.empty_calls = 0
        self.background_calls = 0
        self.__class__.instances.append(self)

    def set(self, params):
        self.params = params.copy()

    def compute(self):
        self.compute_calls += 1

    def struct_cleanup(self):
        self.cleanup_calls += 1

    def empty(self):
        self.empty_calls += 1

    def h(self):
        return 0.7

    def Hubble(self, z):
        return np.ones_like(np.asarray(z), dtype=float) / 3000.0

    def comoving_distance(self, z):
        return 1000.0 * np.asarray(z)

    def angular_distance(self, z):
        z = np.asarray(z)
        return 1000.0 * z / (1.0 + z)

    def luminosity_distance(self, z):
        z = np.asarray(z)
        return 1000.0 * z * (1.0 + z)

    def scale_independent_growth_factor(self, z):
        return 1.0 / (1.0 + np.asarray(z))

    def scale_independent_growth_factor_f(self, z):
        return 0.5 + 0.1 * np.asarray(z)

    age = 13.8
    Omega_m = 0.3
    Omega_b = 0.05
    Omega_cdm = 0.25
    Omega_k = 0.0
    Omega_r = 1e-4
    Omega_g = 5e-5
    Omega_nu = 5e-5
    Omega_Lambda = 0.7

    def get_pk_lin(self, k, z, n_k, n_z, n_mu):
        return np.ones((n_k, n_z, n_mu))

    def lensed_cl(self, lmax):
        values = np.ones(lmax + 1)
        return {'tt': values, 'ee': values, 'te': values,
                'bb': values, 'tp': values, 'pp': values}

    def get_background(self):
        self.background_calls += 1
        return {
            'z': np.array([1.0, 0.0]),
            'H [1/Mpc]': np.array([2.0, 1.0]),
        }


def _params(**updates):
    params = {
        'h': 0.7,
        'Omega_m': 0.3,
        'output': 'tCl, mPk',
        'P_k_max_h/Mpc': 1.0,
        'z_max_pk': 1.0,
        'l_max_scalars': 1000,
    }
    params.update(updates)
    return params


def test_cache_reuses_compatible_computation(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()

    with cache.use(_params()):
        pass
    with cache.use(
            _params(output='mPk, tCl'),
            requirements={'output': 'tCl', 'l_max_scalars': 500}):
        pass

    assert len(FakeHiClass.instances) == 1
    assert cache.info()['hits'] == 1
    assert cache.info()['misses'] == 1


def test_cache_upgrades_coverage_and_cleans_previous_instance(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()

    with cache.use(_params()):
        pass
    with cache.use(
            _params(**{'P_k_max_h/Mpc': 2.0, 'z_max_pk': 3.0})):
        pass

    assert len(FakeHiClass.instances) == 2
    first, second = FakeHiClass.instances
    assert first.cleanup_calls == 1
    assert first.empty_calls == 1
    assert second.params['P_k_max_h/Mpc'] == 2.0
    assert second.params['z_max_pk'] == 3.0
    assert cache.info()['misses'] == 2


def test_changed_cosmology_or_precision_recomputes(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()

    with cache.use(_params()):
        pass
    with cache.use(_params(h=0.71)):
        pass
    with cache.use(_params(h=0.71, l_logstep=1.02)):
        pass

    assert len(FakeHiClass.instances) == 3
    assert cache.info()['hits'] == 0
    assert cache.info()['misses'] == 3


def test_clear_is_private_but_releases_native_state(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()
    with cache.use(_params()):
        pass

    cache.clear()

    instance = FakeHiClass.instances[0]
    assert instance.cleanup_calls == 1
    assert instance.empty_calls == 1
    assert cache.info()['computed'] is False


def test_pk_then_cell_share_one_computation(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()
    common_args = {
        'output': 'tCl, pCl, lCl, mPk, dTk',
        'P_k_max_h/Mpc': 50.0,
        'l_max_scalars': 3000,
        'lensing': 'yes',
    }

    pk = spectra_module.Pk.__new__(spectra_module.Pk)
    pk.name = 'pk_m'
    pk.class_args = common_args.copy()
    pk.class_high_prec = {}
    pk._class_cache = cache

    cell = spectra_module.Cell.__new__(spectra_module.Cell)
    cell.name = 'cl_TT_lensed'
    cell.class_args = common_args.copy()
    cell.class_high_prec = {}
    cell._class_cache = cache

    params = {'h': 0.7, 'Omega_m': 0.3}
    pk.get_from_class([0.1], [0.5], params)
    cell.get_from_class([2, 10], params)

    assert len(FakeHiClass.instances) == 1
    assert cache.info()['hits'] == 1


def test_hifast_exposes_only_private_cache_management(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    hifast = HiFast.__new__(HiFast)
    hifast._class_cache = HiClassCache()

    with hifast._class_cache.use(_params()):
        pass
    assert hifast._get_class_cache_info()['computed'] is True

    hifast._clear_class_cache()

    assert hifast._get_class_cache_info()['computed'] is False
    assert not hasattr(hifast, 'clear_class_cache')
    assert not hasattr(hifast, 'get_class_cache_info')


def test_combined_api_computes_cell_and_pk_once(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()
    common_args = {
        'output': 'tCl, pCl, lCl, mPk, dTk',
        'P_k_max_h/Mpc': 50.0,
        'l_max_scalars': 3000,
        'lensing': 'yes',
    }

    pk = spectra_module.Pk.__new__(spectra_module.Pk)
    pk.name = 'pk_m'
    pk.class_args = common_args.copy()
    pk.class_high_prec = {}
    pk._class_cache = cache

    cells = {}
    for cell_name in ('cl_TT_lensed', 'cl_EE_lensed'):
        cell = spectra_module.Cell.__new__(spectra_module.Cell)
        cell.name = cell_name
        cell.class_args = common_args.copy()
        cell.class_high_prec = {}
        cell._class_cache = cache
        cells[cell_name] = cell

    hifast = HiFast.__new__(HiFast)
    hifast._class_cache = cache
    hifast._spectra = {'pk_m': pk, **cells}
    params = {'h': 0.7, 'Omega_m': 0.3}

    # Seed the cache with the same narrow P(k, z) request. CMB metadata has a
    # much wider baseline P_k_max, which must not become a CMB requirement.
    hifast.get_pk_from_class([0.1], [0.5], params, name='m')
    result = hifast.get_from_class(
        params,
        observables={
            'cell': {
                'TT': {'ell': [2, 10]},
                'EE': {'ell': [2, 10]},
            },
            'pk': {'m': {'k': [0.1], 'z': [0.5]}},
        })

    assert result['cell']['TT'].shape == (1, 2)
    assert result['cell']['EE'].shape == (1, 2)
    assert result['pk']['m'].shape == (1, 1, 1)
    assert len(FakeHiClass.instances) == 1
    assert cache.info()['misses'] == 1
    assert cache.info()['hits'] == 4


def test_combined_api_validates_request_structure():
    service = HiClassService({})

    with np.testing.assert_raises_regex(ValueError, 'non-empty dictionary'):
        service.parse_observables({})
    with np.testing.assert_raises_regex(ValueError, 'Unknown observable'):
        service.parse_observables({'background': {'H': {}}})
    with np.testing.assert_raises_regex(ValueError, 'requires exactly'):
        service.parse_observables(
            {'pk': {'m': {'k': [0.1]}}})


def test_background_uses_fast_compute_and_shared_cache(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()
    pk = spectra_module.Pk.__new__(spectra_module.Pk)
    pk.name = 'pk_m'
    pk.class_args = {
        'output': 'tCl, pCl, lCl, mPk, dTk',
        'P_k_max_h/Mpc': 50.0,
        'l_max_scalars': 3000,
        'lensing': 'yes',
        'n_s': 0.96,
        'tau_reio': 0.054,
    }
    pk.class_high_prec = {}
    pk._class_cache = cache
    hifast = HiFast.__new__(HiFast)
    hifast._class_cache = cache
    hifast._spectra = {'pk_m': pk}
    params = {'h': 0.7, 'Omega_m': 0.3, 'n_s': 0.965,
              'tau_reio': 0.055}

    result = hifast.get_background(
        params, z=[0.0, 1.0], quantities=['H', 'age', 'Omega_m'])

    assert result['H'].shape == (2,)
    assert result['age'] == 13.8
    assert result['Omega_m'] == 0.3
    assert 'output' not in FakeHiClass.instances[0].params
    assert 'n_s' not in FakeHiClass.instances[0].params
    assert 'tau_reio' not in FakeHiClass.instances[0].params
    assert cache.info()['misses'] == 1

    hifast.get_pk_from_class([0.1], [0.5], params, name='m')
    hifast.get_background(params, z=0.5, quantities='H')

    assert len(FakeHiClass.instances) == 2
    assert cache.info()['misses'] == 2
    assert cache.info()['hits'] == 1


def test_background_validates_quantities_and_redshift():
    hifast = HiFast.__new__(HiFast)

    with np.testing.assert_raises_regex(ValueError, 'Unknown background'):
        hifast.get_background({}, quantities='not_a_quantity')
    with np.testing.assert_raises_regex(ValueError, 'z is required'):
        hifast.get_background({}, quantities='H')


def test_background_registry_is_the_metadata_source():
    entries = HiClassService.background_info()

    assert [entry['name'] for entry in entries] == list(
        HiClassService.BACKGROUND_QUANTITIES)
    assert next(entry for entry in entries if entry['name'] == 'H') == {
        'name': 'H',
        'hiclassy': 'Hubble(z)',
        'input': 'z',
        'units': 'km/s/Mpc',
    }
    assert next(entry for entry in entries if entry['name'] == 'age') == {
        'name': 'age',
        'hiclassy': 'age',
        'input': '—',
        'units': 'Gyr',
    }


def test_background_table_returns_native_columns(monkeypatch):
    FakeHiClass.instances = []
    monkeypatch.setattr(cache_module.hiclassy, 'HiClass', FakeHiClass)
    cache = HiClassCache()
    pk = spectra_module.Pk.__new__(spectra_module.Pk)
    pk.name = 'pk_m'
    pk.class_args = {'n_s': 0.96, 'tau_reio': 0.054}
    pk.class_high_prec = {}
    pk._class_cache = cache
    hifast = HiFast.__new__(HiFast)
    hifast._class_cache = cache
    hifast._spectra = {'pk_m': pk}

    table = hifast.get_background_table(
        {'h': 0.7, 'Omega_m': 0.3, 'n_s': 0.965,
         'tau_reio': 0.055})
    table['z'][0] = 99.0
    second = hifast.get_background_table(
        {'h': 0.7, 'Omega_m': 0.3, 'n_s': 0.965,
         'tau_reio': 0.055})

    assert set(table) == {'z', 'H [1/Mpc]'}
    np.testing.assert_array_equal(second['z'], [1.0, 0.0])
    assert FakeHiClass.instances[0].background_calls == 1
    assert cache.info()['hits'] == 1
