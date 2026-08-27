"""Unit tests for the private shared HiCLASS computation cache."""

import numpy as np

import hi_fast._class_cache as cache_module
from hi_fast._class_cache import HiClassCache
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

    def get_pk_lin(self, k, z, n_k, n_z, n_mu):
        return np.ones((n_k, n_z, n_mu))

    def lensed_cl(self, lmax):
        values = np.ones(lmax + 1)
        return {'tt': values, 'ee': values, 'te': values,
                'bb': values, 'tp': values, 'pp': values}


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
