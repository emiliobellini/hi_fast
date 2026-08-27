"""Unit tests for cosmological parameter conversion and validation."""

import numpy as np
import pytest

from hi_fast.params import Params


class FakePowerSpectrum:
    """Minimal spectrum metadata and amplitude conversion model."""

    name = 'pk_m'
    input_params_names = [
        'h', 'Omega_m', 'Omega_b', 'ln_A_s_1e10', 'n_s']
    x_names = input_params_names
    x_ranges = [
        [0.5, 0.9], [0.1, 0.8], [0.02, 0.08], [1.0, 5.0], [0.7, 1.3]]
    x_ranges_by_region = {
        'thin': [
            [0.65, 0.73], [0.28, 0.35], [0.044, 0.047],
            [2.9, 3.2], [0.9, 1.0]],
        'std': x_ranges,
        'ext': x_ranges,
    }

    @staticmethod
    def get_sigma8_from_params(params):
        return 0.25 * params['ln_A_s_1e10']

    @staticmethod
    def get_S8_from_params(params):
        return (0.25 * params['ln_A_s_1e10']
                * np.sqrt(params['Omega_m'] / 0.3))


@pytest.fixture
def handler():
    spectrum = FakePowerSpectrum()
    return Params(spectrum, {'pk_m': spectrum, 'pk_cb': spectrum})


def test_standard_parameter_aliases_are_converted_in_dependency_order(handler):
    result = handler.get({
        'H0': 70.0,
        'omega_m': 0.147,
        'omega_b': 0.0245,
        'A_s': 2.1e-9,
        'n_s': 0.96,
    })

    assert result['h'] == pytest.approx(0.7)
    assert result['Omega_m'] == pytest.approx(0.3)
    assert result['Omega_b'] == pytest.approx(0.05)
    assert result['ln_A_s_1e10'] == pytest.approx(np.log(21.0))
    assert not {'H0', 'omega_m', 'omega_b', 'A_s'} & set(result)


def test_parameter_name_validation_rejects_missing_and_duplicate_aliases(
        handler):
    base = {
        'h': 0.7, 'Omega_m': 0.3, 'Omega_b': 0.05,
        'ln_A_s_1e10': 3.0, 'n_s': 0.96,
    }
    missing = base.copy()
    missing.pop('n_s')
    with pytest.raises(Exception, match='required, but not provided'):
        handler.get(missing)

    duplicate = {**base, 'H0': 70.0}
    with pytest.raises(Exception, match='Only one parameter'):
        handler.get(duplicate)


def test_trusted_region_validation_and_boolean_query(handler):
    params = {
        'h': 0.8, 'Omega_m': 0.3, 'Omega_b': 0.05,
        'ln_A_s_1e10': 3.0, 'n_s': 0.96,
    }
    assert handler.is_in_bounds(params, trusted_region='std')
    assert not handler.is_in_bounds(params, trusted_region='thin')
    with pytest.raises(ValueError, match='thin trusted region'):
        handler.get(params, trusted_region='thin')
    assert not handler.is_in_bounds(params, trusted_region='wide')
    with pytest.raises(ValueError, match='trusted_region must be one of'):
        handler.get(params, trusted_region='wide')


@pytest.mark.parametrize('alias,target,expected', [
    ('sigma8_m', 0.8, 3.2),
    ('S8_m', 0.8, 3.2),
])
def test_shooting_converts_amplitude_parameters(handler, alias, target,
                                                expected):
    params = {
        'h': 0.7, 'Omega_m': 0.3, 'Omega_b': 0.05,
        alias: target, 'n_s': 0.96,
    }
    result = handler.get(params)
    assert result['ln_A_s_1e10'] == pytest.approx(expected)
    assert alias not in result
