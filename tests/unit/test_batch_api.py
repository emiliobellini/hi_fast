"""Unit tests for the public batch-first HiFast API."""
import numpy as np
import pytest

from hi_fast import HiFast


class FakeParams:
    """Minimal replacement for Params that records no external state."""

    def get(self, params, **kwargs):
        return params.copy()


class FakePk:
    name = 'pk_m'
    input_params_names = ['a', 'b']
    ref = {'params': {'ln_A_s_1e10': 3.0, 'n_s': 0.96}}

    def __init__(self):
        self.batch_lengths = []

    def _values(self, k, z, params, offset):
        self.batch_lengths.append(len(params))
        k = np.asarray(k)
        return np.asarray([
            row['a'] + 2*row['b'] + z_one + offset + k
            for row, z_one in zip(params, z)
        ])

    def get(self, k, z, params):
        return self._values(k, z, params, offset=0.0)

    def get_fk(self, k, z, params):
        return self._values(k, z, params, offset=10.0)


class FakeFk(FakePk):
    name = 'fk_m'
    input_params_names = ['a', 'b']

    def get(self, k, z, params):
        return self._values(k, z, params, offset=20.0)


class FakeCell:
    name = 'cl_TT_lensed'
    input_params_names = ['a', 'b']

    def __init__(self):
        self.batch_lengths = []

    def get(self, ell, params):
        self.batch_lengths.append(len(params))
        ell = np.asarray(ell)
        return np.asarray([
            row['a'] + 2*row['b'] + ell for row in params
        ])


@pytest.fixture
def hifast():
    """Construct HiFast without loading TensorFlow models from disk."""
    instance = HiFast.__new__(HiFast)
    pk = FakePk()
    fk = FakeFk()
    cell = FakeCell()
    instance._spectra = {
        pk.name: pk,
        fk.name: fk,
        cell.name: cell,
    }
    instance._params = {
        name: FakeParams() for name in instance._spectra
    }
    return instance


def test_get_params_names_returns_public_array_order(hifast):
    assert hifast.get_params_names('pk_m') == ['a', 'b']
    assert hifast.get_params_names('fk_m') == ['a', 'b']
    assert hifast.get_params_names('cl_TT_lensed') == ['a', 'b']


def test_get_params_names_uses_pk_fallback_for_missing_fk(hifast):
    del hifast._spectra['fk_m']
    hifast._spectra['pk_m'].input_params_names = [
        'a', 'b', 'ln_A_s_1e10', 'n_s']
    assert hifast.get_params_names('fk_m') == ['a', 'b']


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_dictionary_and_array_inputs_are_equivalent(hifast, method, kwargs):
    dictionaries = [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    z = np.array([0.0, 1.0])
    k = np.array([0.1, 0.2])

    from_dicts = getattr(hifast, method)(k, z, dictionaries, **kwargs)
    from_array = getattr(hifast, method)(k, z, array, **kwargs)
    np.testing.assert_allclose(from_array, from_dicts)


def test_cell_dictionary_and_array_inputs_are_equivalent(hifast):
    dictionaries = [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    ell = np.array([2, 10])
    expected = hifast.get_cell(ell, dictionaries, name='TT')
    result = hifast.get_cell(ell, array, name='TT')
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_internal_chunking_preserves_values_and_order(hifast, method, kwargs):
    params = np.arange(14, dtype=float).reshape(7, 2)
    z = np.linspace(0.0, 1.0, 7)
    k = np.array([0.1, 0.2, 0.3])
    function = getattr(hifast, method)

    complete = function(k, z, params, batch_size=None, **kwargs)
    chunked = function(k, z, params, batch_size=3, **kwargs)
    np.testing.assert_allclose(chunked, complete)


def test_cell_internal_chunking_preserves_values_and_order(hifast):
    params = np.arange(14, dtype=float).reshape(7, 2)
    ell = np.array([2, 10, 20])
    complete = hifast.get_cell(ell, params, name='TT')
    chunked = hifast.get_cell(ell, params, name='TT', batch_size=3)
    np.testing.assert_allclose(chunked, complete)


def test_one_dimensional_array_is_one_cosmology(hifast):
    output = hifast.get_pk(
        [0.1, 0.2], 0.5, np.array([1.0, 2.0]), name='m')
    assert output.shape == (1, 2)


def test_squeeze_removes_singleton_batch_dimension(hifast):
    output = hifast.get_pk(
        [0.1, 0.2], 0.5, {'a': 1.0, 'b': 2.0},
        name='m', squeeze=True)
    assert output.shape == (2,)


def test_rejects_wrong_array_width(hifast):
    with pytest.raises(ValueError, match=r'Expected 2 parameter columns'):
        hifast.get_pk([0.1], 0.0, np.ones(3), name='m')


@pytest.mark.parametrize('batch_size', [0, -1, 1.5])
def test_rejects_invalid_batch_size(hifast, batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        hifast.get_pk(
            [0.1], 0.0, {'a': 1.0, 'b': 2.0}, name='m',
            batch_size=batch_size)


def test_rejects_mismatched_parameter_and_redshift_counts(hifast):
    with pytest.raises(ValueError, match='same number of cosmologies'):
        hifast.get_pk(
            [0.1], [0.0, 1.0, 2.0],
            np.array([[1.0, 2.0], [3.0, 4.0]]), name='m')


def test_unknown_spectrum_name_is_informative(hifast):
    with pytest.raises(ValueError, match='not available'):
        hifast.get_params_names('pk_unknown')
