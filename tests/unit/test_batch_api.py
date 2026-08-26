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

    def get_from_class(self, k, z, params, **kwargs):
        """Return the native CLASS orientation: (n_k, n_z)."""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        return k[:, np.newaxis] + z[np.newaxis, :]


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

    def get_from_class(self, ell, params, **kwargs):
        """Return the native one-cosmology CLASS orientation."""
        return np.atleast_1d(ell).astype(float)


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
    assert output.shape == (1, 1, 2)


def test_squeeze_removes_singleton_batch_dimension(hifast):
    output = hifast.get_pk(
        [0.1, 0.2], 0.5, {'a': 1.0, 'b': 2.0},
        name='m', squeeze=True)
    assert output.shape == (2,)


@pytest.mark.parametrize('squeeze, expected_shape', [
    (False, (1, 2, 3)),
    (True, (2, 3)),
])
def test_pk_and_class_have_matching_shapes(hifast, squeeze, expected_shape):
    k = np.array([0.1, 0.2, 0.3])
    z = np.array([0.0, 0.5])
    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_pk(k, z, params, name='m', squeeze=squeeze)
    from_class = hifast.get_pk_from_class(
        k, z, params, name='m', squeeze=squeeze)

    assert np.asarray(emulator).shape == expected_shape
    assert np.asarray(from_class).shape == expected_shape


@pytest.mark.parametrize('get_from_pk', [False, True])
@pytest.mark.parametrize('squeeze, expected_shape', [
    (False, (1, 2, 3)),
    (True, (2, 3)),
])
def test_fk_and_class_have_matching_shapes(
        hifast, get_from_pk, squeeze, expected_shape):
    k = np.array([0.1, 0.2, 0.3])
    z = np.array([0.0, 0.5])
    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_fk(
        k, z, params, name='m', get_from_pk=get_from_pk,
        squeeze=squeeze)
    from_class = hifast.get_fk_from_class(
        k, z, params, name='m', squeeze=squeeze)

    assert np.asarray(emulator).shape == expected_shape
    assert np.asarray(from_class).shape == expected_shape


@pytest.mark.parametrize('squeeze, expected_shape', [
    (False, (1, 3)),
    (True, (3,)),
])
def test_cell_and_class_have_matching_shapes(
        hifast, squeeze, expected_shape):
    ell = np.array([2, 10, 20])
    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_cell(
        ell, params, name='TT', squeeze=squeeze)
    from_class = hifast.get_cell_from_class(
        ell, params, name='TT', squeeze=squeeze)

    assert np.asarray(emulator).shape == expected_shape
    assert np.asarray(from_class).shape == expected_shape


def test_rejects_wrong_array_width(hifast):
    with pytest.raises(ValueError, match=r'Expected 2 parameter columns'):
        hifast.get_pk([0.1], 0.0, np.ones(3), name='m')


@pytest.mark.parametrize('batch_size', [0, -1, 1.5])
def test_rejects_invalid_batch_size(hifast, batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        hifast.get_pk(
            [0.1], 0.0, {'a': 1.0, 'b': 2.0}, name='m',
            batch_size=batch_size)


@pytest.mark.parametrize('method, kwargs, offset', [
    ('get_pk', {'name': 'm'}, 0.0),
    ('get_fk', {'name': 'm', 'get_from_pk': False}, 20.0),
    ('get_fk', {'name': 'm', 'get_from_pk': True}, 10.0),
])
def test_pk_and_fk_evaluate_full_cosmology_redshift_grid(
        hifast, method, kwargs, offset):
    k = np.array([0.1, 0.2])
    z = np.array([0.0, 0.5, 1.0])
    params = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = getattr(hifast, method)(k, z, params, **kwargs)
    expected = np.asarray([
        [row[0] + 2*row[1] + z_one + offset + k for z_one in z]
        for row in params
    ])

    assert result.shape == (2, 3, 2)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('method, kwargs, offset', [
    ('get_pk', {'name': 'm'}, 0.0),
    ('get_fk', {'name': 'm', 'get_from_pk': False}, 20.0),
    ('get_fk', {'name': 'm', 'get_from_pk': True}, 10.0),
])
def test_pk_and_fk_paired_mode_evaluates_corresponding_rows(
        hifast, method, kwargs, offset):
    k = np.array([0.1, 0.2])
    z = np.array([0.0, 0.5, 1.0])
    params = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    function = getattr(hifast, method)

    complete = function(
        k, z, params, paired=True, batch_size=None, **kwargs)
    chunked = function(
        k, z, params, paired=True, batch_size=2, **kwargs)
    expected = np.asarray([
        row[0] + 2*row[1] + z_one + offset + k
        for row, z_one in zip(params, z)
    ])

    assert complete.shape == (3, 2)
    np.testing.assert_allclose(complete, expected)
    np.testing.assert_allclose(chunked, expected)


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_paired_mode_rejects_mismatched_lengths(hifast, method, kwargs):
    with pytest.raises(ValueError, match='equal lengths'):
        getattr(hifast, method)(
            [0.1], [0.0, 1.0],
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            paired=True, **kwargs)


def test_unknown_spectrum_name_is_informative(hifast):
    with pytest.raises(ValueError, match='not available'):
        hifast.get_params_names('pk_unknown')
