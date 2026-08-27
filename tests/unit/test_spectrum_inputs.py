"""Regression tests for spectrum coordinate normalization and caching."""

import numpy as np
import pytest

from hi_fast.spectra import Cell, Fk, GridSpectrum, Pk, Spectrum
from hi_fast.scalers import NoneScaler, Scaler


def test_scaler_registry_drives_serialized_type_selection():
    assert Scaler.choose_one(type='None').__class__ is NoneScaler
    assert Scaler.choose_one(type=None).__class__ is NoneScaler
    assert set(Scaler._REGISTRY) == {
        None, 'None', 'StandardScaler', 'LogStandardScaler',
        'MinusLogStandardScaler', 'MinMaxScaler', 'MinMaxCommonScaler',
        'MinMaxPlus1Scaler', 'ExpMinMaxScaler',
    }

    with pytest.raises(ValueError, match="'unknown' not recognized"):
        Scaler.choose_one(type='unknown')


def test_grid_spectrum_hierarchy_separates_pk_and_fk_interfaces():
    assert issubclass(Pk, GridSpectrum)
    assert issubclass(Fk, GridSpectrum)
    assert not issubclass(Fk, Pk)
    assert not hasattr(Fk, 'get_sigma_R')


def test_coordinate_normalization_accepts_general_sequences():
    spectrum = Spectrum.__new__(Spectrum)

    np.testing.assert_array_equal(
        spectrum._to_numpy_array((0, 0.5, 1)), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(
        spectrum._to_numpy_array(np.float64(0.5)), [0.5])


@pytest.mark.parametrize('values, message', [
    ([], 'non-empty'),
    ([[0.0, 1.0]], 'one-dimensional'),
    ([0.0, np.nan], 'finite'),
    (['not-a-number'], 'numeric'),
])
def test_coordinate_normalization_rejects_invalid_inputs(values, message):
    spectrum = Spectrum.__new__(Spectrum)

    with pytest.raises(ValueError, match=message):
        spectrum._to_numpy_array(values)


def _cell_with_reference_grid():
    cell = Cell.__new__(Cell)
    cell.ref = {
        'ell': np.arange(2, 11),
        'spectrum': np.arange(2, 11, dtype=float),
    }
    cell.ell_min = 2
    cell.ell_max = 10
    cell.stored = {
        'ell': None,
        'ell_indices': None,
        'ref_spectrum': None,
    }
    return cell


def test_cell_indices_preserve_requested_order_and_duplicates():
    cell = _cell_with_reference_grid()
    ell = cell._to_numpy_array([10, 2, 10])

    cell._store_ell_indices(ell)

    np.testing.assert_array_equal(cell.stored['ell_indices'], [8, 0, 8])
    np.testing.assert_array_equal(cell.stored['ref_spectrum'], [10, 2, 10])


def test_cell_rejects_fractional_multipoles():
    cell = _cell_with_reference_grid()

    with pytest.raises(ValueError, match='integers'):
        cell._to_numpy_array([2, 3.5])


def test_reference_grid_caches_do_not_alias_caller_arrays():
    pk = Pk.__new__(Pk)
    pk.ref = {'spectrum_z_spline': None}
    pk.stored = {'k': None, 'z': None, 'ref_spectrum': None}
    k = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    pk._store_reference_spectrum(k, z)
    k[0] = 0.5
    z[0] = 0.5

    np.testing.assert_array_equal(pk.stored['k'], [0.1, 0.2])
    np.testing.assert_array_equal(pk.stored['z'], [0.0, 1.0])

    cell = _cell_with_reference_grid()
    ell = np.array([10, 2])
    cell._store_ell_indices(ell)
    ell[0] = 3

    np.testing.assert_array_equal(cell.stored['ell'], [10, 2])


@pytest.mark.parametrize('spectrum, method', [
    (Pk.__new__(Pk), 'get'),
    (Pk.__new__(Pk), 'get_from_class'),
    (Pk.__new__(Pk), 'get_fk'),
    (Fk.__new__(Fk), 'get'),
    (Fk.__new__(Fk), 'get_from_class'),
])
def test_grid_spectra_consistently_reject_nonlinear_requests(
        spectrum, method):
    with pytest.raises(ValueError, match='Nonlinear Pk not yet implemented'):
        getattr(spectrum, method)([0.1], [0.0], {}, nonlinear=True)
