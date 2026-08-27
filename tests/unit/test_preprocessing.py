"""Numerical tests for serialized scalers and PCA reconstruction."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from hi_fast.pca import PCA
from hi_fast.scalers import Scaler
from hi_fast.spectra import Spectrum
from hi_fast._tensorflow_config import tf


def _standard_metadata(name, fitted):
    return {
        'type': name,
        'mean_': fitted.mean_,
        'scale_': fitted.scale_,
        'var_': fitted.var_,
        'n_samples_seen_': fitted.n_samples_seen_,
    }


def _minmax_metadata(name, fitted):
    return {
        'type': name,
        'min_': fitted.min_,
        'scale_': fitted.scale_,
        'data_min_': fitted.data_min_,
        'data_max_': fitted.data_max_,
        'data_range_': fitted.data_range_,
        'n_samples_seen_': fitted.n_samples_seen_,
    }


@pytest.mark.parametrize('name,values,fit_values', [
    ('StandardScaler', np.array([[2.0, 5.0], [4.0, 9.0]]),
     np.array([[1.0, 3.0], [5.0, 11.0]])),
    ('LogStandardScaler', np.array([[2.0, 5.0], [4.0, 9.0]]),
     np.log(np.array([[1.0, 3.0], [5.0, 11.0]]))),
    ('MinusLogStandardScaler', np.array([[-2.0, -5.0], [-4.0, -9.0]]),
     np.log(np.array([[1.0, 3.0], [5.0, 11.0]]))),
])
def test_standard_scaler_variants_round_trip(name, values, fit_values):
    fitted = SklearnStandardScaler().fit(fit_values)
    scaler = Scaler.choose_one(**_standard_metadata(name, fitted))
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)), values)


@pytest.mark.parametrize('name', [
    'MinMaxScaler', 'MinMaxPlus1Scaler', 'ExpMinMaxScaler'])
def test_minmax_scaler_variants_round_trip(name):
    fitted = SklearnMinMaxScaler().fit([[1.0, 3.0], [5.0, 11.0]])
    scaler = Scaler.choose_one(**_minmax_metadata(name, fitted))
    values = np.array([[2.0, 5.0], [4.0, 9.0]])
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)), values)


@pytest.mark.parametrize('low,high', [(0.0, 0.0), (2.0, 2.0), (1.0, 5.0)])
def test_common_minmax_scaler_round_trip(low, high):
    scaler = Scaler.choose_one(
        type='MinMaxCommonScaler', glob_min_=low, glob_max_=high)
    values = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(
            values, replace_infinity=False)), values)


def test_infinity_replacement_preserves_sign_and_uses_finite_scale():
    scaler = Scaler.choose_one(type='None')
    result = scaler.transform(np.array([[1.0, -2.0], [np.inf, -np.inf]]))
    np.testing.assert_allclose(result, [[1.0, -2.0], [10.0, -20.0]])


def test_tensorflow_and_numpy_standard_scaling_agree():
    fitted = SklearnStandardScaler().fit([[1.0, 3.0], [5.0, 11.0]])
    scaler = Scaler.choose_one(**_standard_metadata('StandardScaler', fitted))
    values = np.array([[2.0, 5.0], [4.0, 9.0]])
    tensorflow_result = Spectrum._tf_scaler_transform(scaler, values).numpy()
    np.testing.assert_allclose(tensorflow_result, scaler.transform(values))
    inverse = Spectrum._tf_scaler_inverse_transform(
        scaler, tensorflow_result).numpy()
    np.testing.assert_allclose(inverse, values)


def test_serialized_pca_matches_fitted_transform_and_inverse():
    values = np.array([
        [1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [4.0, 2.0, 1.0],
        [3.0, 5.0, 2.0],
    ])
    fitted = SklearnPCA(n_components=2).fit(values)
    restored = PCA(**{
        name: getattr(fitted, name) for name in (
            'n_components_', 'components_', 'mean_', 'explained_variance_',
            'explained_variance_ratio_', 'singular_values_', 'n_samples_',
            'n_features_in_', 'noise_variance_')
    })
    transformed = restored.transform(values)
    np.testing.assert_allclose(transformed, fitted.transform(values))
    np.testing.assert_allclose(
        restored.inverse_transform(transformed),
        fitted.inverse_transform(fitted.transform(values)))

    tensorflow_result = Spectrum._tf_pca_transform(
        restored, tf.convert_to_tensor(values, dtype=tf.float64)).numpy()
    np.testing.assert_allclose(tensorflow_result, transformed)
    tensorflow_inverse = Spectrum._tf_pca_inverse_transform(
        restored, tf.convert_to_tensor(
            tensorflow_result, dtype=tf.float64)).numpy()
    np.testing.assert_allclose(
        tensorflow_inverse, restored.inverse_transform(transformed))


def test_batch_emulator_pipeline_applies_scalers_and_model():
    fitted = SklearnStandardScaler().fit([[1.0, 3.0], [5.0, 11.0]])
    scaler = Scaler.choose_one(**_standard_metadata('StandardScaler', fitted))

    class DoubleModel:
        def __call__(self, values, training=False):
            assert training is False
            return 2.0 * values

    spectrum = Spectrum.__new__(Spectrum)
    spectrum.x_scaler = scaler
    spectrum.x_pca = None
    spectrum.model = DoubleModel()
    spectrum.y_pca = None
    spectrum.y_scaler = scaler
    values = np.array([[2.0, 5.0], [4.0, 9.0]])

    expected = scaler.inverse_transform(2.0 * scaler.transform(values))
    np.testing.assert_allclose(spectrum._eval_emu_batch(values), expected)


def test_forward_mode_derivative_tracks_redshift_per_batch_row():
    class RedshiftModel(tf.keras.Model):
        def call(self, values, training=False):
            redshift = values[:, 1]
            return tf.stack((redshift**2, 3.0 * redshift), axis=1)

    spectrum = Spectrum.__new__(Spectrum)
    spectrum.x_names = ['a', 'z_pk']
    spectrum.x_scaler = None
    spectrum.x_pca = None
    spectrum.model = RedshiftModel(dtype='float64')
    spectrum.y_pca = None
    spectrum.y_scaler = None

    values, derivative = spectrum._eval_emu_and_dz(
        [{'a': 1.0}, {'a': 2.0}], np.array([0.5, 2.0]))
    np.testing.assert_allclose(values, [[0.25, 1.5], [4.0, 6.0]])
    np.testing.assert_allclose(derivative, [[1.0, 3.0], [4.0, 3.0]])
