"""
.. module:: scalers

:Synopsis: List of possible scalers.
:Author: Emilio Bellini

"""

import numpy as np
import sklearn.preprocessing as skl_pre


class Scaler(object):
    """Base class for all scalers used by the emulator pipeline."""

    _REGISTRY = {}

    def __init__(self, **kwargs):
        """Record the scaler name/type from serialized metadata.

        Args:
            **kwargs: Keyword arguments loaded from disk. Must include a
                ``type`` field identifying the scaler variant.
        """
        self.name = kwargs['type']
        return

    def _replace_inf(self, x, factor=10.):
        """Replace infinities by large finite values.

        Args:
            x (numpy.ndarray): Input array.
            factor (float): Multiplier applied to the largest finite value.

        Returns:
            numpy.ndarray: Array with ``inf`` replaced while preserving sign.
        """
        signs = np.sign(x)
        x_new = np.abs(x)
        x_new[np.isinf(x_new)] = np.nan
        inf = factor*np.nanmax(x_new, axis=0)[np.newaxis, :]
        nans = np.multiply(np.isnan(x_new), inf)
        x_new[np.isnan(x_new)] = 0.
        x_new = np.multiply(x_new + nans, signs)
        return x_new

    @staticmethod
    def choose_one(**kwargs):
        """Instantiate the appropriate scaler based on serialized metadata.

        Args:
            **kwargs: Serialized scaler parameters that must include
                ``type`` identifying the scaler class.

        Returns:
            Scaler | None: Concrete scaler instance or ``None`` for
            ``NoneScaler``.

        Raises:
            ValueError: If ``type`` does not match any known scaler.
        """
        scaler_type = kwargs['type']
        try:
            scaler_class = Scaler._REGISTRY[scaler_type]
        except KeyError as error:
            raise ValueError(
                'Scaler {!r} not recognized. Choose from {}'
                .format(scaler_type, sorted(
                    str(name) for name in Scaler._REGISTRY))) from error
        return scaler_class(**kwargs)

    def transform(self, x):
        """Placeholder to be overridden by subclasses."""
        return None

    def inverse_transform(self, x_scaled):
        """Placeholder to be overridden by subclasses."""
        return None


class NoneScaler(Scaler):
    """No-op scaler useful for debugging or optional preprocessing."""

    def __init__(self, **kwargs):
        """Initialize a no-op scaler from serialized metadata."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = None
        return

    def transform(self, x, replace_infinity=True):
        """Return input as-is, optionally replacing infinities.

        Args:
            x (numpy.ndarray): Input data.
            replace_infinity (bool): When True, call ``_replace_inf`` first.

        Returns:
            numpy.ndarray: Possibly sanitized copy of ``x``.
        """
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        return x_scaled

    def inverse_transform(self, x):
        """Return unscaled values unchanged."""
        return x


class StandardScaler(Scaler):
    """Standardize features by removing the mean and scaling to unit
    variance."""

    def __init__(self, **kwargs):
        """Restore a fitted standard scaler from serialized attributes."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.StandardScaler()
        # Store Scaler attributes
        self.skl_scaler.mean_ = kwargs['mean_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.var_ = kwargs['var_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        """Apply scikit-learn standard scaling after sanitizing infinities."""
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Undo the standard scaling operation."""
        x = self.skl_scaler.inverse_transform(x_scaled)
        return x


class LogStandardScaler(Scaler):
    """Log-transform followed by standard scaling."""

    def __init__(self, **kwargs):
        """Restore a fitted log-standard scaler."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.StandardScaler()
        # Store Scaler attributes
        self.skl_scaler.mean_ = kwargs['mean_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.var_ = kwargs['var_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        """Log the input and apply standard scaling."""
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(np.log(x_scaled))
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Undo scaling and exponential to recover original values."""
        x = np.exp(self.skl_scaler.inverse_transform(x_scaled))
        return x


class MinusLogStandardScaler(Scaler):
    """Log-transform the negated input, then apply standard scaling."""

    def __init__(self, **kwargs):
        """Restore a fitted negative-log standard scaler."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.StandardScaler()
        # Store Scaler attributes
        self.skl_scaler.mean_ = kwargs['mean_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.var_ = kwargs['var_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        """Apply log-scaling to the negative input before standardization."""
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(np.log(-x_scaled))
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Recover the original values by undoing scaling and log."""
        x = -np.exp(self.skl_scaler.inverse_transform(x_scaled))
        return x


class MinMaxScaler(Scaler):
    """Scale each feature to the ``(0, 1)`` interval."""

    def __init__(self, **kwargs):
        """Restore a fitted feature-wise min-max scaler."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.MinMaxScaler()
        # Store Scaler attributes
        self.skl_scaler.min_ = kwargs['min_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.data_min_ = kwargs['data_min_']
        self.skl_scaler.data_max_ = kwargs['data_max_']
        self.skl_scaler.data_range_ = kwargs['data_range_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        """Apply min-max scaling after optional infinity replacement."""
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Undo the min-max scaling."""
        x = self.skl_scaler.inverse_transform(x_scaled)
        return x


class MinMaxCommonScaler(Scaler):
    """Apply a global (feature-shared) min-max scaling."""

    def __init__(self, **kwargs):
        """Restore global min-max bounds from serialized metadata."""
        Scaler.__init__(self, **kwargs)
        # Store Scaler attributes
        self.glob_min_ = kwargs['glob_min_']
        self.glob_max_ = kwargs['glob_max_']
        return

    def transform(self, x, replace_infinity=True):
        """Scale using shared ``glob_min_``/``glob_max_`` limits."""
        if replace_infinity:
            x = self._replace_inf(x)
        if self.glob_min_ == 0. and self.glob_max_ == 0.:
            x_scaled = x
        elif self.glob_min_ == self.glob_max_:
            x_scaled = x/self.glob_max_
        else:
            x_scaled = (x - self.glob_min_)/(self.glob_max_ - self.glob_min_)
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Undo the global min-max scaling."""
        if self.glob_min_ == 0. and self.glob_max_ == 0.:
            x = x_scaled
        elif self.glob_min_ == self.glob_max_:
            x = x_scaled * self.glob_max_
        else:
            x = x_scaled * (self.glob_max_ - self.glob_min_) + self.glob_min_
        return x


class MinMaxPlus1Scaler(Scaler):
    """Scale features to ``(1, 2)`` to avoid zeros."""

    def __init__(self, **kwargs):
        """Restore a fitted shifted min-max scaler."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.MinMaxScaler()
        # Store Scaler attributes
        self.skl_scaler.min_ = kwargs['min_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.data_min_ = kwargs['data_min_']
        self.skl_scaler.data_max_ = kwargs['data_max_']
        self.skl_scaler.data_range_ = kwargs['data_range_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        """Min-max scale then shift by +1."""
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled) + 1.
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Undo the ``(1, 2)`` scaling by subtracting 1 before inverse."""
        x = self.skl_scaler.inverse_transform(x_scaled - 1.)
        return x


class ExpMinMaxScaler(Scaler):
    """Min-max scale features and then exponentiate the result."""

    def __init__(self, **kwargs):
        """Restore a fitted exponential min-max scaler."""
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.MinMaxScaler()
        # Store Scaler attributes
        self.skl_scaler.min_ = kwargs['min_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.data_min_ = kwargs['data_min_']
        self.skl_scaler.data_max_ = kwargs['data_max_']
        self.skl_scaler.data_range_ = kwargs['data_range_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        """Apply min-max scaling followed by ``exp``."""
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        x_scaled = np.exp(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        """Take log first, then undo the min-max scaling."""
        x = np.log(x_scaled)
        x = self.skl_scaler.inverse_transform(x)
        return x


# Serialized scaler name -> implementation. Keeping this in one registry
# makes support for new scaler types explicit and avoids factory conditionals.
Scaler._REGISTRY = {
    None: NoneScaler,
    'None': NoneScaler,
    'StandardScaler': StandardScaler,
    'LogStandardScaler': LogStandardScaler,
    'MinusLogStandardScaler': MinusLogStandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'MinMaxCommonScaler': MinMaxCommonScaler,
    'MinMaxPlus1Scaler': MinMaxPlus1Scaler,
    'ExpMinMaxScaler': ExpMinMaxScaler,
}
