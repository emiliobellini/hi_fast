"""
.. module:: scalers

:Synopsis: List of possible scalers.
:Author: Emilio Bellini

"""

import numpy as np
import sklearn.preprocessing as skl_pre


class Scaler(object):
    """
    Base Scaler class.
    Each one of the other scalers (see below), should
    inherit from this and define three other methods:
    - transform: transform data using fitted scaler
    - inverse_transform: transform back data.
    """

    def __init__(self, **kwargs):
        self.name = kwargs['type']
        return

    def _replace_inf(self, x, factor=10.):
        """
        This is used to replace infinities with
        large numbers. In practice, given an array x,
        it takes the maximum value of abs(x) and
        multiplies it by 'factor'. The resulting
        number is going to replace all infinities
        (with the correct sign).
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
        """
        Main function to get the correct Scaler.

        Arguments:
            - scaler_type (str): type of scaler.

        Return:
            - Scaler (object): get the correct
              scaler and initialize it.
        """
        if kwargs['type'] == 'None' or kwargs['type'] is None:
            return NoneScaler(**kwargs)
        elif kwargs['type'] == 'StandardScaler':
            return StandardScaler(**kwargs)
        elif kwargs['type'] == 'LogStandardScaler':
            return LogStandardScaler(**kwargs)
        elif kwargs['type'] == 'MinusLogStandardScaler':
            return MinusLogStandardScaler(**kwargs)
        elif kwargs['type'] == 'MinMaxScaler':
            return MinMaxScaler(**kwargs)
        elif kwargs['type'] == 'MinMaxCommonScaler':
            return MinMaxCommonScaler(**kwargs)
        elif kwargs['type'] == 'MinMaxPlus1Scaler':
            return MinMaxPlus1Scaler(**kwargs)
        elif kwargs['type'] == 'ExpMinMaxScaler':
            return ExpMinMaxScaler(**kwargs)
        else:
            raise ValueError('Scaler not recognized!')

    def transform(self, x):
        """
        Transform an array x into the corresponding
        x_scaled, using results from the fit method.
        NOTE: If you want to implement a new scaler
        return a new rescaled array.
        """
        return None

    def inverse_transform(self, x_scaled):
        """
        Transform back an array x_scaled into
        the corresponding x, using results from
        the fit method.
        NOTE: If you want to implement a new scaler
        return a new inverse rescaled array.
        """
        return None


class NoneScaler(Scaler):
    """
    Do not rescale.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = None
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        return x_scaled

    def inverse_transform(self, x):
        return x


class StandardScaler(Scaler):
    """
    Standardise features by removing the
    mean and scaling to unit variance.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.StandardScaler()
        self.skl_scaler.mean_ = kwargs['mean_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.var_ = kwargs['var_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.skl_scaler.inverse_transform(x_scaled)
        return x


class LogStandardScaler(Scaler):
    """
    Take the log of the features and then standardise them
    by removing the mean and scaling to unit variance.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.StandardScaler()
        self.skl_scaler.mean_ = kwargs['mean_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.var_ = kwargs['var_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(np.log(x_scaled))
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = np.exp(self.skl_scaler.inverse_transform(x_scaled))
        return x


class MinusLogStandardScaler(Scaler):
    """
    Take the log of the features and then standardise them
    by removing the mean and scaling to unit variance.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.StandardScaler()
        self.skl_scaler.mean_ = kwargs['mean_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.var_ = kwargs['var_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(np.log(-x_scaled))
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = -np.exp(self.skl_scaler.inverse_transform(x_scaled))
        return x


class MinMaxScaler(Scaler):
    """
    Transform features by scaling each
    feature to the (0, 1) range.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.MinMaxScaler()
        self.skl_scaler.min_ = kwargs['min_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.data_min_ = kwargs['data_min_']
        self.skl_scaler.data_max_ = kwargs['data_max_']
        self.skl_scaler.data_range_ = kwargs['data_range_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.skl_scaler.inverse_transform(x_scaled)
        return x


class MinMaxCommonScaler(Scaler):
    """
    Transform features by scaling each
    feature to the (0, 1) range common to all.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.glob_min_ = kwargs['glob_min_']
        self.glob_max_ = kwargs['glob_max_']
        return

    def transform(self, x, replace_infinity=True):
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
        if self.glob_min_ == 0. and self.glob_max_ == 0.:
            x = x_scaled
        elif self.glob_min_ == self.glob_max_:
            x = x_scaled * self.glob_max_
        else:
            x = x_scaled * (self.glob_max_ - self.glob_min_) + self.glob_min_
        return x


class MinMaxPlus1Scaler(Scaler):
    """
    Transform features by scaling each
    feature to the (1, 2) range.
    This can be useful to avoid zeros.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.MinMaxScaler()
        self.skl_scaler.min_ = kwargs['min_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.data_min_ = kwargs['data_min_']
        self.skl_scaler.data_max_ = kwargs['data_max_']
        self.skl_scaler.data_range_ = kwargs['data_range_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled) + 1.
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.skl_scaler.inverse_transform(x_scaled - 1.)
        return x


class ExpMinMaxScaler(Scaler):
    """
    Transform features by scaling each
    feature to the (0, 1) range and then
    takes the exponential of the result.
    """

    def __init__(self, **kwargs):
        Scaler.__init__(self, **kwargs)
        self.skl_scaler = skl_pre.MinMaxScaler()
        self.skl_scaler.min_ = kwargs['min_']
        self.skl_scaler.scale_ = kwargs['scale_']
        self.skl_scaler.data_min_ = kwargs['data_min_']
        self.skl_scaler.data_max_ = kwargs['data_max_']
        self.skl_scaler.data_range_ = kwargs['data_range_']
        self.skl_scaler.n_samples_seen_ = kwargs['n_samples_seen_']
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        x_scaled = np.exp(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = np.log(x_scaled)
        x = self.skl_scaler.inverse_transform(x)
        return x
