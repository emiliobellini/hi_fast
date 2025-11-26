"""
.. module:: pca

:Synopsis: Apply PCA.
:Author: Emilio Bellini

"""

import sklearn.decomposition as skl_dec


class PCA(object):
    """
    Apply PCA.
    It is used to apply the PCA to either x or y.
    This main class has two main methods:
    - transform: transform data using PCA
    - inverse_transform: transform back data.
    """

    def __init__(self, **kwargs):
        self.n_components = kwargs['n_components_']
        self.pca = skl_dec.PCA(n_components=self.n_components)
        # Store PCA attributes
        self.pca.n_components_ = kwargs['n_components_']
        self.pca.components_ = kwargs['components_']
        self.pca.mean_ = kwargs['mean_']
        self.pca.explained_variance_ = kwargs['explained_variance_']
        self.pca.explained_variance_ratio_ =\
            kwargs['explained_variance_ratio_']
        self.pca.singular_values_ = kwargs['singular_values_']
        self.pca.n_samples_ = kwargs['n_samples_']
        self.pca.n_features_in_ = kwargs['n_features_in_']
        self.pca.noise_variance_ = kwargs['noise_variance_']
        return

    def transform(self, x):
        """
        Transform an array x into the corresponding
        x_pca, using results from the fit method.
        """
        x_pca = self.pca.transform(x)
        return x_pca

    def inverse_transform(self, x_pca):
        """
        Transform back an array x_pca into
        the corresponding x, using results from
        the fit method.
        """
        x = self.pca.inverse_transform(x_pca)
        return x
