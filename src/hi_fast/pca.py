"""
.. module:: pca

:Synopsis: Apply PCA.
:Author: Emilio Bellini

"""

import sklearn.decomposition as skl_dec


class PCA(object):
    """Thin wrapper around ``sklearn.decomposition.PCA``.

    The class rehydrates a pre-trained PCA object from serialized
    attributes, allowing forward and inverse transformations without
    re-fitting on disk-loaded emulator data.
    """

    def __init__(self, **kwargs):
        """Restore PCA state from keyword arguments.

        Args:
            **kwargs: Serialized PCA attributes produced by scikit-learn,
                e.g., ``n_components_``, ``components_``, ``mean_``, etc.

        Raises:
            KeyError: If any required PCA attribute is missing in
                ``kwargs``.
        """
        self.pca = skl_dec.PCA(n_components=kwargs['n_components_'])
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
        """Project data into PCA space.

        Args:
            x (array-like): Samples with shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Transformed samples with shape
            ``(n_samples, n_components)``.
        """
        x_pca = self.pca.transform(x)
        return x_pca

    def inverse_transform(self, x_pca):
        """Map PCA-space data back to the original feature space.

        Args:
            x_pca (array-like): PCA coefficients with shape
                ``(n_samples, n_components)``.

        Returns:
            numpy.ndarray: Reconstructed samples in the original feature
            space.
        """
        x = self.pca.inverse_transform(x_pca)
        return x
