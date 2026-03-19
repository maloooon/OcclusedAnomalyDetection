import numpy as np

def malahanobis_distance_diagonal(x: np.ndarray, v: np.ndarray, cov_diag: np.ndarray) -> float:
    """
    Computes the Mahalanobis distance using only the diagonal of the covariance matrix.

    Parameters:
    x (array-like): A single sample or an array of samples with shape (n_samples, n_features).
    mean (array-like): The mean vector of the distribution with shape (n_features,).
    diag_cov (array-like): The diagonal of the covariance matrix with shape (n_features,).

    Returns:
    distances (float or ndarray): The Mahalanobis distance(s).
    """

    x = np.asarray(x)
    mean = np.asarray(v)
    diag_cov = np.asarray(cov_diag)

    diff = x - mean
    normalized_diff = diff / np.sqrt(diag_cov)
    distances = np.sqrt(np.sum(normalized_diff ** 2, axis=-1))

    if distances.ndim == 0:
        return distances.item()

    return distances