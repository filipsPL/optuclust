import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs

from optuclust import Optimizer


# Fixture to generate a synthetic dataset
@pytest.fixture
def data():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    return X


def _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True):
    """Helper to verify shape and type of labels, centroids, medoids, modes."""
    n_samples, n_features = data.shape

    assert optimizer.labels_ is not None
    assert optimizer.labels_.shape == (n_samples,)

    # Count non-noise clusters
    non_noise = set(optimizer.labels_) - {-1}
    n_clusters = len(non_noise)
    assert n_clusters >= 1

    if expect_cluster_centers:
        assert optimizer.cluster_centers_ is not None
        assert optimizer.cluster_centers_.shape[1] == n_features
    else:
        assert optimizer.cluster_centers_ is None

    assert optimizer.centroids_ is not None
    assert optimizer.centroids_.shape == (n_clusters, n_features)

    assert optimizer.medoids_ is not None
    assert optimizer.medoids_.shape == (n_clusters, n_features)

    assert optimizer.modes_ is not None
    assert optimizer.modes_.shape == (n_clusters, n_features)


# Test for KMeans algorithm
def test_kmeans(data):
    optimizer = Optimizer(algorithm="kmeans", n_trials=10, verbose=False)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)


# Test for KMedoids algorithm
def test_kmedoids(data):
    optimizer = Optimizer(algorithm="kmedoids", n_trials=10, verbose=False)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)


# Test for MiniBatchKMeans algorithm
def test_minibatchkmeans(data):
    optimizer = Optimizer(algorithm="minibatchkmeans", n_trials=10, verbose=False)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)


# Test for DBSCAN algorithm (should not provide cluster centers)
def test_dbscan(data):
    optimizer = Optimizer(algorithm="dbscan", n_trials=10, verbose=False)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=False)


# Test for MeanShift algorithm
def test_meanshift(data):
    optimizer = Optimizer(algorithm="meanshift", n_trials=10, verbose=False)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)


# Test for HDBSCAN algorithm
def test_hdbscan(data):
    optimizer = Optimizer(algorithm="hdbscan", n_trials=10, verbose=False)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=False)


# Test for SOM algorithm
def test_som(data):
    optimizer = Optimizer(algorithm="som", n_trials=10, verbose=False)
    optimizer.fit(data)

    n_samples, n_features = data.shape
    assert optimizer.labels_ is not None
    assert optimizer.labels_.shape == (n_samples,)
    assert optimizer.centroids_ is not None
    assert optimizer.medoids_ is not None
    assert optimizer.modes_ is not None


# Test overall optimization timeout
def test_kmeans_timeout(data):
    from time import time

    optimizer = Optimizer(algorithm="kmeans", timeout=3, n_trials=1000, verbose=False)
    start_time = time()
    optimizer.fit(data)
    elapsed_time = time() - start_time

    assert elapsed_time <= 5.0, "Optimizer did not respect the timeout"


# Test per-trial timeout (should prune the sleeping trial)
def test_trial_timeout1(data):
    optimizer = Optimizer(algorithm="sleep", trial_timeout=1, n_trials=2)
    optimizer.fit(data)
    # All trials should be pruned, so no valid model
    assert optimizer.model_ is None


# Test per-trial timeout (timeout > sleep, so trials succeed)
def test_trial_timeout10(data):
    optimizer = Optimizer(algorithm="sleep", trial_timeout=10, n_trials=2)
    optimizer.fit(data)
    _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)


# Test for storage and resume
def test_storage_and_resume(data):
    storage_path = "test-storage+resume.db"
    storage_uri = f"sqlite:///{storage_path}"

    import os

    try:
        optimizer = Optimizer(
            algorithm="kmeans", n_trials=10, verbose=False, storage=storage_uri
        )
        optimizer.fit(data)
        _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)

        # Run again to test resumption
        optimizer = Optimizer(
            algorithm="kmeans", n_trials=10, verbose=False, storage=storage_uri
        )
        optimizer.fit(data)
        _assert_cluster_descriptors(optimizer, data, expect_cluster_centers=True)

    finally:
        if os.path.exists(storage_path):
            os.remove(storage_path)


def test_invalid_algorithm(data):
    with pytest.raises(ValueError, match="Algorithm must be one of"):
        Optimizer(algorithm="kmeans_dupa", n_trials=10, verbose=False)


def test_invalid_scoring(data):
    with pytest.raises(ValueError, match="Scoring must be one of"):
        Optimizer(algorithm="kmeans", scoring="filips_score", n_trials=10, verbose=False)


def test_not_fitted_error():
    optimizer = Optimizer(algorithm="kmeans", n_trials=10, verbose=False)
    with pytest.raises(NotFittedError):
        _ = optimizer.cluster_centers_


def test_predict_unsupported_algorithm(data):
    optimizer = Optimizer(algorithm="dbscan", n_trials=10, verbose=False)
    optimizer.fit(data)
    with pytest.raises(TypeError, match="does not support predict"):
        optimizer.predict(data)
