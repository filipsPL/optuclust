import pytest
import numpy as np
from optuclust import Optimizer
from sklearn.datasets import make_blobs


# Fixture to generate a synthetic dataset
@pytest.fixture
def data():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    return X


# Test for KMeans algorithm
def test_kmeans(data):
    optimizer = Optimizer(algorithm="kmeans", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.cluster_centers_ is not None, "KMeans should provide cluster centers"
    assert optimizer.medoids_ is not None, "Medoids should be calculated for KMeans"
    assert optimizer.modes_ is not None, "Modes should be calculated for KMeans"
    assert optimizer.centroids_ is not None


# Test for KMedoids algorithm
def test_kmedoids(data):
    optimizer = Optimizer(algorithm="kmedoids", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.cluster_centers_ is not None, "KMedoids should provide cluster centers (as medoids)"
    assert optimizer.medoids_ is not None, "Medoids should be calculated for KMedoids"
    assert optimizer.modes_ is not None, "Modes should be calculated for KMedoids"
    assert optimizer.centroids_ is not None


# Test for MiniBatchKMeans algorithm
def test_minibatchkmeans(data):
    optimizer = Optimizer(algorithm="minibatchkmeans", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.cluster_centers_ is not None, "MiniBatchKMeans should provide cluster centers"
    assert optimizer.medoids_ is not None, "Medoids should be calculated for MiniBatchKMeans"
    assert optimizer.modes_ is not None, "Modes should be calculated for MiniBatchKMeans"
    assert optimizer.centroids_ is not None


# Test for DBSCAN algorithm (should not provide cluster centers)
def test_dbscan(data):
    optimizer = Optimizer(algorithm="dbscan", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.cluster_centers_ is None, "DBSCAN should not provide cluster centers"
    assert optimizer.medoids_ is not None, "Medoids should be calculated for DBSCAN"
    assert optimizer.modes_ is not None, "Modes should be calculated for DBSCAN"
    assert optimizer.centroids_ is not None


# Test for MeanShift algorithm (which calculates cluster centers)
def test_meanshift(data):
    optimizer = Optimizer(algorithm="meanshift", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.cluster_centers_ is not None, "MeanShift should provide cluster centers (modes)"
    assert optimizer.medoids_ is not None, "Medoids should be calculated for MeanShift"
    assert optimizer.modes_ is not None, "Modes should be calculated for MeanShift"
    assert optimizer.centroids_ is not None


# Test for HDBSCAN algorithm
def test_hdbscan(data):
    optimizer = Optimizer(algorithm="hdbscan", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.cluster_centers_ is None, "HDBSCAN should not provide cluster centers"
    assert optimizer.medoids_ is not None, "Medoids should be calculated for HDBSCAN"
    assert optimizer.modes_ is not None, "Modes should be calculated for HDBSCAN"
    assert optimizer.centroids_ is not None



# Test for SOM algorithm
def test_som(data):
    optimizer = Optimizer(algorithm="som", n_trials=10, verbose=False)
    optimizer.fit(data)

    assert optimizer.medoids_ is not None, "Medoids should be calculated for SOM"
    assert optimizer.modes_ is not None, "Modes should be calculated for SOM"
    assert optimizer.centroids_ is not None


# Test for KMeans algorithm
def test_kmeans_timeout(data):
    from time import time
    optimizer = Optimizer(algorithm="kmeans", timeout=3, n_trials=1000, verbose=False)
    start_time = time()
    optimizer.fit(data)
    elapsed_time = time() - start_time
    print(f"Elapsed time: {elapsed_time}s")

    # Ensure the optimizer respects the timeout
    assert elapsed_time <= 3.5, "Optimizer did not respect the timeout"
    # assert optimizer.cluster_centers_ is not None, "KMeans should provide cluster centers"

# Test for timeout - fake algorithm
def test_trial_timeout1(data):
    optimizer = Optimizer(algorithm="sleep", trial_timeout=1, n_trials=2)
    optimizer.fit(data)
    # Ensure the optimizer respects the timeout
    assert optimizer.cluster_centers_ is None, "KMeans should NOT provide cluster centers"

def test_trial_timeout10(data):
    optimizer = Optimizer(algorithm="sleep", trial_timeout=10, n_trials=2)
    optimizer.fit(data)
    # Ensure the optimizer respects the timeout
    # should NOT fail

    assert optimizer.medoids_ is not None, "Medoids should be calculated for SOM"
    assert optimizer.modes_ is not None, "Modes should be calculated for SOM"
    assert optimizer.centroids_ is not None

# Test for storage and resume 
def test_storage_and_resume(data):
    storage_path = "test-storage+resume.db"
    storage_uri = f"sqlite:///{storage_path}"
    
    import os

    try:
        # Run the optimizer
        optimizer = Optimizer(algorithm="kmeans", n_trials=10, verbose=False, storage=storage_uri)
        optimizer.fit(data)

        # Assertions
        assert optimizer.cluster_centers_ is not None, "KMeans should provide cluster centers"
        assert optimizer.medoids_ is not None, "Medoids should be calculated for KMeans"
        assert optimizer.modes_ is not None, "Modes should be calculated for KMeans"
        assert optimizer.centroids_ is not None, "Centroids should be calculated for KMeans"
        
        # Run the optimizer again
        optimizer = Optimizer(algorithm="kmeans", n_trials=10, verbose=False, storage=storage_uri)
        optimizer.fit(data)

        # Assertions
        assert optimizer.cluster_centers_ is not None, "KMeans should provide cluster centers"
        assert optimizer.medoids_ is not None, "Medoids should be calculated for KMeans"
        assert optimizer.modes_ is not None, "Modes should be calculated for KMeans"
        assert optimizer.centroids_ is not None, "Centroids should be calculated for KMeans"


    finally:
        # Remove the SQLite file after the test
        if os.path.exists(storage_path):
            os.remove(storage_path)

