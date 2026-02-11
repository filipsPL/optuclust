# optuclust

**optuclust** is a Python module for optimizing clustering algorithms using the [Optuna](https://optuna.org/) framework. It provides a scikit-learn compatible API with support for a variety of clustering methods and offers additional capabilities such as the calculation of centroids, medoids, and modes for clusters.

[![Python manual install](https://github.com/filipsPL/optuclust/actions/workflows/python-package.yml/badge.svg)](https://github.com/filipsPL/optuclust/actions/workflows/python-package.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18608559.svg)](https://doi.org/10.5281/zenodo.18608559)

## Features

- **Parameter Optimization:** Optimize clustering parameters for various algorithms using **Optuna**.
- **Supported Clustering Methods:**
  - Algorithms from scikit-learn, such as KMeans, DBSCAN, and Agglomerative Clustering.
  - Advanced methods like HDBSCAN, Self-Organizing Maps (SOM), and kMedoids.
- **Metrics and Scoring:**
  - `silhouette_score`
  - `calinski_harabasz_score`
  - `davies_bouldin_score` (automatically minimized)
  - Noise points (label=-1) are filtered out before score computation for density-based algorithms.
- **Clustering Insights:** Provides centroids (arithmetic mean), medoids (Euclidean), and modes (KDE with Scott's bandwidth) for clusters, even if the algorithm does not natively support these features. All descriptors are computed eagerly during `fit()` and work in any number of dimensions.
- **Scikit-learn Compatible:** Inherits from `BaseEstimator` and `ClusterMixin`. Works with `clone()`, `check_is_fitted()`, and scikit-learn pipelines.
- **ClustGridSearch Class:** A utility to test all clustering algorithms and identify the best one.
- **Timeout Management:** Separate timeouts for optimization runs (`timeout`) and individual trials (`trial_timeout`).
- **Storage and Resume:** Store optimization results in a SQLite database for future analysis, and resume the optimization process later.

## Installation

### The easiest method - with `pip`

`pip install optuclust`

### From repository

1. Clone this repository:

   ```bash
   git clone git@github.com:filipsPL/optuclust.git
   ```

2. Navigate to the cloned directory and install the required dependencies:

   ```bash
   cd optuclust
   pip install -r requirements.txt
   ```

3. Install **optuclust**:

   ```bash
   python setup.py install
   ```

**Requires:** Python >= 3.8, scikit-learn >= 1.1

## Usage

### 1. Optimizing a Clustering Algorithm

```python
from optuclust import Optimizer
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Instantiate and fit the optimizer for KMeans
optimizer = Optimizer(algorithm="kmeans", n_trials=50, scoring="silhouette_score", verbose=True)
optimizer.fit(X)

# Access cluster details
print("Cluster Labels:", optimizer.labels_)
print("Centroids:", optimizer.centroids_)
print("Medoids:", optimizer.medoids_)
print("Modes:", optimizer.modes_)
print("Cluster Centers (native):", optimizer.cluster_centers_)
```

### 2. ClustGridSearch

```python
from optuclust import ClustGridSearch
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Initialize ClustGridSearch to test all algorithms
grid_search = ClustGridSearch(mode="full", scoring="silhouette_score", verbose=True)

# Fit and get the best method
grid_search.fit(X)
print("Best Algorithm:", grid_search.best_estimator_.algorithm)
print("Best Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)
```

### 3. Benchmark Example

To benchmark different clustering algorithms, you can use the provided example script:

```bash
python example-loop.py
```

The benchmark will evaluate different clustering methods on various datasets and save the performance metrics and plots.

## Supported Algorithms

```python
algorithms = [
    'kmeans', 'kmedoids', 'minibatchkmeans', 'dbscan', 'agglomerativeclustering',
    'meanshift', 'spectralclustering', 'gaussianmixture', 'hdbscan',
    'affinitypropagation', 'birch', 'optics', 'som'
]
```

**Note:** Not all algorithms support `predict()` on new data. Algorithms with inductive prediction: `kmeans`, `minibatchkmeans`, `meanshift`, `birch`, `gaussianmixture`, `kmedoids`, `som`. Calling `predict()` on other algorithms (e.g. `dbscan`, `hdbscan`) will raise a `TypeError`.

## Parameters

### Optimizer Class

- **algorithm:** The clustering algorithm to optimize. Options include those listed in Supported Algorithms.
- **n_trials:** Number of Optuna trials for optimization. Default is 50.
- **scoring:** The metric to optimize. Options are `silhouette_score`, `calinski_harabasz_score`, and `davies_bouldin_score`.
- **verbose:** Enable additional logging if set to `True`. Can also be an `int` to set Optuna's verbosity level directly.
- **show_progress_bar:** Display a progress bar during optimization. Default is `True`.
- **timeout:** Maximum duration (in seconds) for all trials in the optimization process.
- **trial_timeout:** Maximum duration (in seconds) for each individual trial (Unix only, uses `SIGALRM`).
- **storage:** Optuna storage URI, e.g. `sqlite:///optimization.db`. When provided, enables resuming a previous optimization run.
- **logfile:** Reserved for future use.

### Fitted Attributes

After calling `fit(X)`:

- **labels\_:** Cluster labels for each sample.
- **best\_params\_:** Dictionary of the best hyperparameters found.
- **model\_:** The fitted clustering model with the best parameters.
- **study\_:** The Optuna `Study` object with full trial history.
- **centroids\_:** Arithmetic mean of each cluster (excludes noise points).
- **medoids\_:** Most central data point in each cluster (Euclidean distance).
- **modes\_:** Highest density point in each cluster (KDE with Scott's rule bandwidth).
- **cluster\_centers\_:** Native cluster centers from the model (if available), otherwise `None`.

### ClustGridSearch Class

- **mode:**
  - `full`: Test all algorithms.
  - `fast`: Test a subset of algorithms (`kmeans` and `hdbscan`).
- **n_trials:** Number of Optuna trials for each algorithm. Default is 20.
- **scoring:** Metric to select the best clustering algorithm. Options are `silhouette_score`, `calinski_harabasz_score`, and `davies_bouldin_score`.
- **verbose:** Enable detailed logging if set to `True`.
- **show_progress_bar:** Display a progress bar for each algorithm.

## Running Tests

We use **pytest** for testing. To run tests, simply run:

```bash
pytest -v
```
