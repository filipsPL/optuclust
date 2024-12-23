# optuclust

**optuclust** is a Python module for optimizing clustering algorithms using the [Optuna](https://optuna.org/) framework. It provides support for a variety of clustering methods and offers additional capabilities such as the calculation of centroids, medoids, and modes for clusters.

[![Python manual install](https://github.com/filipsPL/optuclust/actions/workflows/python-package.yml/badge.svg)](https://github.com/filipsPL/optuclust/actions/workflows/python-package.yml)

## Features

- **Parameter Optimization:** Optimize clustering parameters for various algorithms using **Optuna**.
- **Supported Clustering Methods:**
  - Algorithms from scikit-learn, such as KMeans, DBSCAN, and Agglomerative Clustering.
  - Advanced methods like HDBSCAN, Self-Organizing Maps (SOM), and kMedoids.
- **Metrics and Scoring:**
  - `silhouette_score`
  - `calinski_harabasz_score`
  - `-1 * davies_bouldin_score` (all scores are maximized for consistency).
- **Clustering Insights:** Provides centroids, medoids, and modes for clusters, even if the algorithm does not natively support these features.
- **ClustGridSearch Class:** A powerful utility to test all clustering algorithms and identify the best one.
- **Timeout Management:** Separate timeouts for optimization runs (`timeout`) and individual trials (`trial_timeout`).
- **Storage and resume:** Store individual optimization results in sqlite database for future analysis, and resume optimization process later, if needed.

## Installation

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

## Parameters

### Optimizer Class

- **algorithm:** The clustering algorithm to optimize. Options include those listed in Supported Algorithms.
- **n_trials:** Number of Optuna trials for optimization. Default is 50.
- **scoring:** The metric to optimize. Options are `silhouette_score`, `calinski_harabasz_score`, and `-1 * davies_bouldin_score`.
- **verbose:** Print additional logs if set to `True`.
- **show_progress_bar:** Display a progress bar during optimization. Default is `True`.
- **timeout:** Maximum duration (in seconds) for all trials in the optimization process.
- **trial_timeout:** Maximum duration (in seconds) for each individual trial.
- **storage:** Storage object that optuna can use, eg `sqlite:///optimization.db`. This option imply attempt to resume optimization process if the storage is defined and exists.

### ClustGridSearch Class

- **mode:**
  - `full`: Test all algorithms.
  - `fast`: Test a subset of algorithms (`kmeans` and `hdbscan`).
- **n_trials:** Number of Optuna trials for each algorithm. Default is 20.
- **scoring:** Metric to select the best clustering algorithm. Options are `silhouette_score`, `calinski_harabasz_score`, and `-1 * davies_bouldin_score`.
- **verbose:** Print detailed logs if set to `True`.
- **show_progress_bar:** Display a progress bar for each algorithm.

## Running Tests

We use **pytest** for testing. To run tests, simply run:

```bash
pytest -v
```
