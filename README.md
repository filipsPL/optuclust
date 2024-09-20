# optuclust

**optuclust** is a Python module for optimizing clustering algorithms using the [Optuna](https://optuna.org/) framework. It provides support for a variety of clustering methods and offers additional capabilities such as the calculation of centroids, medoids, and modes for clusters.

## Features

- Optimize clustering parameters for various algorithms using **Optuna**.
- Supported clustering methods:
  - from scikit-learn zoo, and
  - HDBSCAN
  - SOM
  - kMedoids
- Provides centroids, medoids, and modes for clusters, even if the algorithm does not natively support these features.
- Scoring functions: `silhouette_score`, `calinski_harabasz_score`, `-1 * davies_bouldin_score` (all to be maximized)
- `ClustGridSearch` class to test all clustering algorithms and find the best one
  
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
print("Best Algorithm:", grid_search.best_algorithm_name)
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
    'som', 'kmeans', 'kmedoids', 'minibatchkmeans', 'dbscan', 'agglomerativeclustering',
    'meanshift', 'spectralclustering', 'gaussianmixture', 'hdbscan',
    'affinitypropagation', 'birch', 'optics', 
]
```

## Running Tests

We use **pytest** for testing. To run tests, simply run:

```bash
pytest
```
