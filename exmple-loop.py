import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from optuclust import Optimizer

# Create a sample dataset (blobs for clustering)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)

# List of all algorithms to test
algorithms = [
    'som', 'kmeans', 'kmedoids', 'minibatchkmeans', 'dbscan', 'agglomerativeclustering', 'meanshift', 'spectralclustering',
    'affinitypropagation', 'birch', 'optics', 'gaussianmixture', 'hdbscan'
]

# Loop over each algorithm and run the Optimizer
for algorithm in algorithms:
    print(f"Running Optimizer for {algorithm}")

    # Instantiate the Optimizer
    optimizer = Optimizer(algorithm=algorithm, n_trials=140, scoring='silhouette_score', verbose=False)

    # Fit the Optimizer to the data
    try:
        optimizer.fit(X)

        # Print results
        print(f"Algorithm: {algorithm}")
        print(f"Best Parameters: {optimizer.best_params_}")
        print(f"Cluster Centers: {optimizer.cluster_centers_}")
        print(f"Cluster centroids: {optimizer.centroids_}")
        print(f"Cluster medoids: {optimizer.medoids_}")
        print(f"Labels: {optimizer.labels_[:10]}")  # Printing first 10 labels for brevity
        print("-" * 50)
    except Exception as e:
        print(f"Error running {algorithm}: {str(e)}")
        print("-" * 50)
