from optuclust import ClustGridSearch  # Import the ClustGridSearch class

import numpy as np
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=2024)

# Create ClustGridSearch instance with 'full' mode
clust_search = ClustGridSearch(mode='full', n_trials=10, scoring='silhouette_score', verbose=True)

# Fit the ClustGridSearch to the data
clust_search.fit(X)

print("=" * 50)

results = clust_search.cv_results_

# Loop through the dictionary and print results in a human-readable format
print("Clustering Results:\n")
for i in range(len(results['algorithm'])):
    print(f"Algorithm: {results['algorithm'][i]}")
    print(f"Mean Test Score: {results['mean_test_score'][i]:.4f}")
    print("Parameters:")
    for param, value in results['params'][i].items():
        print(f"  {param}: {value}")
    print(f"Model: {results['model'][i]}")
    print("-" * 50)  # Separator line for readability

print("=" * 50)
# Print the summary and parameters of the best algorithm
print(f"\nBest Algorithm: {clust_search.best_estimator_.algorithm}")
print(f"Best Score: {clust_search.best_score_}")
print(f"Best Parameters: {clust_search.best_params_}")
print("=" * 50)

# Get labels, medoids, and centroids
labels = clust_search.labels_
medoids = clust_search.medoids_
centroids = clust_search.centroids_
modes = clust_search.modes_
centers = clust_search.cluster_centers_

print("\nLabels:")
print(labels)
print("\nMedoids:")
print(medoids)
print("\nCentroids:")
print(centroids)
print("\nModes:")
print(modes)
print("\nCluster centers:")
print(centers)
