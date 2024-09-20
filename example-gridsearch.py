from sklearn.datasets import make_blobs
from optuclust import ClustGridSearch  # Import the ClustGridSearch class

# Generate a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Initialize the ClustGridSearch with 'full' mode
grid_search = ClustGridSearch(mode="fast", scoring="silhouette_score", verbose=True)

# Fit the clustering search
grid_search.fit(X)


# Get all results

print("--------- Search summary ----------")

results = grid_search.get_results()
for result in results:
    print(result)

print("--------- The best method ----------")


# Get the best estimator, score, algorithm name, and parameters
print("Best Clustering Method:", grid_search.best_algorithm_name)
print("Best Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)


best_method = grid_search.best_estimator_


# Get the best estimator and the best score
print("Best Clustering Method:", grid_search.best_estimator_)
print("Best Score:", grid_search.best_score_)




# Print the cluster labels, centroids, and medoids (if available)
if best_method is not None:
    print("\nBest Method Details:")

    print("Best method name:", best_method.best_algorithm_name)
    
    # Print cluster labels
    print("Cluster Labels:", best_method.labels_[:10])  # Print first 10 labels
    
    # Print centroids (if available)
    if hasattr(best_method, 'centroids_') and best_method.centroids_ is not None:
        print("Centroids:", best_method.centroids_)
    else:
        print("Centroids: Not available for this method.")
    
    # Print medoids (if available)
    if hasattr(best_method, 'medoids_') and best_method.medoids_ is not None:
        print("Medoids:", best_method.medoids_)
    else:
        print("Medoids: Not available for this method.")
