from optuclust import Optimizer
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=1410)

optimizer = Optimizer(algorithm="dbscan", n_trials=50, verbose=True)

optimizer.fit(X)


# Access cluster details
print("Cluster Labels:", optimizer.labels_)
print("Centroids:", optimizer.centroids_)
print("Medoids:", optimizer.medoids_)
print("Modes:", optimizer.modes_)