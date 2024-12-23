from optuclust import Optimizer
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=1410)

print("The first round of optimization")
optimizer = Optimizer(algorithm="dbscan", n_trials=20, storage="sqlite:///example-storage+resume.db")
optimizer.fit(X)


print("\n\nThe second round of optimization - start from the previous state")
optimizer = Optimizer(algorithm="dbscan", n_trials=20, storage="sqlite:///example-storage+resume.db")
optimizer.fit(X)
