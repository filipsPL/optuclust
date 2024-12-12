from optuclust import Optimizer
from sklearn.datasets import make_blobs
from time import time

def data():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    return X


optimizer = Optimizer(algorithm="sleep", trial_timeout=1, n_trials=2)
start_time = time()
optimizer.fit(data)
elapsed_time = time() - start_time
print(f"Elapsed time: {elapsed_time}s")
