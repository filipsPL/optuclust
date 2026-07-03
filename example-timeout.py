import time

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from optuclust import Optimizer


def data():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    return X


# 'trial_timeout' is demonstrated here with a deliberately slow model-building
# step. This is monkeypatched in just for this example: there is no "sleep"
# algorithm in the public API, since a fake timeout-inducing algorithm is a
# testing concern, not a real clustering method.
_original_suggest_model = Optimizer._suggest_model


def _slow_suggest_model(self, trial, X):
    time.sleep(3)
    return KMeans(n_clusters=3, n_init="auto")


Optimizer._suggest_model = _slow_suggest_model

optimizer = Optimizer(algorithm="kmeans", trial_timeout=1, n_trials=2)
start_time = time.time()
optimizer.fit(data())
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time}s")

Optimizer._suggest_model = _original_suggest_model
