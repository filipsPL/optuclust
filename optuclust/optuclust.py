import optuna
import numpy as np
from sklearn.neighbors import KernelDensity
import time

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering, AffinityPropagation,
                             Birch, OPTICS)
from sklearn.mixture import GaussianMixture

import hdbscan
from kmedoids import KMedoids
from sklearn_som.som import SOM


class Optimizer(BaseEstimator, ClusterMixin):

    def __init__(self, algorithm, n_trials=50, scoring='silhouette_score', verbose=False, show_progress_bar=True):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.scoring = scoring
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.best_params_ = None
        self.study = None
        self.model = None
        self.labels_ = None
        self.X_ = None  # Store X after fitting

        # Set Optuna logging verbosity based on 'verbose' parameter
        if isinstance(verbose, bool):
            optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
        elif isinstance(verbose, int):
            optuna.logging.set_verbosity(verbose)

    def fit(self, X, y=None):
        self.X_ = X  # Store X for later use

        def objective(trial):
            model = self._suggest_model(trial, X)
            model.fit(X)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)

            # Scoring logic, pass the trial object for pruning
            score = self._compute_score(X, labels)
            return score

        # Determine direction of optimization
        direction = 'maximize'
        if self.scoring == 'davies_bouldin_score':
            direction = 'minimize'

        self.study = optuna.create_study(direction=direction)
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.show_progress_bar)
        self.best_params_ = self.study.best_params
        self.model = self._get_best_model(X)
        self.model.fit(X)

        # Store labels_
        self.labels_ = self.model.labels_ if hasattr(self.model, 'labels_') else self.model.predict(X)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        return self.model.predict(X)

    def _compute_score(self, X, labels):
        # Prune if there is only one cluster
        if len(set(labels)) <= 1:
            raise optuna.TrialPruned("Only one cluster found, pruning this trial.")

        if self.scoring == 'silhouette_score':
            score = silhouette_score(X, labels)
        elif self.scoring == 'calinski_harabasz_score':
            score = calinski_harabasz_score(X, labels)
        elif self.scoring == 'davies_bouldin_score':
            score = davies_bouldin_score(X, labels)  # For minimization
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring}")
        return score

    def _suggest_model(self, trial, X):
        # Implement the parameter suggestions for each algorithm here
        # For brevity, only a couple of algorithms are shown
        # KMeans
        if self.algorithm == 'kmeans':
            n_clusters = trial.suggest_int('n_clusters', 2, 50)
            max_iter = trial.suggest_int('max_iter', 100, 500)
            tol = trial.suggest_float('tol', 1e-6, 1e-2)
            return KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, n_init="auto")  # n_init="auto"

        # MiniBatchKMeans
        elif self.algorithm == 'minibatchkmeans':
            n_clusters = trial.suggest_int('n_clusters', 2, 50)
            batch_size = trial.suggest_int('batch_size', 10, 200)
            max_iter = trial.suggest_int('max_iter', 100, 500)
            tol = trial.suggest_float('tol', 1e-6, 1e-2)
            return MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter, tol=tol, n_init="auto")  # n_init="auto"

        # DBSCAN
        elif self.algorithm == 'dbscan':
            eps = trial.suggest_float('eps', 0.1, 10.0)
            min_samples = trial.suggest_int('min_samples', 2, 10)
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])

            if metric == 'minkowski':
                p = trial.suggest_int('p', 1, 5)  # Add power parameter for Minkowski distance
                return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p)
            else:
                return DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        # MeanShift
        elif self.algorithm == 'meanshift':
            bandwidth = trial.suggest_float('bandwidth', 0.1, 10.0)
            bin_seeding = trial.suggest_categorical('bin_seeding', [True, False])
            return MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)

        # AgglomerativeClustering
        elif self.algorithm == 'agglomerativeclustering':
            n_clusters = trial.suggest_int('n_clusters', 2, 50)
            linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
            return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        # SpectralClustering
        elif self.algorithm == 'spectralclustering':
            n_clusters = trial.suggest_int('n_clusters', 2, 50)
            n_neighbors = trial.suggest_int('n_neighbors', 2, 20)
            eigen_tol = trial.suggest_float('eigen_tol', 1e-6, 1e-2)
            return SpectralClustering(n_clusters=n_clusters, n_neighbors=n_neighbors, eigen_tol=eigen_tol)

        # AffinityPropagation
        elif self.algorithm == 'affinitypropagation':
            damping = trial.suggest_float('damping', 0.5, 0.99)
            convergence_iter = trial.suggest_int('convergence_iter', 10, 200)
            return AffinityPropagation(damping=damping, convergence_iter=convergence_iter)

        # Birch
        elif self.algorithm == 'birch':
            n_clusters = trial.suggest_int('n_clusters', 2, 50)
            threshold = trial.suggest_float('threshold', 0.1, 1.0)
            branching_factor = trial.suggest_int('branching_factor', 20, 100)
            return Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)

        # OPTICS
        elif self.algorithm == 'optics':
            eps = trial.suggest_float('eps', 0.1, 10.0)
            min_samples = trial.suggest_int('min_samples', 2, 10)
            cluster_method = trial.suggest_categorical('cluster_method', ['xi', 'dbscan'])
            return OPTICS(eps=eps, min_samples=min_samples, cluster_method=cluster_method)

        # GaussianMixture
        elif self.algorithm == 'gaussianmixture':
            n_components = trial.suggest_int('n_components', 2, 10)
            covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
            return GaussianMixture(n_components=n_components, covariance_type=covariance_type)

        # HDBSCAN
        elif self.algorithm == 'hdbscan':
            min_cluster_size = trial.suggest_int('min_cluster_size', 2, 50)
            min_samples = trial.suggest_int('min_samples', 1, 10)

            cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0, 1)
            allow_single_cluster = trial.suggest_categorical('allow_single_cluster', [True, False])
            # metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
            return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                   min_samples=min_samples,
                                   cluster_selection_epsilon=cluster_selection_epsilon,
                                   allow_single_cluster=allow_single_cluster)

        elif self.algorithm == 'kmedoids':
            n_clusters = trial.suggest_int('n_clusters', 2, 50)
            method = trial.suggest_categorical('method', ['fasterpam', "pam", "alternate", "fastermsc", "fastmsc", "pamsil", "pammedsil"])
            return KMedoids(n_clusters=n_clusters, method=method, metric="euclidean")

        elif self.algorithm == 'som':
            m = trial.suggest_int('m', 2, 20)  # Grid height
            n = trial.suggest_int('n', 2, 20)  # Grid width

            # Initialize SOM
            som = SOM(m=m, n=n, dim=X.shape[1])

            # Train SOM
            som.fit(X)

            # Generate labels (assign each datapoint to its predicted cluster)
            labels = som.predict(X)
            self._labels = labels  # Use internal variable instead of labels_ property
            return som  # Return the trained SOM model

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _get_best_model(self, X):
        # Recreate the best model with optimized parameters
        trial = optuna.trial.FixedTrial(self.best_params_)
        return self._suggest_model(trial, X)

    @property
    def cluster_centers_(self):
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        else:
            return None

    @property
    def centroids_(self):
        """
        Calculate centroids for clusters, even if cluster_centers_ is not provided by the model.
        """
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        elif self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            centroids = []
            for label in unique_labels:
                if label == -1:  # Handle noise in clustering algorithms like DBSCAN
                    continue
                cluster_points = self.X_[self.labels_ == label]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)
            return np.array(centroids)
        else:
            return None

    @property
    def medoids_(self):
        """
        Calculate medoids for clusters, even if medoids are not provided by the model.
        """
        if isinstance(self.model, KMedoids):
            return self.model.cluster_centers_
        elif self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            medoids = []
            for label in unique_labels:
                if label == -1:  # Handle noise in clustering algorithms like DBSCAN
                    continue
                cluster_points = self.X_[self.labels_ == label]
                if len(cluster_points) == 0:
                    continue
                # Compute pairwise distances
                distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :]), axis=2)
                # Find the index of the point with minimal total distance to other points
                medoid_index = np.argmin(np.sum(distances, axis=1))
                medoids.append(cluster_points[medoid_index])
            return np.array(medoids)
        else:
            return None

    @property
    def modes_(self):
        """
        Calculate modes for clusters using Kernel Density Estimation (KDE).
        """
        if self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            modes = []
            for label in unique_labels:
                if label == -1:  # Handle noise in clustering algorithms like DBSCAN
                    continue
                cluster_points = self.X_[self.labels_ == label]
                if len(cluster_points) == 0:
                    continue
                # Fit Kernel Density Estimate for the cluster
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(cluster_points)
                # Generate a grid of points to find the highest density
                grid = np.linspace(cluster_points.min(axis=0), cluster_points.max(axis=0), 100)
                grid_points = np.meshgrid(*[grid[:, i] for i in range(cluster_points.shape[1])])
                grid_points = np.stack([gp.ravel() for gp in grid_points], axis=-1)
                densities = kde.score_samples(grid_points)
                # Find the mode (highest density point)
                mode_index = np.argmax(densities)
                mode = grid_points[mode_index]
                modes.append(mode)
            return np.array(modes)
        else:
            return None


class ClustGridSearch(BaseEstimator, ClusterMixin):

    def __init__(self, mode='full', n_trials=20, scoring='silhouette_score', verbose=False, show_progress_bar=True):
        """
        Initialize the ClustGridSearch.

        :param mode: 'full' to test all algorithms, 'fast' to test a subset (kmeans and hdbscan).
        :param n_trials: Number of trials for each algorithm's hyperparameter optimization.
        :param scoring: The metric used to select the best clustering (default: 'silhouette_score').
        :param verbose: Whether to print additional information during the search.
        """
        self.mode = mode
        self.n_trials = n_trials
        self.scoring = scoring
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar

        self.cv_results_ = {}
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None

        # Define algorithms to test based on mode
        if self.mode == "full":
            self.algorithms = [
                'kmeans', 'kmedoids', 'minibatchkmeans', 'dbscan', 'agglomerativeclustering', 'meanshift', 'spectralclustering',
                'affinitypropagation', 'birch', 'optics', 'gaussianmixture', 'hdbscan'
            ]
        elif self.mode == "fast":
            self.algorithms = ['kmeans', 'hdbscan']
        else:
            raise ValueError("Invalid mode. Use 'full' or 'fast'.")

    def fit(self, X, y=None):
        """
        Run clustering for all selected algorithms and return the best one based on the chosen scoring.

        :param X: Input data for clustering.
        """
        results = []
        for idx, algorithm in enumerate(self.algorithms):
            if self.verbose:
                print(f"\nTesting algorithm: {algorithm}")

            optimizer = Optimizer(algorithm=algorithm,
                                  n_trials=self.n_trials,
                                  scoring=self.scoring,
                                  verbose=self.verbose,
                                  show_progress_bar=self.show_progress_bar)

            try:
                optimizer.fit(X)
                score = optimizer.study.best_value
                results.append({'algorithm': algorithm, 'mean_test_score': score, 'params': optimizer.best_params_, 'model': optimizer})
            except Exception as e:
                if self.verbose:
                    print(f"Error for algorithm {algorithm}: {e}")

        if not results:
            raise ValueError("No algorithms produced valid results.")

        # Convert results to cv_results_ dict similar to scikit-learn's GridSearchCV
        self.cv_results_ = {
            'algorithm': [res['algorithm'] for res in results],
            'mean_test_score': [res['mean_test_score'] for res in results],
            'params': [res['params'] for res in results],
            'model': [res['model'] for res in results],
        }

        # Determine best score and estimator
        reverse = self.scoring != 'davies_bouldin_score'
        scores = self.cv_results_['mean_test_score']
        if reverse:
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmin(scores)
        self.best_index_ = best_idx
        self.best_score_ = scores[best_idx]
        self.best_params_ = self.cv_results_['params'][best_idx]
        self.best_estimator_ = self.cv_results_['model'][best_idx]
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.best_estimator_.labels_

    @property
    def labels_(self):
        return self.best_estimator_.labels_

    @property
    def cluster_centers_(self):
        return self.best_estimator_.cluster_centers_

    @property
    def centroids_(self):
        return self.best_estimator_.centroids_

    @property
    def medoids_(self):
        return self.best_estimator_.medoids_

    @property
    def modes_(self):
        return self.best_estimator_.modes_
