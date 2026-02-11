import logging
import signal
import time

import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted

import hdbscan
from kmedoids import KMedoids
from sklearn_som.som import SOM

logger = logging.getLogger("optuclust")


class Optimizer(BaseEstimator, ClusterMixin):

    VALID_ALGORITHMS = [
        "kmeans",
        "minibatchkmeans",
        "dbscan",
        "meanshift",
        "agglomerativeclustering",
        "spectralclustering",
        "affinitypropagation",
        "birch",
        "optics",
        "gaussianmixture",
        "hdbscan",
        "kmedoids",
        "sleep",
        "som",
    ]

    VALID_SCORING = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ]

    ALGORITHMS_WITH_PREDICT = {
        "kmeans",
        "minibatchkmeans",
        "meanshift",
        "birch",
        "gaussianmixture",
        "kmedoids",
        "som",
        "sleep",
    }

    SAFE_DEFAULTS = {
        "kmeans": {"n_clusters": 3, "max_iter": 300, "tol": 1e-4, "n_init": 10},
        "minibatchkmeans": {
            "n_clusters": 8,
            "batch_size": 100,
            "max_iter": 300,
            "tol": 1e-4,
            "n_init": 10,
        },
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "p": 2,
        },
        "meanshift": {"bandwidth": 2.5, "bin_seeding": True},
        "agglomerativeclustering": {"n_clusters": 3, "linkage": "ward"},
        "spectralclustering": {
            "n_clusters": 3,
            "n_neighbors": 10,
            "eigen_tol": 1e-4,
        },
        "affinitypropagation": {"damping": 0.9, "convergence_iter": 15},
        "birch": {"n_clusters": 3, "threshold": 0.5, "branching_factor": 50},
        "optics": {
            "min_samples": 5,
            "cluster_method": "xi",
        },
        "gaussianmixture": {"n_components": 3, "covariance_type": "full"},
        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 1,
            "cluster_selection_epsilon": 0.0,
            "allow_single_cluster": False,
        },
        "kmedoids": {"n_clusters": 3, "method": "pam", "metric": "euclidean"},
        "som": {
            "m": 10,
            "n": 10,
            "dim": None,
        },
    }

    def __init__(
        self,
        algorithm,
        n_trials=50,
        scoring="silhouette_score",
        verbose=False,
        show_progress_bar=True,
        timeout=None,
        trial_timeout=None,
        storage=None,
        logfile=None,
    ):
        if algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {self.VALID_ALGORITHMS}")
        if scoring not in self.VALID_SCORING:
            raise ValueError(f"Scoring must be one of {self.VALID_SCORING}")
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_trials must be a positive integer")

        self.algorithm = algorithm
        self.n_trials = n_trials
        self.scoring = scoring
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.timeout = timeout
        self.trial_timeout = trial_timeout
        self.storage = storage
        self.logfile = logfile

    def fit(self, X, y=None):
        # Configure optuna verbosity locally
        if isinstance(self.verbose, bool):
            optuna.logging.set_verbosity(
                optuna.logging.INFO if self.verbose else optuna.logging.WARNING
            )
            _show_progress_bar = self.show_progress_bar if not self.verbose else False
        elif isinstance(self.verbose, int):
            optuna.logging.set_verbosity(self.verbose)
            _show_progress_bar = self.show_progress_bar
        else:
            _show_progress_bar = self.show_progress_bar

        # Resolve storage
        storage = self.storage
        if storage is None:
            storage = optuna.storages.InMemoryStorage()

        study_name = f"study_{self.algorithm}_{self.scoring}"
        logger.info("Storage: %s, internal study name: %s", storage, study_name)

        def timeout_handler(signum, frame):
            raise TimeoutError("Objective function timed out")

        def objective(trial):
            if self.trial_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.trial_timeout))

            try:
                model = self._suggest_model(trial, X)
                model.fit(X)
                labels = (
                    model.labels_ if hasattr(model, "labels_") else model.predict(X)
                )
                score = self._compute_score(X, labels)
                return score
            except TimeoutError:
                trial.report(float("-inf"), step=0)
                raise optuna.TrialPruned("Trial pruned due to timeout")
            finally:
                if self.trial_timeout:
                    signal.alarm(0)

        # Determine direction of optimization
        direction = "maximize"
        if self.scoring == "davies_bouldin_score":
            direction = "minimize"

        self.study_ = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        try:
            n_existing = len(self.study_.trials)
            if n_existing > 0:
                logger.info(
                    "Resuming optimization from storage, starting from trial %d.",
                    n_existing,
                )
            else:
                logger.info("Starting a new optimization.")

            self.study_.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=_show_progress_bar,
                timeout=self.timeout,
            )
            self.best_params_ = self.study_.best_params
            logger.info(
                "Optimization completed. Best parameters: %s", self.best_params_
            )

            self.model_ = self._get_best_model(X)
            self.model_.fit(X)

            self.labels_ = (
                self.model_.labels_
                if hasattr(self.model_, "labels_")
                else self.model_.predict(X)
            )
            logger.info(
                "Final model fitted. Number of clusters: %d",
                len(set(self.labels_)),
            )

            # Eagerly compute cluster descriptors
            self.centroids_ = self._compute_centroids(X)
            self.medoids_ = self._compute_medoids(X)
            self.modes_ = self._compute_modes(X)

        except ValueError as e:
            if "No trials are completed yet" in str(e):
                logger.warning(
                    "All trials were pruned. No valid results were obtained."
                )
                self.best_params_ = None
                self.model_ = None
                self.labels_ = None
                self.centroids_ = None
                self.medoids_ = None
                self.modes_ = None
            else:
                logger.error("Error during optimization: %s", str(e))
                raise

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        check_is_fitted(self)
        if self.model_ is None:
            raise ValueError(
                "No valid model available. Ensure that trials completed successfully."
            )
        if self.algorithm not in self.ALGORITHMS_WITH_PREDICT:
            raise TypeError(
                f"Algorithm '{self.algorithm}' does not support predict(). "
                f"Algorithms with predict: {sorted(self.ALGORITHMS_WITH_PREDICT)}"
            )
        return self.model_.predict(X)

    def _compute_score(self, X, labels):
        # Filter out noise points for all metrics
        mask = labels != -1
        non_noise_labels = labels[mask]

        if len(set(non_noise_labels)) <= 1:
            raise optuna.TrialPruned(
                "Only one cluster found (excluding noise), pruning this trial."
            )

        X_filtered = X[mask]

        if self.scoring == "silhouette_score":
            score = silhouette_score(X_filtered, non_noise_labels)
        elif self.scoring == "calinski_harabasz_score":
            score = calinski_harabasz_score(X_filtered, non_noise_labels)
        elif self.scoring == "davies_bouldin_score":
            score = davies_bouldin_score(X_filtered, non_noise_labels)
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring}")
        return score

    def _suggest_model(self, trial, X):

        if self.algorithm == "kmeans":
            n_clusters = trial.suggest_int("n_clusters", 2, 50)
            max_iter = trial.suggest_int("max_iter", 100, 500)
            tol = trial.suggest_float("tol", 1e-6, 1e-2)
            return KMeans(
                n_clusters=n_clusters, max_iter=max_iter, tol=tol, n_init="auto"
            )

        elif self.algorithm == "minibatchkmeans":
            n_clusters = trial.suggest_int("n_clusters", 2, 50)
            batch_size = trial.suggest_int("batch_size", 10, 200)
            max_iter = trial.suggest_int("max_iter", 100, 500)
            tol = trial.suggest_float("tol", 1e-6, 1e-2)
            return MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                max_iter=max_iter,
                tol=tol,
                n_init="auto",
            )

        elif self.algorithm == "dbscan":
            eps = trial.suggest_float("eps", 0.1, 10.0)
            min_samples = trial.suggest_int("min_samples", 2, 10)
            metric = trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
            )
            if metric == "minkowski":
                p = trial.suggest_int("p", 1, 5)
                return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p)
            else:
                return DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        elif self.algorithm == "meanshift":
            bandwidth = trial.suggest_float("bandwidth", 0.1, 10.0)
            bin_seeding = trial.suggest_categorical("bin_seeding", [True, False])
            return MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)

        elif self.algorithm == "agglomerativeclustering":
            n_clusters = trial.suggest_int("n_clusters", 2, 50)
            linkage = trial.suggest_categorical(
                "linkage", ["ward", "complete", "average", "single"]
            )
            return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        elif self.algorithm == "spectralclustering":
            n_clusters = trial.suggest_int("n_clusters", 2, 50)
            n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
            eigen_tol = trial.suggest_float("eigen_tol", 1e-6, 1e-2)
            return SpectralClustering(
                n_clusters=n_clusters, n_neighbors=n_neighbors, eigen_tol=eigen_tol
            )

        elif self.algorithm == "affinitypropagation":
            damping = trial.suggest_float("damping", 0.5, 0.99)
            convergence_iter = trial.suggest_int("convergence_iter", 10, 200)
            return AffinityPropagation(
                damping=damping, convergence_iter=convergence_iter
            )

        elif self.algorithm == "birch":
            n_clusters = trial.suggest_int("n_clusters", 2, 50)
            threshold = trial.suggest_float("threshold", 0.1, 1.0)
            branching_factor = trial.suggest_int("branching_factor", 20, 100)
            return Birch(
                n_clusters=n_clusters,
                threshold=threshold,
                branching_factor=branching_factor,
            )

        elif self.algorithm == "optics":
            min_samples = trial.suggest_int("min_samples", 2, 10)
            cluster_method = trial.suggest_categorical(
                "cluster_method", ["xi", "dbscan"]
            )
            return OPTICS(
                max_eps=np.inf,
                min_samples=min_samples,
                cluster_method=cluster_method,
            )

        elif self.algorithm == "gaussianmixture":
            n_components = trial.suggest_int("n_components", 2, 10)
            covariance_type = trial.suggest_categorical(
                "covariance_type", ["full", "tied", "diag", "spherical"]
            )
            return GaussianMixture(
                n_components=n_components, covariance_type=covariance_type
            )

        elif self.algorithm == "hdbscan":
            min_cluster_size = trial.suggest_int("min_cluster_size", 2, 50)
            min_samples = trial.suggest_int("min_samples", 1, 10)
            cluster_selection_epsilon = trial.suggest_float(
                "cluster_selection_epsilon", 0, 1
            )
            allow_single_cluster = trial.suggest_categorical(
                "allow_single_cluster", [True, False]
            )
            return hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                allow_single_cluster=allow_single_cluster,
            )

        elif self.algorithm == "kmedoids":
            n_clusters = trial.suggest_int("n_clusters", 2, 50)
            method = trial.suggest_categorical(
                "method",
                [
                    "fasterpam",
                    "pam",
                    "alternate",
                    "fastermsc",
                    "fastmsc",
                    "pamsil",
                    "pammedsil",
                ],
            )
            return KMedoids(n_clusters=n_clusters, method=method, metric="euclidean")

        elif self.algorithm == "sleep":
            # Fake algorithm to induce timeout for testing
            time.sleep(3)
            return KMeans(n_clusters=3, n_init="auto")

        elif self.algorithm == "som":
            m = trial.suggest_int("m", 2, 20)
            n = trial.suggest_int("n", 2, 20)
            return SOM(m=m, n=n, dim=X.shape[1])

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _get_best_model(self, X):
        params = dict(self.best_params_)
        # Remove conditional parameters that don't apply
        if self.algorithm == "dbscan" and params.get("metric") != "minkowski":
            params.pop("p", None)
        trial = optuna.trial.FixedTrial(params)
        return self._suggest_model(trial, X)

    def _compute_centroids(self, X):
        """Compute arithmetic mean centroid for each cluster."""
        if self.labels_ is None:
            return None
        unique_labels = np.unique(self.labels_)
        centroids = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = X[self.labels_ == label]
            centroids.append(cluster_points.mean(axis=0))
        if len(centroids) == 0:
            return None
        return np.array(centroids)

    def _compute_medoids(self, X):
        """Compute medoid (point with minimum total squared Euclidean distance) for each cluster."""
        if self.labels_ is None:
            return None
        unique_labels = np.unique(self.labels_)
        medoids = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = X[self.labels_ == label]
            if len(cluster_points) == 0:
                continue
            # Squared Euclidean pairwise distances
            distances = np.sum(
                (cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :]) ** 2,
                axis=2,
            )
            medoid_index = np.argmin(np.sum(distances, axis=1))
            medoids.append(cluster_points[medoid_index])
        if len(medoids) == 0:
            return None
        return np.array(medoids)

    def _compute_modes(self, X):
        """Compute mode (highest density point) for each cluster using KDE."""
        if self.labels_ is None:
            return None
        unique_labels = np.unique(self.labels_)
        modes = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = X[self.labels_ == label]
            if len(cluster_points) == 0:
                continue
            kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(
                cluster_points
            )
            # Evaluate density at actual data points instead of exponential grid
            densities = kde.score_samples(cluster_points)
            mode_index = np.argmax(densities)
            modes.append(cluster_points[mode_index])
        if len(modes) == 0:
            return None
        return np.array(modes)

    @property
    def cluster_centers_(self):
        check_is_fitted(self)
        if self.model_ is not None and hasattr(self.model_, "cluster_centers_"):
            return self.model_.cluster_centers_
        return None


class ClustGridSearch(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        mode="full",
        n_trials=20,
        scoring="silhouette_score",
        verbose=False,
        show_progress_bar=True,
    ):
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

        if self.mode == "full":
            self.algorithms = [
                "kmeans",
                "kmedoids",
                "minibatchkmeans",
                "dbscan",
                "agglomerativeclustering",
                "meanshift",
                "spectralclustering",
                "affinitypropagation",
                "birch",
                "optics",
                "gaussianmixture",
                "hdbscan",
            ]
        elif self.mode == "fast":
            self.algorithms = ["kmeans", "hdbscan"]
        else:
            raise ValueError("Invalid mode. Use 'full' or 'fast'.")

    def fit(self, X, y=None):
        """
        Run clustering for all selected algorithms and return the best one based on the chosen scoring.

        :param X: Input data for clustering.
        """
        results = []
        for algorithm in self.algorithms:
            logger.info("Testing algorithm: %s", algorithm)

            optimizer = Optimizer(
                algorithm=algorithm,
                n_trials=self.n_trials,
                scoring=self.scoring,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )

            try:
                optimizer.fit(X)
                score = optimizer.study_.best_value
                results.append(
                    {
                        "algorithm": algorithm,
                        "mean_test_score": score,
                        "params": optimizer.best_params_,
                        "model": optimizer,
                    }
                )
            except Exception as e:
                logger.error("Error for algorithm %s: %s", algorithm, e)

        if not results:
            raise ValueError("No algorithms produced valid results.")

        self.cv_results_ = {
            "algorithm": [res["algorithm"] for res in results],
            "mean_test_score": [res["mean_test_score"] for res in results],
            "params": [res["params"] for res in results],
            "model": [res["model"] for res in results],
        }

        reverse = self.scoring != "davies_bouldin_score"
        scores = self.cv_results_["mean_test_score"]
        if reverse:
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmin(scores)
        self.best_index_ = best_idx
        self.best_score_ = scores[best_idx]
        self.best_params_ = self.cv_results_["params"][best_idx]
        self.best_estimator_ = self.cv_results_["model"][best_idx]
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.best_estimator_.labels_

    @property
    def labels_(self):
        check_is_fitted(self)
        return self.best_estimator_.labels_

    @property
    def cluster_centers_(self):
        check_is_fitted(self)
        return self.best_estimator_.cluster_centers_

    @property
    def centroids_(self):
        check_is_fitted(self)
        return self.best_estimator_.centroids_

    @property
    def medoids_(self):
        check_is_fitted(self)
        return self.best_estimator_.medoids_

    @property
    def modes_(self):
        check_is_fitted(self)
        return self.best_estimator_.modes_
