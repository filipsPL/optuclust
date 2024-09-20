import optuna
import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.cluster import AffinityPropagation, Birch, OPTICS
from sklearn.mixture import GaussianMixture
import hdbscan
from kmedoids import KMedoids
import time

from sklearn_som.som import SOM

class Optimizer:
    def __init__(self, algorithm, n_trials=50, scoring=['silhouette_score'], verbose=False, show_progress_bar=True):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.scoring = scoring if isinstance(scoring, list) else [scoring]
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.best_params_ = None
        self.study = None
        self.model = None
        self.X_ = None  # Store X after fitting

        # Set Optuna logging verbosity based on 'verbose' parameter
        if isinstance(verbose, bool):
            optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
        elif isinstance(verbose, int):
            optuna.logging.set_verbosity(verbose)
    
    def fit(self, X):
        self.X_ = X  # Store X for later use

        def objective(trial):
            model = self._suggest_model(trial, X)
            model.fit(X)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)

            # Scoring logic, pass the trial object for pruning
            scores = self._compute_scores(X, labels, trial)
            return scores

        # Create a study with directions for all objectives (we are maximizing all)
        self.study = optuna.create_study(directions=['maximize'] * len(self.scoring))
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.show_progress_bar)
        self.best_params_ = self.study.best_params
        self.model = self._get_best_model(X)
        self.model.fit(X)

    def _compute_scores(self, X, labels, trial):
        # Prune if there is only one cluster
        if len(set(labels)) <= 1:
            raise optuna.TrialPruned("Only one cluster found, pruning this trial.")

        scores = []
        for scoring_method in self.scoring:
            if scoring_method == 'silhouette_score':
                score = silhouette_score(X, labels)
            elif scoring_method == 'calinski_harabasz_score':
                score = calinski_harabasz_score(X, labels)
            elif scoring_method == 'davies_bouldin_score':
                score = -1 * davies_bouldin_score(X, labels)  # Return negative davies_bouldin_score
            else:
                raise ValueError(f"Unsupported scoring method: {scoring_method}")
            scores.append(score)

        return scores

    @property
    def labels_(self):
        if hasattr(self.model, 'labels_'):
            return self.model.labels_
        elif self.X_ is not None:
            return self.model.predict(self.X_)  # Use stored X_
        else:
            raise ValueError("Model has not been fitted and X is not available for prediction.")

    def _suggest_model(self, trial, X):
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
            method = trial.suggest_categorical('method', ['fasterpam',  "pam", "alternate", "fastermsc", "fastmsc", "pamsil", "pammedsil"])
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

    def _compute_score(self, X, labels, trial):
        # Check if only one cluster was found and prune the trial
        if len(set(labels)) <= 1:
            raise optuna.TrialPruned("Only one cluster found, pruning this trial.")
        
        scores = []
        for scoring_method in self.scoring:
            if scoring_method == 'silhouette_score':
                score = silhouette_score(X, labels)
            elif scoring_method == 'calinski_harabasz_score':
                score = calinski_harabasz_score(X, labels)
            elif scoring_method == 'davies_bouldin_score':
                score = -1 * davies_bouldin_score(X, labels)  # Return negative davies_bouldin_score
            else:
                raise ValueError(f"Unsupported scoring method: {scoring_method}")
            scores.append(score)
        
        return sum(scores) / len(scores)


    def _get_best_model(self, X):
        # Recreate the best model with optimized parameters
        trial = optuna.trial.FixedTrial(self.best_params_)
        return self._suggest_model(trial, X)  # Pass X to _suggest_model

    @property
    def cluster_centers_(self):
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None

    @property
    def centroids_(self):
        # For algorithms that don't provide centroids directly (e.g., DBSCAN), calculate using NumPy
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        else:
            # Calculate centroids manually for each cluster
            unique_labels = np.unique(self.labels_)
            return np.array([self.X_[self.labels_ == label].mean(axis=0) for label in unique_labels])

    @property
    def medoids_(self):
        # For algorithms that don’t have medoids, calculate using the pairwise distance between points
        if isinstance(self.model, KMedoids):
            return self.model.cluster_centers_
        else:
            # Calculate medoids manually by minimizing intra-cluster distance
            unique_labels = np.unique(self.labels_)
            medoids = []
            for label in unique_labels:
                cluster_points = self.X_[self.labels_ == label]
                distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :]), axis=2)
                medoids.append(cluster_points[np.argmin(np.sum(distances, axis=1))])
            return np.array(medoids)
        

class ClustGridSearch:
    def __init__(self, mode="full", scoring="silhouette_score", verbose=False):
        """
        Initialize the ClustGridSearch.
        
        :param mode: 'full' to test all algorithms, 'fast' to test a subset (kmeans and hdbscan)
        :param scoring: The metric used to select the best clustering (default: 'silhouette_score')
        :param verbose: Whether to print additional information during the search
        """
        self.mode = mode
        self.scoring = scoring
        self.verbose = verbose
        self.cv_results_ = []
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_algorithm_name = None
        self.best_params_ = None
        
        # Define algorithms to test based on mode
        if self.mode == "full":
            self.algorithms = [
                'som', 'kmeans', 'kmedoids', 'minibatchkmeans', 'dbscan', 
                'agglomerativeclustering', 'meanshift', 'spectralclustering', 
                'affinitypropagation', 'birch', 'optics', 'gaussianmixture', 'hdbscan'
            ]
        elif self.mode == "fast":
            self.algorithms = ['kmeans', 'hdbscan']
        else:
            raise ValueError("Invalid mode. Use 'full' or 'fast'.")

    def fit(self, X):
        """
        Run clustering for all selected algorithms and return the best one based on the chosen scoring.
        
        :param X: Input data for clustering
        """
        for algorithm in self.algorithms:
            if self.verbose:
                print(f"Testing algorithm: {algorithm}")
            
            optimizer = Optimizer(algorithm=algorithm, n_trials=20, scoring=self.scoring, verbose=self.verbose)
            
            start_time = time.time()
            try:
                # Perform clustering
                optimizer.fit(X)
                labels = optimizer.labels_

                # Calculate scores
                sil_score = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                db_score = -1 * davies_bouldin_score(X, labels)
                
                # Record clustering time and number of clusters
                clustering_time = time.time() - start_time
                n_clusters = len(np.unique(labels))

                # Store results in cv_results_, including the fitted model
                self.cv_results_.append({
                    'algorithm': algorithm,
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                    'davies_bouldin_score': db_score,
                    'clustering_time': clustering_time,
                    'n_clusters': n_clusters,
                    'parameters': optimizer.best_params_,
                    'model': optimizer  # Store the fitted model
                })

            except optuna.TrialPruned:
                print(f"Trial pruned for algorithm: {algorithm}")
            except Exception as e:
                print(f"Error for algorithm {algorithm}: {e}")

        # Extract the best result from cv_results_
        self._extract_best_result()

    def _extract_best_result(self):
        """
        Extract the best algorithm and its parameters from cv_results_ based on the chosen scoring metric.
        """
        if self.cv_results_:
            best_result = max(self.cv_results_, key=lambda result: result[self.scoring])
            self.best_algorithm_name = best_result['algorithm']
            self.best_score_ = best_result[self.scoring]
            self.best_params_ = best_result['parameters']
            self.best_estimator_ = best_result['model']  # Retrieve the best fitted model

    def get_results(self):
        """
        Return a detailed list of results for all tested algorithms.
        """
        return self.cv_results_
