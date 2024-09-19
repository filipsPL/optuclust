import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from concurrent.futures import ProcessPoolExecutor
from optuna import TrialPruned
from optuclust import Optimizer
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

# Directory containing datasets
DATA_DIR = "data/"
# Directory to save results
RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define clustering algorithms
ALGORITHMS = [
    'som', 'kmeans',
    #   'kmedoids', 'minibatchkmeans', 'dbscan', 
    # 'agglomerativeclustering', 'meanshift', 'spectralclustering', 
    # 'affinitypropagation', 'birch', 'optics', 'gaussianmixture', 'hdbscan'
]

# Prepare performance tracking
performance_data = []

# Read all CSV files from the data directory
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

# Prepare the figure for the final combined plots
n_rows = len(csv_files)
n_cols = len(ALGORITHMS)
final_fig, final_axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))

# Function to perform clustering for a specific dataset and algorithm
def perform_clustering(file_name, algorithm, row_idx, col_idx):
    print(f"Processing dataset: {file_name} with algorithm: {algorithm}")
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :2].values  # Use the first two columns without headers
    X = StandardScaler().fit_transform(X)  # Scale the data

    # Create the optimizer
    optimizer = Optimizer(algorithm=algorithm, n_trials=20, scoring='silhouette_score', verbose=False)
    
    start_time = time.time()
    try:
        # Perform clustering
        optimizer.fit(X)
        labels = optimizer.labels_

        # Calculate silhouette score, calinski_harabasz_score, and -1 * davies_bouldin_score
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = -1 * davies_bouldin_score(X, labels)

        # Record clustering time
        clustering_time = time.time() - start_time

        # Calculate number of clusters
        n_clusters = len(np.unique(labels))

        # Plot the clusters with a qualitative color palette
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
        
        # Add centroids to the plot, if available
        if hasattr(optimizer, 'centroids_') and optimizer.centroids_ is not None:
            ax.scatter(optimizer.centroids_[:, 0], optimizer.centroids_[:, 1], c='red', s=200, marker='X', label='Centroid')
        
        # Add medoids with different color and symbol, if available
        if hasattr(optimizer, 'medoids_') and optimizer.medoids_ is not None:
            ax.scatter(optimizer.medoids_[:, 0], optimizer.medoids_[:, 1], c='blue', s=200, marker='*', label='Medoid')

        # Add plot title and legend
        ax.set_title(f"{algorithm}\nSilhouette: {sil_score:.2f}, CH: {ch_score:.2f}, DB: {db_score:.2f}, Clusters: {n_clusters}\nTime: {clustering_time:.2f}s")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        # Save the individual plot
        plot_file = os.path.join(RESULTS_DIR, f"{file_name}_{algorithm}.png")
        plt.savefig(plot_file)
        plt.close(fig)

        # Add to final combined figure
        final_ax = final_axes[row_idx, col_idx]
        scatter = final_ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
        if hasattr(optimizer, 'centroids_') and optimizer.centroids_ is not None:
            final_ax.scatter(optimizer.centroids_[:, 0], optimizer.centroids_[:, 1], c='red', s=200, marker='X')
        if hasattr(optimizer, 'medoids_') and optimizer.medoids_ is not None:
            final_ax.scatter(optimizer.medoids_[:, 0], optimizer.medoids_[:, 1], c='blue', s=200, marker='*')
        final_ax.set_title(f"{algorithm}\nSilhouette: {sil_score:.2f}, CH: {ch_score:.2f}, DB: {db_score:.2f}, Clusters: {n_clusters}\nTime: {clustering_time:.2f}s")
        final_ax.set_xlabel('x')
        final_ax.set_ylabel('y')

        return {
            'dataset': file_name,
            'algorithm': algorithm,
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'clustering_time': clustering_time,
            'n_clusters': n_clusters
        }
    
    except TrialPruned:
        print(f"Trial was pruned for algorithm: {algorithm} on dataset: {file_name}")
        return {
            'dataset': file_name,
            'algorithm': algorithm,
            'silhouette_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'davies_bouldin_score': np.nan,
            'clustering_time': np.nan,
            'n_clusters': np.nan
        }
    
    except Exception as e:
        print(f"Error for algorithm {algorithm} on dataset {file_name}: {e}")
        return {
            'dataset': file_name,
            'algorithm': algorithm,
            'silhouette_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'davies_bouldin_score': np.nan,
            'clustering_time': np.nan,
            'n_clusters': np.nan
        }

# Function to handle a single dataset
def process_dataset(file_name, row_idx):
    results = []
    for col_idx, algorithm in enumerate(ALGORITHMS):
        result = perform_clustering(file_name, algorithm, row_idx, col_idx)
        results.append(result)
    return results

# Perform the benchmark in parallel
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_dataset, file_name, row_idx) for row_idx, file_name in enumerate(csv_files)]
    for future in futures:
        performance_data.extend(future.result())

# Save the final combined figure
final_fig.tight_layout()
final_fig_path = os.path.join(RESULTS_DIR, "benchmark_combined_results.png")
final_fig.savefig(final_fig_path)
plt.close(final_fig)

# Save performance data to a CSV file
performance_df = pd.DataFrame(performance_data)
performance_df.to_csv(os.path.join(RESULTS_DIR, "performance_data.csv"), index=False)

print("Benchmarking completed. Results saved to:", RESULTS_DIR)
