"""
Clustering Algorithms and Model Benchmarking Module
Implements K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and GMM.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

class NetflixClusterModels:
    def __init__(self, optimal_k: int = 6):
        self.optimal_k = optimal_k
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=15, max_iter=300)
        self.gmm_model = GaussianMixture(n_components=optimal_k, random_state=42, n_init=5)
        self.agg_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        self.dbscan_model = DBSCAN(eps=1.8, min_samples=15)
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        
        self.benchmark_results: Dict[str, Any] = {}
        self.k_optimization_results: List[Dict[str, Any]] = []

    def evaluate_k_range(self, X: np.ndarray, k_range: range = range(2, 11)) -> List[Dict[str, Any]]:
        """
        Runs Elbow Method and evaluates Silhouette, Davies-Bouldin, and Calinski-Harabasz across K values.
        """
        results = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
            labels = km.fit_predict(X)
            
            sil = float(silhouette_score(X, labels))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            inertia = float(km.inertia_)
            
            results.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4),
                "davies_bouldin_score": round(db, 4),
                "calinski_harabasz_score": round(ch, 2)
            })
        self.k_optimization_results = results
        return results

    def benchmark_algorithms(self, X: np.ndarray, sample_size: int = 4000) -> Dict[str, Dict[str, Any]]:
        """
        Benchmarks multiple clustering algorithms on the engineered feature matrix.
        """
        # Subsample for computationally intensive algorithms if necessary
        n_samples = len(X)
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_eval = X[indices]
        else:
            X_eval = X

        benchmark = {}

        # 1. K-Means
        km_labels = self.kmeans_model.fit_predict(X)
        km_sil = float(silhouette_score(X_eval, km_labels[indices] if n_samples > sample_size else km_labels))
        km_db = float(davies_bouldin_score(X_eval, km_labels[indices] if n_samples > sample_size else km_labels))
        km_ch = float(calinski_harabasz_score(X_eval, km_labels[indices] if n_samples > sample_size else km_labels))
        benchmark["K-Means (K=6)"] = {
            "algorithm": "K-Means",
            "silhouette_score": round(km_sil, 4),
            "davies_bouldin_score": round(km_db, 4),
            "calinski_harabasz_score": round(km_ch, 2),
            "num_clusters": self.optimal_k,
            "status": "Production Selected",
            "strengths": "Spherical partition, high computational efficiency, deterministic cluster centers"
        }

        # 2. Agglomerative Hierarchical
        agg_labels = self.agg_model.fit_predict(X_eval)
        agg_sil = float(silhouette_score(X_eval, agg_labels))
        agg_db = float(davies_bouldin_score(X_eval, agg_labels))
        agg_ch = float(calinski_harabasz_score(X_eval, agg_labels))
        benchmark["Agglomerative Hierarchical"] = {
            "algorithm": "Hierarchical (Ward)",
            "silhouette_score": round(agg_sil, 4),
            "davies_bouldin_score": round(agg_db, 4),
            "calinski_harabasz_score": round(agg_ch, 2),
            "num_clusters": self.optimal_k,
            "status": "Benchmarked",
            "strengths": "Hierarchical taxonomy tree, captures nested sub-genre structures"
        }

        # 3. Gaussian Mixture Model (GMM)
        self.gmm_model.fit(X)
        gmm_labels = self.gmm_model.predict(X_eval)
        gmm_sil = float(silhouette_score(X_eval, gmm_labels))
        gmm_db = float(davies_bouldin_score(X_eval, gmm_labels))
        gmm_ch = float(calinski_harabasz_score(X_eval, gmm_labels))
        benchmark["Gaussian Mixture (GMM)"] = {
            "algorithm": "GMM (Soft Clustering)",
            "silhouette_score": round(gmm_sil, 4),
            "davies_bouldin_score": round(gmm_db, 4),
            "calinski_harabasz_score": round(gmm_ch, 2),
            "num_clusters": self.optimal_k,
            "status": "Benchmarked",
            "strengths": "Probabilistic membership, elliptical covariance modeling"
        }

        # 4. DBSCAN
        db_labels = self.dbscan_model.fit_predict(X_eval)
        n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if n_db_clusters > 1:
            # Filter noise for silhouette calculation
            valid_mask = db_labels != -1
            if valid_mask.sum() > 10:
                db_sil = float(silhouette_score(X_eval[valid_mask], db_labels[valid_mask]))
                db_dbs = float(davies_bouldin_score(X_eval[valid_mask], db_labels[valid_mask]))
                db_chs = float(calinski_harabasz_score(X_eval[valid_mask], db_labels[valid_mask]))
            else:
                db_sil, db_dbs, db_chs = 0.0, 0.0, 0.0
        else:
            db_sil, db_dbs, db_chs = 0.0, 0.0, 0.0

        benchmark["DBSCAN (Density-Based)"] = {
            "algorithm": "DBSCAN",
            "silhouette_score": round(db_sil, 4),
            "davies_bouldin_score": round(db_dbs, 4),
            "calinski_harabasz_score": round(db_chs, 2),
            "num_clusters": n_db_clusters,
            "status": "Benchmarked",
            "strengths": "Arbitrary shape clusters, automated outlier / niche content isolation"
        }

        self.benchmark_results = benchmark
        return benchmark

    def fit_final_model(self, X: np.ndarray) -> np.ndarray:
        """Fits primary KMeans model and NearestNeighbors index on the entire dataset."""
        labels = self.kmeans_model.fit_predict(X)
        self.nn_model.fit(X)
        return labels

    def predict_new(self, x_vec: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Predicts cluster for a new feature vector and returns:
        (predicted_cluster, centroid_distances, nearest_neighbor_indices)
        """
        if x_vec.ndim == 1:
            x_vec = x_vec.reshape(1, -1)
            
        cluster_id = int(self.kmeans_model.predict(x_vec)[0])
        # Distances to all centroids
        distances = np.linalg.norm(self.kmeans_model.cluster_centers_ - x_vec, axis=1)
        # Nearest neighbors
        nn_dists, nn_indices = self.nn_model.kneighbors(x_vec, n_neighbors=6)
        
        return cluster_id, distances, nn_indices[0]
