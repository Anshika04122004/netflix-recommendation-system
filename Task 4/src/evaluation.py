"""
Evaluation & Unsupervised Validation Module
Provides mathematical validation routines for clustering quality assessment.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

class ClusterEvaluator:
    @staticmethod
    def calculate_all_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Computes comprehensive clustering validation metrics.
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters < 2:
            return {
                "silhouette_score": 0.0,
                "davies_bouldin_score": 0.0,
                "calinski_harabasz_score": 0.0,
                "cluster_sizes": {int(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
                "cluster_balance_entropy": 0.0
            }
            
        sil_score = float(silhouette_score(X, labels))
        db_score = float(davies_bouldin_score(X, labels))
        ch_score = float(calinski_harabasz_score(X, labels))
        
        # Calculate cluster balance entropy
        counts = pd.Series(labels).value_counts().values
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(len(counts))
        normalized_balance = float(entropy / max_entropy) if max_entropy > 0 else 1.0
        
        # Cluster size breakdown
        cluster_sizes = {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}
        
        return {
            "num_clusters": n_clusters,
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_score": round(db_score, 4),
            "calinski_harabasz_score": round(ch_score, 2),
            "cluster_balance_score": round(normalized_balance, 4),
            "cluster_sizes": cluster_sizes
        }
