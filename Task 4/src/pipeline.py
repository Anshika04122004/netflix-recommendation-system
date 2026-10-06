"""
End-to-End Orchestrator Pipeline for Task 4: Netflix Content Segmentation
Integrates Ingestion, Feature Engineering, Multi-Model Clustering, Profiling, and Artifact Serialization.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib

from .data_loader import NetflixDataLoader
from .feature_engineering import NetflixFeatureEngineer
from .clustering_models import NetflixClusterModels
from .cluster_profiler import NetflixClusterProfiler
from .evaluation import ClusterEvaluator

class NetflixSegmentationPipeline:
    def __init__(self, data_path: str = "Dataset.csv", optimal_k: int = 6):
        self.data_path = data_path
        self.optimal_k = optimal_k
        self.loader = NetflixDataLoader(data_path)
        self.engineer = NetflixFeatureEngineer()
        self.cluster_models = NetflixClusterModels(optimal_k=optimal_k)
        self.profiler = NetflixClusterProfiler()
        self.evaluator = ClusterEvaluator()
        
        self.df: pd.DataFrame = None
        self.X: np.ndarray = None
        self.labels: np.ndarray = None
        self.projections: Dict[str, Any] = None
        self.profiles: Dict[int, Any] = None
        self.benchmark: Dict[str, Any] = None
        self.k_optim: list = []

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Executes Steps 1 to 5 of Task 4 in one seamless, validated flow.
        """
        print(">>> [Step 1] Ingesting & Engineering Features...")
        self.df = self.loader.load_and_clean()
        self.X = self.engineer.extract_features(self.df, fit=True)
        print(f"    Feature matrix shape: {self.X.shape}")

        print(">>> [Step 2] Benchmarking Clustering Algorithms & Optimizing K...")
        self.k_optim = self.cluster_models.evaluate_k_range(self.X, k_range=range(2, 11))
        self.benchmark = self.cluster_models.benchmark_algorithms(self.X)
        self.labels = self.cluster_models.fit_final_model(self.X)
        self.df['cluster'] = self.labels

        print(">>> [Step 3] Computing Projections for Step 4 Visualization...")
        proj_data = self.engineer.compute_projections(self.X, fit=True)
        self.projections = {
            "pca_2d": proj_data["pca_2d"].tolist(),
            "pca_3d": proj_data["pca_3d"].tolist(),
            "explained_variance_2d": proj_data["pca_2d_explained_variance"],
            "explained_variance_3d": proj_data["pca_3d_explained_variance"]
        }
        self.df['pca_x'] = proj_data["pca_2d"][:, 0]
        self.df['pca_y'] = proj_data["pca_2d"][:, 1]
        self.df['pca_z'] = proj_data["pca_3d"][:, 2]

        print(">>> [Step 4] Profiling Discovered Content Segments...")
        self.profiles = self.profiler.profile_all_clusters(self.df)

        print(">>> [Step 5] Evaluating Unsupervised Performance & Generating Strategic Insights...")
        eval_metrics = self.evaluator.calculate_all_metrics(self.X, self.labels)

        # Build complete serialized artifact dictionary
        results = {
            "summary_stats": self.loader.get_summary_stats(),
            "feature_engineering_info": {
                "num_features": len(self.engineer.feature_names),
                "feature_names": self.engineer.feature_names
            },
            "k_optimization": self.k_optim,
            "algorithm_benchmark": self.benchmark,
            "evaluation_metrics": eval_metrics,
            "cluster_profiles": self.profiles,
            "projections_info": {
                "explained_variance_2d": proj_data["pca_2d_explained_variance"],
                "explained_variance_3d": proj_data["pca_3d_explained_variance"]
            }
        }
        
        return results

    def save_artifacts(self, output_dir: str = "models") -> str:
        """Saves models, encoders, and precomputed catalog payload for the interactive Web App."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model binaries
        joblib.dump(self.engineer, os.path.join(output_dir, "feature_engineer.joblib"))
        joblib.dump(self.cluster_models, os.path.join(output_dir, "cluster_models.joblib"))
        
        # Save precomputed lightweight JSON for high-performance frontend visualization
        catalog_sample = self.df[[
            'show_id', 'title', 'type', 'director', 'country', 'release_year',
            'rating', 'duration', 'listed_in', 'cluster', 'pca_x', 'pca_y', 'pca_z'
        ]].to_dict(orient='records')
        
        payload = {
            "summary": self.loader.get_summary_stats(),
            "k_optimization": self.k_optim,
            "algorithm_benchmark": self.benchmark,
            "cluster_profiles": self.profiles,
            "pca_variance_2d": self.projections["explained_variance_2d"],
            "pca_variance_3d": self.projections["explained_variance_3d"],
            "total_titles": len(self.df),
            "titles": catalog_sample
        }
        
        json_path = os.path.join(output_dir, "segmentation_artifacts.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            
        print(f" Artifacts successfully saved to {output_dir}/")
        return json_path
