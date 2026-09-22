"""
Main Pipeline Runner for Netflix Content Type Prediction System (Task 2).
Executes end-to-end data ingestion, EDA, feature engineering, model training,
cross-validation, hyperparameter tuning, ensembling, evaluation, and artifact generation.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Set base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset, clean_data, get_dataset_summary
from src.feature_engineering import prepare_training_data
from src.models import (
    get_candidate_models,
    evaluate_with_cross_validation,
    tune_and_build_ensemble,
    save_pipeline_artifacts
)
from src.evaluation import (
    evaluate_multiple_models,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_model_comparison_bar_chart,
    plot_feature_importance,
    generate_eda_figures,
    save_evaluation_summary
)
from src.predict import NetflixPredictor


def run_pipeline(dataset_path: str = "Dataset.csv"):
    start_time = time.time()
    print("=" * 80)
    print("      NETFLIX CONTENT TYPE PREDICTION SYSTEM (TASK 2) - FULL PIPELINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: Data Ingestion, Cleaning & Exploratory Analysis
    # ---------------------------------------------------------
    print("\n[Step 1/5] Loading & Cleaning Dataset...")
    raw_df = load_dataset(os.path.join(BASE_DIR, dataset_path))
    df = clean_data(raw_df)
    summary = get_dataset_summary(df)
    
    print(f"Loaded {summary['total_records']:,} verified titles.")
    print(f"Target Distribution: {summary['target_distribution']}")
    print(f"Release Year Span: {summary['release_year_range'][0]} - {summary['release_year_range'][1]}")
    print(f"Unique Countries: {summary['unique_countries']}, Unique Ratings: {len(summary['unique_ratings'])}")

    print("\nGenerating Exploratory Data Analysis (EDA) charts...")
    generate_eda_figures(df, save_dir=os.path.join(BASE_DIR, "reports/figures"))

    # ---------------------------------------------------------
    # STEP 2: Feature Engineering & Preprocessing
    # ---------------------------------------------------------
    print("\n[Step 2/5] Engineering Temporal, Text & Categorical Features...")
    X_train, X_test, y_train, y_test, feat_extractor, tfidf_vec = prepare_training_data(
        df, test_size=0.2, random_state=42
    )
    print(f"Extracted {X_train.shape[1]} engineered features across {len(X_train):,} training and {len(X_test):,} testing instances.")

    # ---------------------------------------------------------
    # STEP 3: Multi-Model Benchmark & Cross-Validation
    # ---------------------------------------------------------
    print("\n[Step 3/5] Benchmarking Diverse Classification Algorithms (5-Fold Stratified CV)...")
    candidate_models = get_candidate_models(random_state=42)
    cv_summary = evaluate_with_cross_validation(candidate_models, X_train, y_train, n_splits=5, random_state=42)
    print("\n--- 5-Fold Stratified Cross-Validation Results ---")
    print(cv_summary.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 4: Hyperparameter Optimization & Ensembling
    # ---------------------------------------------------------
    print("\n[Step 4/5] Hyperparameter Tuning & Building Calibrated Ensembles...")
    best_ensemble, fitted_models = tune_and_build_ensemble(X_train, y_train, random_state=42)

    # ---------------------------------------------------------
    # STEP 5: Test Evaluation & Diagnostics Generation
    # ---------------------------------------------------------
    print("\n[Step 5/5] Evaluating Fitted Models on Held-Out Test Set...")
    test_summary, metrics_dict = evaluate_multiple_models(fitted_models, X_test, y_test)
    print("\n--- Final Test Set Benchmark ---")
    print(test_summary.to_string(index=False))

    # Generate Diagnostics Visualizations
    print("\nGenerating Diagnostic Plots and Figures...")
    y_test_pred = best_ensemble.predict(X_test)
    plot_confusion_matrix(
        y_test.values, y_test_pred, 
        classes=['Movie', 'TV Show'],
        model_name="Calibrated Voting Ensemble",
        save_path=os.path.join(BASE_DIR, "reports/figures/confusion_matrix.png")
    )
    plot_roc_and_pr_curves(
        fitted_models, X_test, y_test, 
        save_dir=os.path.join(BASE_DIR, "reports/figures")
    )
    plot_model_comparison_bar_chart(
        test_summary, 
        save_path=os.path.join(BASE_DIR, "reports/figures/model_comparison.png")
    )
    
    # Feature Importance
    best_tree_model = fitted_models.get("Random Forest (Tuned)", fitted_models.get("Gradient Boosting (Tuned)"))
    plot_feature_importance(
        best_tree_model, 
        feature_names=X_train.columns.tolist(), 
        top_n=25,
        save_path=os.path.join(BASE_DIR, "reports/figures/feature_importance.png")
    )

    # Save Pipeline & Serialization
    bundle_path = save_pipeline_artifacts(
        model=best_ensemble,
        feature_extractor=feat_extractor,
        tfidf_vectorizer=tfidf_vec,
        output_dir=os.path.join(BASE_DIR, "models")
    )

    # Save Metrics JSON
    export_metrics = {
        "dataset_summary": summary,
        "cv_results": cv_summary.to_dict(orient="records"),
        "test_results": test_summary.to_dict(orient="records"),
        "best_model_name": "Soft Voting Ensemble (RF + GB + LR + ET)",
        "best_model_test_metrics": metrics_dict.get("Soft Voting Ensemble", {})
    }
    save_evaluation_summary(export_metrics, save_path=os.path.join(BASE_DIR, "reports/metrics_summary.json"))

    # ---------------------------------------------------------
    # Verification & Sample Inference Checks
    # ---------------------------------------------------------
    print("\n--- Running Live Inference Verification Checks ---")
    predictor = NetflixPredictor(bundle_path)
    
    test_cases = [
        {
            "title": "Inception",
            "director": "Christopher Nolan",
            "country": "United States, United Kingdom",
            "release_year": 2010,
            "rating": "PG-13",
            "listed_in": "Action & Adventure, Sci-Fi & Fantasy, Thrillers",
            "date_added": "1/1/2020",
            "expected": "Movie"
        },
        {
            "title": "Stranger Things",
            "director": "Unknown",
            "country": "United States",
            "release_year": 2016,
            "rating": "TV-14",
            "listed_in": "TV Dramas, TV Horror, TV Mysteries, TV Sci-Fi & Fantasy",
            "date_added": "7/15/2016",
            "expected": "TV Show"
        },
        {
            "title": "The Irishman",
            "director": "Martin Scorsese",
            "country": "United States",
            "release_year": 2019,
            "rating": "R",
            "listed_in": "Dramas, Crime Movies",
            "date_added": "11/27/2019",
            "expected": "Movie"
        },
        {
            "title": "The Crown",
            "director": "Unknown",
            "country": "United Kingdom, United States",
            "release_year": 2020,
            "rating": "TV-MA",
            "listed_in": "British TV Shows, TV Dramas",
            "date_added": "11/15/2020",
            "expected": "TV Show"
        }
    ]

    for item in test_cases:
        res = predictor.predict_single(item)
        is_match = "PASS" if res['prediction'] == item['expected'] else "FAIL"
        print(f"[{is_match}] '{item['title']}' -> Predicted: {res['prediction']} ({res['confidence']}%), Expected: {item['expected']}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" Pipeline completed successfully in {elapsed:.2f} seconds!")
    print(f" Saved Model: {bundle_path}")
    print(f" Generated Reports: reports/figures/ & reports/metrics_summary.json")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline("Dataset.csv")
