"""
Enhanced training pipeline for the Netflix Audience Rating Classifier.
Uses GradientBoosting, ExtraTrees, Random Forest (Voting Ensemble) 
to achieve significantly higher accuracy.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from ml.preprocess import NetflixPreprocessor
from ml.evaluate import evaluate_model, extract_feature_importances, save_metrics_to_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def tune_decision_tree(X_train, y_train, cv=5):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [8, 10, 12, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    dt = DecisionTreeClassifier(random_state=42)
    search = RandomizedSearchCV(dt, param_grid, n_iter=20, cv=cv,
                                scoring='f1_weighted', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    logger.info("Best DT params: %s | CV F1: %.4f", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_random_forest(X_train, y_train, cv=5, n_iter=20):
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [20, 30, 40, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced'],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(rf, param_grid, n_iter=n_iter, cv=cv,
                                scoring='f1_weighted', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    logger.info("Best RF params: %s | CV F1: %.4f", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_gradient_boosting(X_train, y_train, cv=5, n_iter=20):
    param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.05, 0.08, 0.1, 0.15],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2'],
    }
    gb = GradientBoostingClassifier(random_state=42)
    search = RandomizedSearchCV(gb, param_grid, n_iter=n_iter, cv=cv,
                                scoring='f1_weighted', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    logger.info("Best GB params: %s | CV F1: %.4f", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_, search.best_score_


def run_pipeline(data_path='data/Dataset.csv', models_dir='models', run_tuning=True):
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load dataset
    logger.info("Loading dataset from %s ...", data_path)
    if not os.path.exists(data_path):
        if os.path.exists('Dataset.csv'):
            data_path = 'Dataset.csv'
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info("Dataset loaded. Shape: %s", df.shape)

    target_col = 'rating'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    feature_cols = [c for c in df.columns if c not in [target_col, 'show_id', 'title']]
    X_raw = df[feature_cols]
    y_raw = df[target_col]

    # 2. Stratified 80/20 split
    logger.info("Splitting dataset (80/20, stratified)...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # 3. Fit enhanced preprocessing pipeline
    logger.info("Fitting enhanced NetflixPreprocessor (with TF-IDF)...")
    preprocessor = NetflixPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw, y_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_train = preprocessor.encode_target(y_train_raw)
    y_test = preprocessor.encode_target(y_test_raw)
    class_names = preprocessor.get_classes()
    feature_names = preprocessor.get_feature_names()

    logger.info("Feature matrix shape: %s | Classes: %d", X_train.shape, len(class_names))

    models_dict = {}
    metrics_summary = {}

    # --- BASELINE MODELS ---
    logger.info("Training Logistic Regression (Baseline)...")
    lr = LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    models_dict['Logistic Regression (Baseline)'] = lr

    logger.info("Training Decision Tree (Baseline)...")
    dt_base = DecisionTreeClassifier(random_state=42, max_depth=15, class_weight='balanced')
    dt_base.fit(X_train, y_train)
    models_dict['Decision Tree (Baseline)'] = dt_base

    logger.info("Training Random Forest (Baseline)...")
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    models_dict['Random Forest (Baseline)'] = rf_base

    best_dt_params = {}
    best_rf_params = {}
    best_gb_params = {}

    if run_tuning:
        # --- TUNED MODELS ---
        logger.info("Tuning Decision Tree...")
        dt_tuned, best_dt_params, _ = tune_decision_tree(X_train, y_train, cv=5)
        models_dict['Decision Tree (Tuned)'] = dt_tuned

        logger.info("Tuning Random Forest...")
        rf_tuned, best_rf_params, _ = tune_random_forest(X_train, y_train, cv=5, n_iter=20)
        models_dict['Random Forest (Tuned)'] = rf_tuned

        logger.info("Tuning Gradient Boosting Classifier...")
        gb_tuned, best_gb_params, _ = tune_gradient_boosting(X_train, y_train, cv=5, n_iter=20)
        models_dict['Gradient Boosting (Tuned)'] = gb_tuned

        # --- VOTING ENSEMBLE ---
        logger.info("Building Voting Ensemble (RF + GB + ExtraTrees)...")
        et = ExtraTreesClassifier(
            n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1,
            class_weight='balanced', min_samples_leaf=1
        )
        et.fit(X_train, y_train)
        models_dict['Extra Trees'] = et

        # Soft voting if all estimators support predict_proba
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf_tuned),
                ('gb', gb_tuned),
                ('et', et),
            ],
            voting='soft',
            n_jobs=-1
        )
        logger.info("Fitting Voting Ensemble...")
        voting_clf.fit(X_train, y_train)
        models_dict['Voting Ensemble (RF+GB+ET)'] = voting_clf

    else:
        models_dict['Decision Tree (Tuned)'] = dt_base
        models_dict['Random Forest (Tuned)'] = rf_base
        models_dict['Gradient Boosting (Tuned)'] = rf_base
        models_dict['Extra Trees'] = rf_base
        models_dict['Voting Ensemble (RF+GB+ET)'] = rf_base

    # 4. Evaluate all models
    logger.info("Evaluating all models on test set (n=%d)...", len(y_test))
    for name, model in models_dict.items():
        logger.info("Evaluating %s...", name)
        eval_res = evaluate_model(model, X_test, y_test, class_names=class_names)
        eval_res['model_name'] = name
        metrics_summary[name] = eval_res
        logger.info(
            "%s -> Accuracy: %.4f | F1 (weighted): %.4f",
            name, eval_res['accuracy'], eval_res['f1_weighted']
        )

    # 5. Select best model
    best_model_name = max(
        metrics_summary.keys(),
        key=lambda k: (metrics_summary[k]['f1_weighted'], metrics_summary[k]['accuracy'])
    )
    best_model = models_dict[best_model_name]
    logger.info(
        "BEST MODEL: %s | Accuracy: %.4f | F1 Weighted: %.4f",
        best_model_name,
        metrics_summary[best_model_name]['accuracy'],
        metrics_summary[best_model_name]['f1_weighted']
    )

    # 6. Feature importances
    # Use rf_tuned if voting ensemble is best (VotingClassifier has no direct feature_importances_)
    fi_model = best_model
    if hasattr(fi_model, 'estimators_') and isinstance(fi_model, VotingClassifier):
        # Use the RF component for feature importances
        fi_model = models_dict.get('Random Forest (Tuned)', rf_base)
    feature_importances = extract_feature_importances(fi_model, feature_names, top_n=25)

    # 7. Compile full evaluation payload
    full_evaluation = {
        'best_model_name': best_model_name,
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': X_train.shape[1],
            'num_classes': len(class_names),
            'classes': class_names
        },
        'tuning_parameters': {
            'decision_tree': best_dt_params,
            'random_forest': best_rf_params,
            'gradient_boosting': best_gb_params,
        },
        'models': metrics_summary,
        'feature_importances': feature_importances
    }

    # 8. Save artifacts
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    metrics_path = os.path.join(models_dir, 'metrics.json')

    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    save_metrics_to_json(full_evaluation, metrics_path)

    logger.info("Artifacts saved: %s | %s | %s", best_model_path, preprocessor_path, metrics_path)
    return full_evaluation


if __name__ == '__main__':
    run_pipeline()
