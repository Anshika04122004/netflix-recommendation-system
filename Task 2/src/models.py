"""
Models Module for Netflix Content Type Classification.
Defines model definitions, training loops, hyperparameter tuning, and model persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of diverse classification candidate models configured with standard baselines.
    
    Args:
        random_state: Seed for reproducibility.
        
    Returns:
        dict of model_name -> un-fitted estimator/pipeline.
    """
    models = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced', random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.85, random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=random_state
        ),
        "K-Nearest Neighbors": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ]),
        "Support Vector Classifier": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state))
        ]),
        "Naive Bayes": GaussianNB(),
    }
    return models


def evaluate_with_cross_validation(
    models: Dict[str, Any], 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates all candidate models using Stratified K-Fold Cross Validation.
    
    Args:
        models: Dictionary of models.
        X: Feature matrix.
        y: Target series.
        n_splits: Number of CV folds.
        random_state: Random state for reproducibility.
        
    Returns:
        pd.DataFrame: Table comparing CV mean and std for accuracy, precision, recall, f1, and roc_auc.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    results = []
    print(f"Starting {n_splits}-Fold Stratified Cross-Validation across {len(models)} algorithms...")
    
    for name, model in models.items():
        print(f"  --> Evaluating: {name}...")
        cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        row = {
            "Model": name,
            "CV Accuracy Mean": cv_scores['test_accuracy'].mean(),
            "CV Accuracy Std": cv_scores['test_accuracy'].std(),
            "CV Precision Mean": cv_scores['test_precision'].mean(),
            "CV Recall Mean": cv_scores['test_recall'].mean(),
            "CV F1 Mean": cv_scores['test_f1'].mean(),
            "CV F1 Std": cv_scores['test_f1'].std(),
            "CV ROC-AUC Mean": cv_scores['test_roc_auc'].mean(),
            "Train Accuracy Mean": cv_scores['train_accuracy'].mean(),
            "Fit Time (s)": cv_scores['fit_time'].mean()
        }
        results.append(row)
        
    summary_df = pd.DataFrame(results).sort_values(by="CV ROC-AUC Mean", ascending=False).reset_index(drop=True)
    return summary_df


def tune_and_build_ensemble(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Performs grid search hyperparameter tuning on top base learners and constructs an ensemble.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        random_state: Random seed.
        
    Returns:
        Tuple of (fitted best ensemble model, dictionary of all fitted candidate models)
    """
    print("\n--- Tuning Hyperparameters for Top Estimators ---")
    
    # 1. Tuned Random Forest
    rf_param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [8, 12, 16],
        'min_samples_split': [4, 8],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1),
        rf_param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best Random Forest params: {rf_grid.best_params_} (ROC-AUC: {rf_grid.best_score_:.4f})")

    # 2. Tuned Gradient Boosting
    gb_param_grid = {
        'n_estimators': [100, 180],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        gb_param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_
    print(f"Best Gradient Boosting params: {gb_grid.best_params_} (ROC-AUC: {gb_grid.best_score_:.4f})")

    # 3. Tuned Logistic Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'))
    ])
    lr_param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__solver': ['lbfgs', 'liblinear']
    }
    lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"Best Logistic Regression params: {lr_grid.best_params_} (ROC-AUC: {lr_grid.best_score_:.4f})")

    # 4. Extra Trees
    extra_trees = ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=random_state, n_jobs=-1)
    extra_trees.fit(X_train, y_train)

    # 5. Soft Voting Ensemble
    voting_ensemble = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('lr', best_lr),
            ('et', extra_trees)
        ],
        voting='soft',
        weights=[2, 3, 1, 2]
    )
    voting_ensemble.fit(X_train, y_train)

    # 6. Stacking Classifier
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('et', extra_trees)
        ],
        final_estimator=LogisticRegression(class_weight='balanced', random_state=random_state),
        cv=5,
        n_jobs=-1
    )
    stacking_ensemble.fit(X_train, y_train)

    all_fitted = {
        "Random Forest (Tuned)": best_rf,
        "Gradient Boosting (Tuned)": best_gb,
        "Logistic Regression (Tuned)": best_lr,
        "Extra Trees": extra_trees,
        "Soft Voting Ensemble": voting_ensemble,
        "Stacking Ensemble": stacking_ensemble
    }

    return voting_ensemble, all_fitted


def save_pipeline_artifacts(
    model: Any, 
    feature_extractor: Any, 
    tfidf_vectorizer: Any, 
    output_dir: str = "models"
) -> str:
    """
    Serializes all pipeline components for deployment into joblib files.
    
    Args:
        model: Trained classifier.
        feature_extractor: Fitted tabular feature extractor.
        tfidf_vectorizer: Fitted TF-IDF vectorizer.
        output_dir: Directory to save serialized objects.
        
    Returns:
        str: Path to saved artifact dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "model": model,
        "feature_extractor": feature_extractor,
        "tfidf_vectorizer": tfidf_vectorizer,
        "classes": ["Movie", "TV Show"]
    }
    bundle_path = os.path.join(output_dir, "content_type_pipeline.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"Serialized complete inference bundle to: {bundle_path}")
    return bundle_path


def load_pipeline_artifacts(artifact_path: str = "models/content_type_pipeline.joblib") -> Dict[str, Any]:
    """
    Loads serialized inference bundle.
    """
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Pipeline artifact not found at: {artifact_path}")
    return joblib.load(artifact_path)
