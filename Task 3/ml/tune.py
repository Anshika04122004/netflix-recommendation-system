"""
Hyperparameter tuning module for Decision Tree and Random Forest classifiers.
Auspify Technologies - Machine Learning Internship | Task 3
"""

import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def tune_decision_tree(X_train, y_train, cv=5):
    """
    Tune Decision Tree Classifier hyperparameters using GridSearchCV with 5-fold CV.
    Target hyperparameters: max_depth, min_samples_split, min_samples_leaf, criterion.
    """
    logger.info("Starting Decision Tree hyperparameter tuning (GridSearchCV, 5-fold CV)...")
    param_grid = {
        'max_depth': [8, 12, 16, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    base_dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_dt,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info("Decision Tree Best Parameters: %s", grid_search.best_params_)
    logger.info("Decision Tree Best CV Weighted F1: %.4f", grid_search.best_score_)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_random_forest(X_train, y_train, cv=5, n_iter=15):
    """
    Tune Random Forest Classifier hyperparameters using RandomizedSearchCV with 5-fold CV.
    Target hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
    """
    logger.info("Starting Random Forest hyperparameter tuning (RandomizedSearchCV, 5-fold CV)...")
    param_distributions = {
        'n_estimators': [100, 150, 200],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 8],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    base_rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)

    logger.info("Random Forest Best Parameters: %s", random_search.best_params_)
    logger.info("Random Forest Best CV Weighted F1: %.4f", random_search.best_score_)
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
