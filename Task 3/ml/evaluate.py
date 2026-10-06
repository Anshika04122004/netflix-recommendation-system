"""
Model evaluation module for the Netflix Audience Rating Classifier.
Calculates Accuracy, Precision, Recall, F1-Score, Classification Report,
Confusion Matrix, and Feature Importances using actual test set predictions.
Auspify Technologies - Machine Learning Internship | Task 3
"""

import json
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluates a trained model on the test dataset and returns comprehensive metrics.
    """
    y_pred = model.predict(X_test)

    # Core scalar metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    precision_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    # Detailed classification report
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Clean report for JSON serialization
    cleaned_report = {}
    for k, v in report_dict.items():
        if isinstance(v, dict):
            cleaned_report[k] = {
                metric: round(float(val), 4) for metric, val in v.items()
            }
        else:
            cleaned_report[k] = round(float(v), 4)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        'accuracy': round(accuracy, 4),
        'precision_weighted': round(precision_weighted, 4),
        'recall_weighted': round(recall_weighted, 4),
        'f1_weighted': round(f1_weighted, 4),
        'precision_macro': round(precision_macro, 4),
        'recall_macro': round(recall_macro, 4),
        'f1_macro': round(f1_macro, 4),
        'classification_report': cleaned_report,
        'confusion_matrix': cm
    }


def extract_feature_importances(model, feature_names, top_n=25):
    """
    Extracts top N feature importances from tree-based estimators.
    """
    if not hasattr(model, 'feature_importances_'):
        return []

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    results = []
    for idx in indices:
        results.append({
            'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
            'importance': round(float(importances[idx]), 5)
        })
    return results


def save_metrics_to_json(metrics_dict, filepath='models/metrics.json'):
    """
    Saves evaluated metrics dictionary to JSON file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Evaluation metrics successfully saved to %s", filepath)
