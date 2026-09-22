"""
Evaluation and Diagnostics Module for Netflix Content Type Classification.
Computes comprehensive metrics and generates publication-grade visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    balanced_accuracy_score
)

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculates key binary classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["average_precision"] = 0.0
            metrics["log_loss"] = 0.0
    return metrics


def evaluate_multiple_models(
    models_dict: Dict[str, Any], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Evaluates a collection of trained models on the test set.
    """
    results = []
    full_metrics = {}
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-raw_scores))
        else:
            y_proba = None
            
        m = calculate_metrics(y_test, y_pred, y_proba)
        full_metrics[name] = m
        
        row = {
            "Model": name,
            "Accuracy": m["accuracy"],
            "Balanced Acc": m["balanced_accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1_score"],
            "ROC-AUC": m.get("roc_auc", np.nan),
            "Avg Precision (PR-AUC)": m.get("average_precision", np.nan),
            "Log Loss": m.get("log_loss", np.nan)
        }
        results.append(row)
        
    summary_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return summary_df, full_metrics


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: List[str] = ['Movie', 'TV Show'],
    model_name: str = "Best Ensemble Model",
    save_path: str = "reports/figures/confusion_matrix.png"
):
    """
    Plots a normalized & annotated confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
            
    sns.heatmap(
        cm, annot=annot, fmt='', cmap='Blues', cbar=True,
        xticklabels=classes, yticklabels=classes, ax=ax,
        linewidths=1.5, linecolor='white', annot_kws={"size": 13, "weight": "bold"}
    )
    
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, weight='bold', pad=15)
    ax.set_ylabel("True Category", fontsize=12, weight='bold')
    ax.set_xlabel("Predicted Category", fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


def plot_roc_and_pr_curves(
    models_dict: Dict[str, Any], 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    save_dir: str = "reports/figures"
):
    """
    Plots comparative ROC and Precision-Recall curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. ROC Curves
    plt.figure(figsize=(9, 7), dpi=300)
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
            
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', linewidth=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, weight='bold')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12, weight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    roc_path = os.path.join(save_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to: {roc_path}")

    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 7), dpi=300)
    no_skill = y_test.sum() / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'Baseline (Prevalence = {no_skill:.3f})', linewidth=1.5)
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", linewidth=2)
            
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('Recall', fontsize=12, weight='bold')
    plt.ylabel('Precision', fontsize=12, weight='bold')
    plt.title('Precision-Recall (PR) Curves Comparison', fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    pr_path = os.path.join(save_dir, "pr_curves.png")
    plt.tight_layout()
    plt.savefig(pr_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision-Recall curves to: {pr_path}")


def plot_model_comparison_bar_chart(
    summary_df: pd.DataFrame, 
    save_path: str = "reports/figures/model_comparison.png"
):
    """
    Bar chart comparing models across key metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_df = summary_df.melt(
        id_vars=["Model"], 
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric", 
        value_name="Score"
    )
    
    plt.figure(figsize=(12, 6), dpi=300)
    palette = sns.color_palette("Set2", 5)
    ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Benchmarking Machine Learning Models Across Key Metrics", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Classification Model", fontsize=12, weight='bold')
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12, weight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=20, ha='right', fontsize=10, weight='semibold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison plot to: {save_path}")


def plot_feature_importance(
    model: Any, 
    feature_names: List[str], 
    top_n: int = 25, 
    save_path: str = "reports/figures/feature_importance.png"
):
    """
    Plots feature importance from tree-based estimators.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf', None), 'feature_importances_'):
        importances = model.named_steps['clf'].feature_importances_
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
        importances = model.estimators_[0].feature_importances_
    else:
        print("Model does not expose feature_importances_. Skipping plot.")
        return

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 8), dpi=300)
    colors = sns.color_palette("viridis", len(feat_df))
    bars = plt.barh(feat_df['Feature'], feat_df['Importance'], color=colors, edgecolor='none', height=0.7)
    
    plt.title(f"Top {top_n} Most Discriminative Features for Content Type Prediction", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Relative Feature Importance (Gini / Impurity Reduction)", fontsize=12, weight='bold')
    plt.ylabel("Feature", fontsize=12, weight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                 va='center', ha='left', fontsize=9, color='#333333', weight='semibold')
                 
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to: {save_path}")


def generate_eda_figures(df: pd.DataFrame, save_dir: str = "reports/figures"):
    """
    Generates exploratory data analysis plots for project reports and dashboard.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Content Distribution Pie & Bar
    plt.figure(figsize=(10, 4.5), dpi=300)
    type_counts = df['type'].value_counts()
    
    plt.subplot(1, 2, 1)
    colors = ['#E50914', '#221F1F']
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=colors, 
            startangle=140, explode=(0.05, 0), textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white' if colors[0]=='#221F1F' else 'black'})
    plt.title("Distribution of Movies vs TV Shows", fontsize=13, weight='bold')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='type', palette=['#E50914', '#564d4d'])
    plt.title("Total Count per Content Type", fontsize=13, weight='bold')
    plt.xlabel("Content Type", fontsize=11, weight='bold')
    plt.ylabel("Count", fontsize=11, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_content_distribution.png"), bbox_inches='tight')
    plt.close()

    # 2. Release Year Distribution by Type
    plt.figure(figsize=(11, 5), dpi=300)
    recent_df = df[df['release_year'] >= 1990]
    sns.histplot(data=recent_df, x='release_year', hue='type', multiple='stack', 
                 palette=['#E50914', '#1f77b4'], bins=30, kde=True)
    plt.title("Historical Release Year Distribution by Content Type (1990 - Present)", fontsize=13, weight='bold')
    plt.xlabel("Release Year", fontsize=11, weight='bold')
    plt.ylabel("Number of Titles", fontsize=11, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_temporal_trends.png"), bbox_inches='tight')
    plt.close()

    # 3. Rating Distribution by Type
    plt.figure(figsize=(12, 5), dpi=300)
    rating_order = df['rating'].value_counts().index[:12]
    sns.countplot(data=df[df['rating'].isin(rating_order)], x='rating', hue='type', 
                  order=rating_order, palette=['#E50914', '#1f77b4'])
    plt.title("Audience Ratings Breakdown by Content Type", fontsize=13, weight='bold')
    plt.xlabel("Audience Rating", fontsize=11, weight='bold')
    plt.ylabel("Count", fontsize=11, weight='bold')
    plt.xticks(rotation=30)
    plt.legend(title="Content Type")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_rating_distribution.png"), bbox_inches='tight')
    plt.close()
    print(f"Saved EDA visualizations to {save_dir}/")


def save_evaluation_summary(metrics_dict: Dict[str, Any], save_path: str = "reports/metrics_summary.json"):
    """
    Saves metrics summary into formatted JSON.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics summary to: {save_path}")
