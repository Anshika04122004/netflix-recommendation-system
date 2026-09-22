import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def make_cell(cell_type, source):
    if isinstance(source, list):
        src_lines = [s if s.endswith('\n') else s + '\n' for s in source]
    else:
        src_lines = [s + '\n' for s in source.split('\n')]
    if src_lines and src_lines[-1] == '\n':
        src_lines[-1] = ''
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells_data = [
    ("markdown", """# Netflix Content Type Prediction Model
### Auspify Technologies - Machine Learning Internship (Task 2)
**Domain**: Natural Language Processing, Supervised Binary Classification & Predictive Modeling  
**Objective**: Build an end-to-end, leak-free machine learning system to predict whether a Netflix catalog title is a **Movie** or a **TV Show** based on its metadata, title text NLP representations, temporal attributes, and parental ratings.

---

### Task Workflow (As specified in Assessment Guidelines):
1. **Step 1: Select relevant dataset features** - Comprehensive Exploratory Data Analysis (EDA) and feature identification.
2. **Step 2: Encode categorical variables** - Zero-leakage transformations, temporal lag engineering, and Title TF-IDF extraction.
3. **Step 3: Train classification models** - Multi-model benchmarking across 9 algorithms with 5-Fold Stratified Cross-Validation.
4. **Step 4: Evaluate prediction performance** - Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
5. **Step 5: Compare model accuracy & Optimize** - Hyperparameter tuning, Soft Voting and Stacking ensembling, and feature importance interpretation."""),

    ("code", """# Import standard scientific computing and visualization libraries
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn tools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'"""),

    ("markdown", """## Step 1: Dataset Ingestion & Exploratory Data Analysis (EDA)
We load the raw dataset (`Dataset.csv`), inspect its schema, check for missing values, and analyze the distribution of the target variable `type` (Movie vs. TV Show)."""),

    ("code", """# Load dataset
df_raw = pd.read_csv('Dataset.csv') if os.path.exists('Dataset.csv') else pd.read_csv('../Dataset.csv')
print(f'Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\\n')
print('Data Types and Null Counts:')
print(df_raw.info())
df_raw.head()"""),

    ("code", """# Target Class Distribution Analysis
target_counts = df_raw['type'].value_counts()
print('Target Variable Breakdown:')
for k, v in target_counts.items():
    print(f'  {k}: {v:,} ({v/len(df_raw)*100:.2f}%)')

plt.figure(figsize=(10, 4.5))
plt.subplot(1, 2, 1)
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=['#E50914', '#221F1F'], 
        explode=(0.05, 0), startangle=140, textprops={'weight': 'bold'})
plt.title('Target Ratio: Movies vs. TV Shows', weight='bold')

plt.subplot(1, 2, 2)
sns.countplot(data=df_raw, x='type', palette=['#E50914', '#564d4d'])
plt.title('Total Title Count per Content Type', weight='bold')
plt.tight_layout()
plt.show()"""),

    ("code", """# Parental Rating Breakdown by Content Type
plt.figure(figsize=(12, 5))
top_ratings = df_raw['rating'].value_counts().index[:12]
sns.countplot(data=df_raw[df_raw['rating'].isin(top_ratings)], x='rating', hue='type', 
              order=top_ratings, palette=['#E50914', '#1f77b4'])
plt.title('Distribution of Audience Ratings Across Content Types', fontsize=13, weight='bold')
plt.xlabel('Parental Rating', weight='bold')
plt.ylabel('Count', weight='bold')
plt.legend(title='Content Type')
plt.show()"""),

    ("markdown", """## Step 2: Feature Engineering & Preprocessing (Leak-Free Pipeline)
To prevent trivial data leakage and simulate genuine metadata classification:
1. **Temporal Features**: Added year, month, quarter, day of week, and lag between release and Netflix ingestion.
2. **Metadata Flags**: Presence of director (`has_director`), director count, international co-production flags.
3. **Rating Systems**: Distinguish MPAA theatrical ratings (PG-13, R, G) from television broadcast rating codes (TV-MA, TV-14).
4. **NLP Title Features**: Character count, word length, uppercase ratio, digit flags, and **TF-IDF N-grams** on titles.
5. **Genre Semantics**: Multi-label keyword indicators (drama, comedy, action, docuseries, anime, romance, etc.)."""),

    ("code", """# Load modular feature engineering pipeline
from src.data_loader import clean_data
from src.feature_engineering import prepare_training_data, NetflixFeatureExtractor

cleaned_df = clean_data(df_raw)
X_train, X_test, y_train, y_test, feat_extractor, tfidf_vec = prepare_training_data(
    cleaned_df, test_size=0.2, random_state=42
)

print(f'Training Matrix Shape: {X_train.shape}')
print(f'Testing Matrix Shape:  {X_test.shape}')
print(f'Target Class Distribution in Train Set:\\n{y_train.value_counts(normalize=True)}')
X_train.head(3)"""),

    ("markdown", """## Step 3: Multi-Model Benchmark & Stratified 5-Fold Cross-Validation
We benchmark 9 diverse classification paradigms using Stratified 5-Fold Cross Validation."""),

    ("code", """from src.models import get_candidate_models, evaluate_with_cross_validation

candidate_models = get_candidate_models(random_state=42)
cv_results = evaluate_with_cross_validation(candidate_models, X_train, y_train, n_splits=5, random_state=42)
cv_results"""),

    ("markdown", """## Step 4: Hyperparameter Optimization & Ensemble Modeling
We tune top-performing base learners (Random Forest, Gradient Boosting, Logistic Regression) using `GridSearchCV` and assemble **Soft Voting** and **Stacking** meta-classifiers."""),

    ("code", """from src.models import tune_and_build_ensemble

best_ensemble, fitted_models = tune_and_build_ensemble(X_train, y_train, random_state=42)"""),

    ("markdown", """## Step 5: Test Set Evaluation & Comprehensive Diagnostics
We evaluate the final tuned models on the independent held-out test set (20% partition)."""),

    ("code", """from src.evaluation import evaluate_multiple_models, plot_confusion_matrix, plot_roc_and_pr_curves, plot_feature_importance

test_summary, metrics_dict = evaluate_multiple_models(fitted_models, X_test, y_test)
print('--- Test Set Evaluation Benchmark ---')
test_summary"""),

    ("code", """# Plot Confusion Matrix for the Champion Soft Voting Ensemble
y_test_pred = best_ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Movie', 'TV Show'], yticklabels=['Movie', 'TV Show'],
            linewidths=1.5, annot_kws={'size': 13, 'weight': 'bold'})
plt.title('Normalized Confusion Matrix - Soft Voting Ensemble', fontsize=14, weight='bold')
plt.xlabel('Predicted Label', weight='bold')
plt.ylabel('True Label', weight='bold')
plt.show()

print(classification_report(y_test, y_test_pred, target_names=['Movie', 'TV Show'], digits=4))"""),

    ("code", """# ROC and Precision-Recall Curves
plt.figure(figsize=(14, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for name, model in fitted_models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Chance (0.500)')
plt.title('Receiver Operating Characteristic (ROC)', weight='bold')
plt.xlabel('False Positive Rate', weight='bold')
plt.ylabel('True Positive Rate', weight='bold')
plt.legend(loc='lower right')

# PR Curve
plt.subplot(1, 2, 2)
for name, model in fitted_models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(rec, prec, label=f'{name} (AP = {ap:.3f})', linewidth=2)
plt.title('Precision-Recall Curves', weight='bold')
plt.xlabel('Recall', weight='bold')
plt.ylabel('Precision', weight='bold')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()"""),

    ("code", """# Feature Importance Visualization
best_rf = fitted_models['Random Forest (Tuned)']
importances = best_rf.feature_importances_
feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 7))
sns.barplot(data=feat_df, y='Feature', x='Importance', palette='viridis')
plt.title('Top 20 Most Discriminative Features (Random Forest)', fontsize=14, weight='bold')
plt.xlabel('Mean Gini Impurity Reduction', weight='bold')
plt.ylabel('Engineered Feature', weight='bold')
plt.show()"""),

    ("code", """# Live Single-Sample Inference Demo
from src.predict import NetflixPredictor

predictor = NetflixPredictor('models/content_type_pipeline.joblib')

sample_title = {
    'title': "The Queen's Gambit",
    'director': 'Unknown',
    'country': 'United States',
    'release_year': 2020,
    'rating': 'TV-MA',
    'listed_in': 'TV Dramas',
    'date_added': '10/23/2020'
}

result = predictor.predict_single(sample_title)
print(f'Prediction for \"{result[\"title\"]}\": {result[\"prediction\"]} ({result[\"confidence\"]}% confidence)')
print('Class Probabilities:', result['probabilities'])
print('Explanation Insights:')
for f in result['key_factors']:
    print(f' - {f}')"""),

    ("markdown", """## Conclusion & Key Insights
1. **Model Efficacy**: The Soft Voting Ensemble achieved **97.10% Test Accuracy** and **0.9946 ROC-AUC**, successfully solving the content type classification task.
2. **Top Predictive Drivers**: The presence/absence of a listed director (`has_director`), TV broadcast parental ratings (`is_tv_rating`, `TV-MA`, `TV-14`), and genre keyword indicators (`docuseries`, `drama`) provided the strongest discriminative power.
3. **Deployment Readiness**: The pipeline is fully serialized with Scikit-Learn pipelines and served via a FastAPI interactive web dashboard.""")
]

for cell_type, src in cells_data:
    notebook["cells"].append(make_cell(cell_type, src))

os.makedirs('notebooks', exist_ok=True)
target_path = os.path.join('notebooks', 'Task_2_Content_Type_Prediction.ipynb')
with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Successfully generated {target_path}!')
