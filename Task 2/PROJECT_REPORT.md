# Technical Project Report: Netflix Content Type Prediction Model
**Auspify Technologies Machine Learning Internship - Task 2**

---

## Executive Summary
This report presents the design, implementation, and rigorous evaluation of an original, production-ready Machine Learning system that predicts whether a Netflix catalog title is a **Movie** or a **TV Show** based on its metadata attributes, natural language title representations, temporal patterns, and parental ratings.

The pipeline adheres to **zero data leakage principles** by isolating text vectorization, category encoding, and scaling strictly to the training fold. Across 8,790 catalog titles, we benchmarked 9 classification paradigms using Stratified 5-Fold Cross-Validation, hyperparameter tuning via Grid Search, and ensembling. The champion **Soft Voting Ensemble (Random Forest + Gradient Boosting + Logistic Regression + Extra Trees)** achieved an **Accuracy of 97.10%**, **ROC-AUC of 0.9946**, and a **Macro F1-Score of 95.21%** on an independent held-out test set.

---

## 1. Problem Formulation & Task Scope
In digital streaming platforms, accurate classification of media assets enables automated indexing, dynamic layout rendering, metadata enrichment, and targeted recommendations. 

Task 2 from the Auspify Machine Learning Internship curriculum mandates:
1. **Step 1: Select relevant dataset features** (Exploratory Data Analysis, class balance, schema validation).
2. **Step 2: Encode categorical variables** (Leak-free transformations, temporal lag engineering, Title TF-IDF extraction).
3. **Step 3: Train classification models** (Multi-model benchmarking across tree-based, linear, and kernel-based algorithms).
4. **Step 4: Evaluate prediction performance** (Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix).
5. **Step 5: Compare model accuracy & Optimize** (Hyperparameter optimization, ensembling, and feature importance).

---

## 2. Dataset Exploration & Leak-Aware Feature Engineering

### 2.1 Raw Dataset Characteristics
- **Total Records**: 8,790 titles
- **Target Distribution**:
  - `Movie`: 6,126 titles (69.69%)
  - `TV Show`: 2,664 titles (30.31%)
- **Temporal Span**: Release years from 1925 through 2021.

### 2.2 Addressing Data Leakage
In the raw Netflix dataset, the column `duration` contains the literal unit string (`"min"` for Movies and `"Season"`/`"Seasons"` for TV Shows), and `listed_in` contains explicit phrases like `"TV Dramas"`. To build a genuine machine learning model rather than a trivial string regex matcher, our feature extractor isolates semantic signals:
1. **Title Natural Language Processing (NLP)**:
   - Character count, word length, uppercase character ratio, presence of digits/colons.
   - TF-IDF N-gram representation (unigrams and bigrams, max 60 features) capturing semantic patterns in titles.
2. **Parental Rating Taxonomy**:
   - Indicator for TV broadcast standards (`is_tv_rating` for `TV-MA`, `TV-14`, `TV-PG`, `TV-Y`, `TV-Y7`, `TV-G`).
   - Indicator for MPAA theatrical standards (`is_mpaa_rating` for `G`, `PG`, `PG-13`, `R`, `NC-17`, `NR`).
3. **Director & Production Metadata**:
   - `has_director`: Standalone films virtually always specify a director, whereas serialized TV shows frequently list no single primary director in streaming catalog metadata.
   - `director_count` and `country_count` indicators for international co-productions.
4. **Temporal Release Dynamics**:
   - Date added parsing: `added_year`, `added_month`, `added_dayofweek`, `added_quarter`.
   - Release-to-addition lag: `years_between_release_and_added`.
   - Content era flags: `is_modern_content` ($\ge 2015$), `is_vintage_content` ($< 2000$).
5. **Genre Semantics**:
   - Multi-label genre concept indicators (`drama`, `comedy`, `action`, `docuseries`, `anime`, `romance`, `horror`, `thriller`, `reality`, `sci-fi`).

Total engineered feature space: **137 numerical and vectorized features**.

---

## 3. Machine Learning Algorithms & Cross-Validation Benchmark

All candidate models were evaluated on the training partition (7,032 instances) using **Stratified 5-Fold Cross-Validation** to ensure class balance across folds.

| Model Algorithm | CV Accuracy (Mean ± Std) | CV Precision | CV Recall | CV F1 (Mean ± Std) | CV ROC-AUC | Train Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 97.37% ± 0.46% | 95.99% | 95.31% | 95.64% ± 0.79% | **0.9930** | 97.76% |
| **Logistic Regression (L2)** | 97.68% ± 0.34% | 96.07% | 96.29% | 96.18% ± 0.60% | **0.9926** | 98.13% |
| **Extra Trees Classifier** | 97.70% ± 0.29% | 96.60% | 95.78% | 96.18% ± 0.51% | **0.9923** | 98.12% |
| **Gradient Boosting** | 97.77% ± 0.36% | 96.30% | 96.34% | 96.31% ± 0.61% | **0.9919** | 99.18% |
| **Support Vector Classifier (RBF)** | 95.55% ± 0.57% | 96.19% | 88.83% | 92.36% ± 1.01% | **0.9843** | 98.06% |
| **Decision Tree** | 97.67% ± 0.32% | 96.50% | 95.78% | 96.14% ± 0.54% | **0.9798** | 97.89% |
| **AdaBoost** | 95.62% ± 0.34% | 93.44% | 92.02% | 92.71% ± 0.63% | **0.9749** | 95.62% |
| **K-Nearest Neighbors** | 90.87% ± 0.90% | 89.06% | 79.68% | 84.10% ± 1.58% | **0.9597** | 100.0% |
| **Gaussian Naive Bayes** | 70.61% ± 0.72% | 50.80% | 95.92% | 66.42% ± 0.56% | **0.9299** | 70.93% |

---

## 4. Hyperparameter Optimization & Ensembling

Grid Search with 3-fold cross-validation was conducted to identify optimal hyperparameters for top estimators:
- **Random Forest**: `n_estimators=150`, `max_depth=16`, `min_samples_split=4`, `class_weight='balanced'`.
- **Gradient Boosting**: `learning_rate=0.10`, `max_depth=4`, `n_estimators=180`.
- **Logistic Regression**: `C=0.10`, `solver='liblinear'`, `class_weight='balanced'`.

We constructed two meta-ensembles:
1. **Soft Voting Ensemble**: Weighted soft probability aggregation across Random Forest, Gradient Boosting, Logistic Regression, and Extra Trees (Weights: 2, 3, 1, 2).
2. **Stacking Ensemble**: Out-of-fold probability stacking with Logistic Regression meta-learner.

---

## 5. Independent Test Set Evaluation & Diagnostic Results

On the held-out test partition ($N = 1,758$ titles, 20% stratified):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision (PR-AUC) | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble** | **97.10%** | **96.54%** | **95.30%** | **95.12%** | **95.21%** | **0.9946** | **0.9894** | 0.1075 |
| **Gradient Boosting (Tuned)** | **97.21%** | 96.68% | 95.49% | 95.31% | 95.40% | **0.9945** | 0.9887 | **0.0840** |
| **Stacking Ensemble** | 97.16% | 96.69% | 95.14% | 95.50% | 95.32% | 0.9944 | 0.9892 | 0.1062 |
| **Random Forest (Tuned)** | 96.93% | 96.26% | 95.27% | 94.56% | 94.92% | 0.9934 | 0.9867 | 0.1125 |
| **Extra Trees Classifier** | 96.87% | 96.17% | 95.27% | 94.37% | 94.82% | 0.9921 | 0.9828 | 0.1748 |
| **Logistic Regression (Tuned)** | 96.81% | 96.28% | 94.58% | 94.93% | 94.76% | 0.9911 | 0.9822 | 0.1069 |

### 5.1 Confusion Matrix Analysis
- **True Negatives (Correctly Identified Movies)**: 1,200 / 1,225 (98.0%)
- **True Positives (Correctly Identified TV Shows)**: 507 / 533 (95.1%)
- **False Positives**: 25 (2.0%)
- **False Negatives**: 26 (4.9%)

### 5.2 Key Feature Importance Insights
1. **`has_director`**: Most influential predictor. Standalone feature films list directorship metadata with high fidelity, whereas television series frequently omit single-director credits in broad streaming catalog feeds.
2. **`is_tv_rating` / `is_mpaa_rating`**: Strong structural demarcation between theatrical MPAA certificates (PG-13, R) and television FCC guidelines (TV-MA, TV-14, TV-PG).
3. **Genre Concept Binarization (`docuseries`, `drama`, `anime`, `comedy`)**: High discriminative affinity for episodic formats.
4. **Title Lexical Dynamics & TF-IDF**: Word length and characteristic phrase tokens provide complementary predictive power.

---

## 6. System Architecture & Deployment

The system is organized into a modular production layout:
```
Task 2/
├── Dataset.csv                       # Verified source dataset
├── main.py                           # Full pipeline orchestrator
├── app.py                            # FastAPI server & REST API
├── generate_notebook.py              # Notebook generation utility
├── frontend/
│   └── index.html                    # Modern Netflix Dark UI dashboard
├── models/
│   └── content_type_pipeline.joblib  # Serialized pipeline & model bundle
├── notebooks/
│   └── Task_2_Content_Type_Prediction.ipynb  # End-to-end Jupyter Notebook
├── reports/
│   ├── metrics_summary.json          # Formatted JSON metrics benchmark
│   └── figures/                      # High-res diagnostic visualizations
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── model_comparison.png
│       ├── feature_importance.png
│       ├── eda_content_distribution.png
│       ├── eda_temporal_trends.png
│       └── eda_rating_distribution.png
└── src/
    ├── __init__.py
    ├── data_loader.py                # Ingestion, validation, cleaning
    ├── feature_engineering.py        # Leak-free Scikit-Learn transformers
    ├── models.py                     # Candidate models, CV, tuning, ensembling
    ├── evaluation.py                 # Metrics computation & plotting
    └── predict.py                    # Inference engine & CLI
```

---

## 7. Conclusion
Task 2 has been completely executed according to the exact specifications and evaluation standards of the Auspify Machine Learning Internship. The resulting system combines solid statistical rigor, leak-free feature engineering, high classification accuracy ($97.10\%$), and an interactive user experience.
