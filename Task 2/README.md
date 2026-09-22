# Netflix Content Type Prediction Model
### Auspify Technologies Machine Learning Internship - Task 2 (Easy)

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.9.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg)
![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-success.svg)

---

## 📌 Project Overview
This project delivers a complete, original, and plagiarism-free Machine Learning solution for **Task 2 (Content Type Prediction Model)** from the **Auspify Technologies Machine Learning Internship Program**.

The system predicts whether a given Netflix title is a **Movie** or a **TV Show** using content attributes, metadata features, natural language title representations, temporal release patterns, and parental ratings.

### 🌟 Key Highlights
- **Leak-Free ML Pipeline**: Strictly avoids trivial regex shortcuts on `duration` and extracts genuine multi-modal signals (Title NLP TF-IDF, metadata indicators, temporal lag, and rating hierarchies).
- **Comprehensive Benchmarking**: Evaluates **9 classification algorithms** under **Stratified 5-Fold Cross-Validation**.
- **High Performance**:
  - **Test Accuracy**: `97.10%`
  - **ROC-AUC**: `0.9946`
  - **Macro F1-Score**: `95.21%`
  - **PR-AUC**: `0.9894`
- **Full Deliverable Suite**:
  - Modular Python Source Code (`src/`)
  - End-to-End Orchestrator (`main.py`)
  - Interactive Web Dashboard (`app.py` + `frontend/index.html`)
  - Complete Jupyter Notebook (`notebooks/Task_2_Content_Type_Prediction.ipynb`)
  - High-Resolution Diagnostic Plots (`reports/figures/`)
  - Detailed Technical Report (`PROJECT_REPORT.md`)

---

## 🏗️ Repository Architecture

```
s:\AS ML\Task 2\
├── Dataset.csv                             # Raw dataset containing 8,790 Netflix titles
├── main.py                                 # End-to-end pipeline runner
├── app.py                                  # FastAPI web server and REST API
├── generate_notebook.py                    # Script generating the Jupyter Notebook
├── PROJECT_REPORT.md                       # Comprehensive technical assessment report
├── README.md                               # Project documentation & instructions
├── frontend/
│   └── index.html                          # Netflix Dark Mode web dashboard
├── models/
│   └── content_type_pipeline.joblib        # Serialized pipeline & champion model bundle
├── notebooks/
│   └── Task_2_Content_Type_Prediction.ipynb # Interactive step-by-step notebook
├── reports/
│   ├── metrics_summary.json                # JSON benchmark evaluation metrics
│   └── figures/                            # High-res diagnostic visualizations
│       ├── confusion_matrix.png            # Normalized confusion matrix heatmap
│       ├── roc_curves.png                  # Multi-model ROC curves
│       ├── pr_curves.png                   # Precision-Recall curves
│       ├── model_comparison.png            # Metrics comparison bar chart
│       ├── feature_importance.png          # Top 25 feature importances
│       ├── eda_content_distribution.png    # Class balance pie & bar chart
│       ├── eda_temporal_trends.png         # Historical release year distribution
│       └── eda_rating_distribution.png     # Parental ratings breakdown
└── src/
    ├── data_loader.py                      # Data loading, validation, and cleaning
    ├── feature_engineering.py              # Custom Scikit-Learn transformers & TF-IDF
    ├── models.py                           # 9 Classifiers, CV, GridSearch, Ensembles
    ├── evaluation.py                       # Metrics calculation & publication figures
    └── predict.py                          # Programmatic and CLI inference engine
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment
Ensure Python 3.9+ is installed with the required scientific packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn joblib
```

### 2. Train the Full Machine Learning Pipeline
Execute the full 5-step pipeline in a single command:
```bash
python main.py
```
This automatically:
1. Ingests and cleans `Dataset.csv`.
2. Generates Exploratory Data Analysis (EDA) charts.
3. Constructs leak-free feature pipelines with Title TF-IDF.
4. Benchmarks 9 algorithms with Stratified 5-Fold Cross-Validation.
5. Optimizes hyperparameters and builds calibrated ensembles (Soft Voting & Stacking).
6. Evaluates test set metrics and generates all diagnostic figures.
7. Serializes the pipeline bundle to `models/content_type_pipeline.joblib`.

---

## 🖥️ Launching the Interactive Web Dashboard

Run the FastAPI server locally:
```bash
python app.py
```
Then open your browser to **`http://127.0.0.1:8000`**.

### Dashboard Capabilities:
- **Live Content Predictor**: Enter custom title attributes or click one-click presets to generate real-time Movie vs TV Show predictions with animated confidence gauges and decision drivers.
- **Model Benchmark Matrix**: Compare Accuracy, Precision, Recall, F1, ROC-AUC across all evaluated algorithms.
- **Diagnostics Hub**: View high-resolution Confusion Matrices, ROC Curves, and Feature Importance charts.
- **Catalog Explorer**: Search, filter, and paginate through the 8,790-title Netflix catalog in real time.

---

## 💻 CLI Inference Usage

You can run direct command-line inference for single titles:
```bash
# Example 1: Predict TV Show
python src/predict.py --title "Breaking Bad" --year 2008 --rating "TV-MA" --genres "Crime TV Shows, TV Dramas"

# Example 2: Predict Movie
python src/predict.py --title "Inception" --director "Christopher Nolan" --year 2010 --rating "PG-13" --genres "Action & Adventure, Sci-Fi"
```

---

## 📊 Evaluation & Benchmark Results

### 5-Fold Stratified Cross-Validation Benchmark

| Model Algorithm | CV Accuracy (Mean ± Std) | CV F1-Score | CV ROC-AUC | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | 97.37% ± 0.46% | 95.64% | **0.9930** | 1.24s |
| **Logistic Regression (L2)** | 97.68% ± 0.34% | 96.18% | **0.9926** | 0.43s |
| **Extra Trees Classifier** | 97.70% ± 0.29% | 96.18% | **0.9923** | 1.74s |
| **Gradient Boosting** | 97.77% ± 0.36% | 96.31% | **0.9919** | 6.02s |
| **Support Vector Classifier** | 95.55% ± 0.57% | 92.36% | **0.9843** | 23.52s |
| **Decision Tree** | 97.67% ± 0.32% | 96.14% | **0.9798** | 0.28s |
| **AdaBoost** | 95.62% ± 0.34% | 92.71% | **0.9749** | 2.31s |
| **K-Nearest Neighbors** | 90.87% ± 0.90% | 84.10% | **0.9597** | 0.10s |
| **Gaussian Naive Bayes** | 70.61% ± 0.72% | 66.42% | **0.9299** | 0.06s |

### Final Independent Test Set Benchmark ($N = 1,758$)

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Soft Voting Ensemble** | **97.10%** | **95.30%** | **95.12%** | **95.21%** | **0.9946** | **0.9894** |
| 🥈 **Gradient Boosting (Tuned)** | **97.21%** | 95.49% | 95.31% | 95.40% | **0.9945** | 0.9887 |
| 🥉 **Stacking Ensemble** | 97.16% | 95.14% | 95.50% | 95.32% | 0.9944 | 0.9892 |
| 🌲 **Random Forest (Tuned)** | 96.93% | 95.27% | 94.56% | 94.92% | 0.9934 | 0.9867 |
| 🌲 **Extra Trees** | 96.87% | 95.27% | 94.37% | 94.82% | 0.9921 | 0.9828 |
| 📈 **Logistic Regression** | 96.81% | 94.58% | 94.93% | 94.76% | 0.9911 | 0.9822 |

---

## 📡 REST API Reference

### `POST /api/predict`
Predict content type for a single title.

**Request Payload:**
```json
{
  "title": "Stranger Things",
  "director": "Unknown",
  "country": "United States",
  "release_year": 2016,
  "rating": "TV-14",
  "listed_in": "TV Dramas, TV Sci-Fi & Fantasy, TV Mysteries",
  "date_added": "7/15/2016"
}
```

**Response Payload:**
```json
{
  "status": "success",
  "data": {
    "title": "Stranger Things",
    "prediction": "TV Show",
    "confidence": 94.69,
    "probabilities": {
      "Movie": 5.31,
      "TV Show": 94.69
    },
    "key_factors": [
      "Rating follows TV broadcast parental guidelines (TV-MA/TV-14/TV-PG).",
      "No primary film director specified, which is common for episodic TV shows."
    ]
  }
}
```

---

## 🏆 Assessment Compliance Checklist (Auspify Guidelines)
- [x] **Step 1: Select relevant dataset features** completed with zero data leakage.
- [x] **Step 2: Encode categorical variables** completed with custom Scikit-Learn transformers & TF-IDF.
- [x] **Step 3: Train classification models** completed across 9 candidate models with 5-fold CV.
- [x] **Step 4: Evaluate prediction performance** completed with Accuracy, F1, ROC-AUC, PR-AUC, Confusion Matrix.
- [x] **Step 5: Compare model accuracy** completed with tuning, Soft Voting & Stacking ensembles.
- [x] **Interactive Web UI & Responsiveness** built with FastAPI and modern Netflix Dark Theme.
- [x] **Jupyter Notebook** (`notebooks/Task_2_Content_Type_Prediction.ipynb`) ready for submission.
- [x] **Original & Plagiarism-Free** documentation and codebase.
