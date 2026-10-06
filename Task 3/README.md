# Netflix Audience Rating Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> An end-to-end, production-grade Machine Learning web application predicting parental audience rating categories (`TV-MA`, `TV-14`, `R`, `PG-13`, `TV-PG`, `TV-Y7`, `TV-Y`, etc.) for Netflix content using metadata attributes — powered by a **Voting Ensemble (Random Forest + Gradient Boosting + ExtraTrees)** classifier with TF-IDF enrichment on descriptions and cast.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Specifications](#-dataset-specifications)
3. [Feature Engineering Pipeline](#-feature-engineering-pipeline)
4. [Machine Learning Models & Tuning](#-machine-learning-models--tuning)
5. [Evaluation Results](#-evaluation-results)
6. [Application Screenshots](#-application-screenshots)
7. [System Architecture](#-system-architecture)
8. [API Documentation](#-api-documentation)
9. [Setup & Execution Guide](#-setup--execution-guide)
10. [Project Checklist](#-project-checklist)

---

## 🎯 Project Overview

This project builds a **multi-class machine learning classification system** to predict the audience rating category of Netflix titles using content metadata. The pipeline covers the full spectrum from raw data ingestion to a deployed React dashboard with a FastAPI REST backend.

**Key objectives:**
- Classify content across **14 rating categories** (`TV-MA`, `TV-14`, `TV-PG`, `R`, `PG-13`, `TV-Y7`, `TV-Y`, `PG`, `TV-G`, `NR`, `G`, `TV-Y7-FV`, `NC-17`, `UR`)
- Apply **rich feature engineering** including temporal, geographic, genre, text (TF-IDF), and interaction features
- Train and compare **5 ML architectures**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and a soft-voting Ensemble
- Tune hyperparameters via **5-fold cross-validation** using `RandomizedSearchCV`
- Evaluate with **Accuracy, Weighted F1, Macro F1, Precision, Recall, Confusion Matrix**
- Serve predictions via a **FastAPI REST API** with Swagger docs
- Deliver a **React + Tailwind dashboard** for interactive real-time inference and analytics

---

## 📊 Dataset Specifications

The project uses `Dataset.csv` (8,790 Netflix catalog records, 10 columns, 0 missing values):

| Column | Role | Treatment |
| :--- | :--- | :--- |
| `show_id` | Identifier | **Excluded** — arbitrary index, no predictive signal |
| `title` | String | **Excluded** — prevents title memorization, preserves generalization |
| `type` | Content format | Binary (`Movie` / `TV Show`), one-hot encoded |
| `director` | Crew attribute | Director popularity score + presence flag |
| `cast` | Crew attribute | **TF-IDF vectorized** (top 30 terms) for semantic signal |
| `country` | Geography | Country count, primary country, regional flags |
| `date_added` | Date | Parsed into `added_year`, `added_month`; `content_age` derived |
| `release_year` | Time | Numerical; `release_decade`, `is_recent` derived |
| `duration` | Runtime | Split into `duration_value` (numeric) + `duration_type` (`min`/`Season`) |
| `listed_in` | Genres | 42 binary genre flags + 12 semantic group flags + `genre_count` |
| `description` | Synopsis | **TF-IDF vectorized** (top 50 bigram terms) for semantic enrichment |
| `rating` | **TARGET** | 14 discrete multi-class labels |

**Dataset Metrics:**
- Total Records: **8,790**  
- Missing Values: **0**  
- Train Split: **7,032 (80%)** | Test Split: **1,758 (20%)**  
- Movies: **6,126 (69.7%)** | TV Shows: **2,664 (30.3%)**

---

## ⚙️ Feature Engineering Pipeline

The custom `NetflixPreprocessor` in [`ml/preprocess.py`](ml/preprocess.py) engineers **280+ features** across 6 categories:

### 1. Temporal & Age Features
- `added_year`, `added_month` — from `date_added`
- `release_year`, `content_age` (`added_year - release_year`)
- `release_decade` (e.g. 1990, 2010), `is_recent` (released ≥ 2015)

### 2. Runtime & Format Features
- `duration_value` — numeric runtime (e.g. `90`, `3`)
- `duration_type` — `min` vs `Season`
- `movie_duration_mins`, `season_count` — split by format
- `is_movie`, `is_tv_show` — binary content type flags

### 3. Geographic & Regional Features
- `country_count` — number of co-producing countries
- `primary_country` — top 16 global territories + `Other`
- `international_content`, `us_primary`, `india_primary` — regional flags

### 4. Genre & Category Vectorization
- `genre_count` — total genres per title
- **42 individual binary genre columns** (`genre_Action & Adventure`, `genre_Horror Movies`, etc.)
- **12 semantic genre group flags** — aggregated categories with strong rating signal:
  - `is_kids_content`, `is_adult_drama`, `is_comedy`, `is_horror`
  - `is_documentary`, `is_international`, `is_animation`, `is_reality`
  - `is_romantic`, `is_scifi`, `is_family`, `is_teen`

### 5. Director & Cast Features
- `has_director` — binary presence flag
- `director_popularity` — normalized frequency (fraction of catalog)
- `cast_size` — number of credited cast members

### 6. Text Semantic Features (TF-IDF)
- **Description TF-IDF** — top 50 unigram/bigram terms from synopsis text (English stop-words removed)
- **Cast TF-IDF** — top 30 actor name terms for actor-genre association signals

### Preprocessing Safety
- `StandardScaler` for all numerical features
- `OneHotEncoder(handle_unknown='ignore')` for categorical columns — zero inference errors on unseen values
- All transformers fitted on training data only (no data leakage)

---

## 🧠 Machine Learning Models & Tuning

### Models Trained

| # | Model | Type | Tuning |
|---|-------|------|--------|
| 1 | **Logistic Regression** | Linear (Baseline) | `class_weight='balanced'`, `C=0.5`, `max_iter=3000` |
| 2 | **Decision Tree** | Non-linear (Baseline & Tuned) | `RandomizedSearchCV` — 20 iterations × 5-fold CV |
| 3 | **Random Forest** | Bagged Ensemble (Baseline & Tuned) | `RandomizedSearchCV` — 20 iterations × 5-fold CV |
| 4 | **Gradient Boosting** | Boosted Ensemble (Tuned) | `RandomizedSearchCV` — 20 iterations × 5-fold CV |
| 5 | **ExtraTrees** | Randomized Ensemble | 300 estimators, `balanced` weights |
| ★ | **Voting Ensemble (RF + GB + ET)** | Soft-Voting Champion | Combines top 3 tuned estimators |

### Hyperparameter Search Spaces

**Decision Tree** (`RandomizedSearchCV`, 20 iterations):
```
criterion:          ['gini', 'entropy']
max_depth:          [8, 10, 12, 15, 20, None]
min_samples_split:  [2, 5, 10]
min_samples_leaf:   [1, 2, 4]
max_features:       ['sqrt', 'log2', None]
```

**Random Forest** (`RandomizedSearchCV`, 20 iterations):
```
n_estimators:       [200, 300, 400]
max_depth:          [20, 30, 40, None]
min_samples_split:  [2, 5]
min_samples_leaf:   [1, 2]
max_features:       ['sqrt', 'log2']
class_weight:       [None, 'balanced']
```

**Gradient Boosting** (`RandomizedSearchCV`, 20 iterations):
```
n_estimators:       [200, 300, 400]
learning_rate:      [0.05, 0.08, 0.1, 0.15]
max_depth:          [4, 5, 6, 7]
subsample:          [0.8, 0.9, 1.0]
max_features:       ['sqrt', 'log2']
min_samples_split:  [2, 5]
```

**Optimization metric:** Weighted F1-Score (handles class imbalance; consistent with 14-class distribution)

---

## 📈 Evaluation Results

All metrics computed on the **held-out 20% test split (1,758 samples)**. No data leakage — transformers fitted on training data only.

> **Note:** Results reflect the trained model stored in `models/metrics.json`. Re-run `python -m ml.train` to regenerate.

| Model | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Voting Ensemble (RF+GB+ET)** | **see metrics.json** | — | — | — | **★ Champion** |
| Gradient Boosting (Tuned) | see metrics.json | — | — | — | Tuned Booster |
| Random Forest (Tuned) | see metrics.json | — | — | — | Tuned Ensemble |
| Extra Trees | see metrics.json | — | — | — | Randomized Trees |
| Decision Tree (Tuned) | see metrics.json | — | — | — | Pruned Tree |
| Random Forest (Baseline) | ~54.8% | — | — | — | Baseline |
| Logistic Regression (Baseline) | ~54.2% | — | — | — | Linear Baseline |
| Decision Tree (Baseline) | ~47.6% | — | — | — | Unpruned Baseline |

Full per-class precision, recall, F1, support, and confusion matrix are persisted in [`models/metrics.json`](models/metrics.json).

### Class Imbalance Handling
- Top 2 classes (`TV-MA`, `TV-14`) account for **61%** of all records
- Rare classes (`UR`, `NC-17`, `TV-Y7-FV`) have < 10 samples each
- Mitigation: stratified train/test split, `class_weight='balanced'` for LR & ExtraTrees, weighted F1 as optimization metric

---

## 🖥️ Application Screenshots

### 1. Dashboard
*KPI cards, rating distribution charts, content type breakdown, and model leaderboard.*
![Dashboard Screenshot](screenshots/01_dashboard.png)

---

### 2. Rating Prediction
*Interactive form with quick-fill presets, cast & description inputs, live confidence score, and full probability distribution.*
![Prediction Result](screenshots/03_prediction_result.png)

---

### 3. Model Comparison
*Pre vs. post-tuning delta cards, comparative accuracy bar charts, and evaluation metrics table.*
![Model Comparison](screenshots/04_model_comparison.png)

---

### 4. Analytics & Diagnostics
*14-class Confusion Matrix heatmap and Top Feature Importances (from RF component of ensemble).*
![Analytics Page](screenshots/06_analytics_confusion_matrix.png)

---

### 5. About / Documentation
*Full methodology, 5-model architecture, feature engineering rationale, and operational notes.*
![About Page](screenshots/07_about.png)

---

## 🏗️ System Architecture

```
netflix-rating-classifier/
├── frontend/                         # React + Vite + Tailwind CSS Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   └── Navbar.jsx            # Navigation bar with live API status
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx         # KPIs, rating distribution, model leaderboard
│   │   │   ├── RatingPrediction.jsx  # Real-time inference form (with cast/description)
│   │   │   ├── ModelComparison.jsx   # All 5 models benchmarked side-by-side
│   │   │   ├── Analytics.jsx         # Confusion matrix, feature importances, trends
│   │   │   └── About.jsx             # Methodology, architecture, dataset details
│   │   ├── services/
│   │   │   └── api.js                # Axios client — health, predict, analytics, model-info
│   │   ├── App.jsx                   # Root component with tab routing
│   │   ├── index.css                 # Base styles, dashboard card system, animations
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/                          # FastAPI Python REST API
│   ├── main.py                       # App factory, CORS, routes, static file serving
│   ├── schemas.py                    # Pydantic v2 request/response models
│   └── services/
│       ├── predictor.py              # Inference engine — loads pkl, runs transform+predict
│       └── analytics.py             # Dataset statistics & model metrics aggregation
│
├── ml/                               # Modular Machine Learning Pipeline
│   ├── preprocess.py                 # NetflixPreprocessor (TF-IDF, genre groups, scaler)
│   ├── train.py                      # Master pipeline — trains, tunes, evaluates all models
│   ├── tune.py                       # RandomizedSearchCV helpers (DT / RF / GB)
│   └── evaluate.py                   # Accuracy, F1, classification report, confusion matrix
│
├── data/
│   └── Dataset.csv                   # Source dataset (8,790 records, 10 columns)
│
├── models/
│   ├── best_model.pkl                # Serialized champion model (VotingClassifier)
│   ├── preprocessor.pkl              # Fitted NetflixPreprocessor (TF-IDF + scaler + encoder)
│   └── metrics.json                  # Full evaluation report for all models
│
├── notebooks/
│   └── netflix_rating_analysis.ipynb # EDA, feature analysis, and modeling walkthrough
│
├── screenshots/                      # Application UI screenshots
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔌 API Documentation

The FastAPI backend auto-generates interactive Swagger UI at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Service health, model load status |
| `GET` | `/api/model-info` | All model metrics, tuning parameters, feature importances |
| `GET` | `/api/analytics` | Catalog distributions, trends, confusion matrix data |
| `POST` | `/api/predict` | Predict rating for provided content metadata |

### Sample `POST /api/predict` Request
```json
{
  "type": "Movie",
  "title": "The Dark Horizon",
  "director": "Christopher Nolan",
  "cast": "Leonardo DiCaprio, Cillian Murphy",
  "country": "United States",
  "date_added": "2021-09-25",
  "release_year": 2020,
  "duration": "148 min",
  "listed_in": "Action & Adventure, Sci-Fi & Fantasy",
  "description": "An action-packed sci-fi thriller about heroes battling interdimensional threats."
}
```

### Sample `POST /api/predict` Response
```json
{
  "predicted_rating": "PG-13",
  "confidence": 0.51,
  "model_used": "VotingClassifier",
  "rating_category_info": {
    "name": "PG-13 (Parents Strongly Cautioned)",
    "badge": "13+",
    "level": "Teens 13+",
    "color": "#8B5CF6",
    "description": "Some material may be inappropriate for pre-teen children under the age of 13."
  },
  "probabilities": {
    "PG-13": 0.51,
    "TV-14": 0.20,
    "TV-MA": 0.17,
    "PG": 0.06,
    "R": 0.04
  },
  "top_contributing_features": [
    { "feature": "Genre: Action & Adventure", "impact": "Active Category" },
    { "feature": "Genre: Sci-Fi & Fantasy", "impact": "Active Category" },
    { "feature": "Content Type: Movie", "impact": "Format" },
    { "feature": "Duration: 148 min", "impact": "Runtime" }
  ]
}
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+
- Node.js v18+ and npm

### 1. Install Python Dependencies
```bash
cd "s:/AS ML/Task 3"
pip install -r requirements.txt
```

### 2. Train Models (Optional — pre-trained artifacts included)
```bash
# Full pipeline: preprocess → train → tune → evaluate → save
python -m ml.train
```
> ⏱ Expected runtime: ~15–30 min (Gradient Boosting tuning is the bottleneck). Pre-trained `models/best_model.pkl` and `models/preprocessor.pkl` are already included.

### 3. Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
> API available at: `http://127.0.0.1:8000` | Swagger docs: `http://127.0.0.1:8000/docs`

### 4. Start the React Frontend (Dev Mode)
```bash
cd "s:/AS ML/Task 3/frontend"
npm install
npm run dev
```
> Frontend available at: `http://127.0.0.1:5173`

### 5. Build Production Bundle (Served by FastAPI)
```bash
cd "s:/AS ML/Task 3/frontend"
npm run build
```
> The built frontend is automatically served at `http://127.0.0.1:8000` by the FastAPI static file mount.

---

## ✅ Project Checklist

- [x] **Source code** — Clean, modular Python (ML pipeline) and React (frontend) codebase
- [x] **5 ML Models** — Logistic Regression, Decision Tree (×2), Random Forest (×2), Gradient Boosting, ExtraTrees, Voting Ensemble
- [x] **Feature engineering** — 280+ features including TF-IDF on descriptions and cast
- [x] **Hyperparameter tuning** — `RandomizedSearchCV` with 5-fold cross-validation on all tree models
- [x] **Evaluation metrics** — Accuracy, Weighted F1, Macro F1, Precision, Recall, Confusion Matrix, Classification Report
- [x] **Saved artifacts** — `models/best_model.pkl`, `models/preprocessor.pkl`, `models/metrics.json`
- [x] **FastAPI REST API** — 4 endpoints with Swagger UI autodocs
- [x] **React Dashboard** — 5 pages: Dashboard, Rating Prediction, Model Comparison, Analytics, About
- [x] **Prediction form** — Supports type, director, cast, country, date, release year, duration, genres, description
- [x] **Screenshots** — Captured in `screenshots/`
- [x] **Jupyter Notebook** — `notebooks/netflix_rating_analysis.ipynb`
- [x] **README** — Full documentation, architecture, API reference, setup guide

---

*Netflix Audience Rating Classifier — Multi-class ML pipeline with Voting Ensemble, TF-IDF enrichment, and React dashboard.*
