# 🎬 Netflix Content Segmentation & Semantic Clustering Pipeline (Task 4)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering%20%26%20PCA-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Auspify Technologies Machine Learning Internship Assessment — Task 4 (Medium)**  
> **Domain:** Unsupervised Machine Learning, High-Dimensional Latent Projections, Pattern Discovery, & Business Intelligence  
> **Dataset:** 8,790 Netflix Titles (`Dataset.csv`)

---

## 🌟 Executive Summary & Project Overview

This project provides an end-to-end, production-grade **Unsupervised Machine Learning Pipeline and Interactive Web Application** for **Task 4: Netflix Content Segmentation**.

The system ingests the raw catalog of **8,790 titles**, transforms them into a normalized **61-dimensional feature matrix** (capturing multi-label genres, duration semantics, parental maturity rating tiers, and global market regions), benchmarks four distinct clustering paradigms (**K-Means**, **Agglomerative Hierarchical (Ward)**, **Gaussian Mixture Models (GMM)**, and **DBSCAN**), evaluates optimal cluster counts using the **Elbow Method (WCSS)**, **Silhouette Analysis**, and **Davies-Bouldin Index**, and segments the catalog into **6 natural behavioral content archetypes**.

---

## 📋 Task 4 Step-by-Step Specifications

| Step | Assessment Requirement | Implementation & Output |
| :--- | :--- | :--- |
| **Step 1** | **Prepare Numerical & Categorical Features** | Multi-label binarization of 42 genres, duration normalization (movies in minutes, TV shows in seasons/episode-minutes), maturity rating tiering (Kids, Family, Teen, Adult), market region mapping, standard scaling via `StandardScaler`. |
| **Step 2** | **Apply Clustering Algorithms** | Evaluated $K \in [2, 10]$. Benchmarked K-Means vs Agglomerative vs GMM vs DBSCAN. Selected **K-Means ($K=6$)** with highest Silhouette score (0.1583) and lowest Davies-Bouldin index (1.7052). |
| **Step 3** | **Identify Content Groups** | Discovered 6 distinct content personas with distinct demographic targets, runtime distributions, and dominant genres. |
| **Step 4** | **Visualize Clusters** | 2D and 3D PCA projection latent maps, cluster volume breakdown charts, and interactive canvas exploration. |
| **Step 5** | **Interpret Cluster Characteristics** | Formulated executive business insights, catalog gap analysis, and created a real-time title simulator for nearest-neighbor content matching. |

---

## 🎭 The 6 Discovered Content Archetypes

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                NETFLIX CONTENT UNIVERSE                                │
├───────────┬───────────────────────────────────────────┬─────────────┬──────────────────┤
│ Cluster   │ Content Archetype                         │ Volume (%)  │ Target Audience  │
├───────────┼───────────────────────────────────────────┼─────────────┼──────────────────┤
│ Cluster 0 │ Mainstream Comedy & Family Features       │ 14.7%       │ Family Co-viewing│
│ Cluster 1 │ Global Serialized Dramas & K-Dramas       │ 15.8%       │ Binge Watchers   │
│ Cluster 2 │ Heritage Cinema & Classic Archives        │ 4.7%        │ Film Scholars    │
│ Cluster 3 │ Real-World Documentaries & Stand-Up       │ 27.0%       │ Adult Non-Fiction│
│ Cluster 4 │ Youth Animation & Western Episodic Shows  │ 14.4%       │ Kids & Parents   │
│ Cluster 5 │ Global Indie Cinema & Emotional Dramas    │ 23.4%       │ Cinephiles       │
└───────────┴───────────────────────────────────────────┴─────────────┴──────────────────┘
```

---

## 📊 Benchmark & Model Comparison

| Algorithm | Clusters ($K$) | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz (↑) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means (Optimal)** | **6** | **0.1583** | **1.7052** | **742.00** | **✅ Selected** |
| Agglomerative Hierarchical (Ward) | 6 | 0.1276 | 1.8143 | 652.75 | Benchmarked |
| Gaussian Mixture Model (GMM) | 6 | 0.0346 | 3.6311 | 353.47 | Benchmarked |
| DBSCAN (Density-Based) | 5 | 0.0227 | 1.7556 | 309.58 | Benchmarked |

---

## 🏗️ Architecture & Directory Structure

```
s:\AS ML\Task 4\
├── Dataset.csv                          # Primary Netflix raw catalog (8,790 titles)
├── MACHINE LEARNING TASK LIST.pdf       # Official internship task requirements
├── Netflix_Content_Segmentation_Task4.ipynb # Complete step-by-step reproducible notebook
├── train_pipeline.py                    # Standalone pipeline training and asset generator
├── app.py                               # FastAPI REST backend & web server
├── requirements.txt                     # Dependencies
├── README.md                            # Comprehensive system documentation
├── src/                                 # Modular Python package
│   ├── __init__.py
│   ├── data_loader.py                   # Data ingestion and cleaning
│   ├── feature_engineering.py           # Multi-label binarizer, scaling, PCA
│   ├── clustering_models.py             # K-Means, Agglomerative, GMM, DBSCAN
│   ├── cluster_profiler.py              # Archetype and statistical profiler
│   ├── evaluation.py                    # Unsupervised validation metrics
│   └── pipeline.py                      # Unified orchestrator
├── models/                              # Precomputed binaries and lightweight JSON
│   ├── segmentation_artifacts.json
│   ├── feature_engineer.joblib
│   └── cluster_models.joblib
├── static/                              # Front-end assets
│   ├── css/
│   │   └── style.css                    # Bespoke Midnight & Coral Plum luxury design
│   ├── js/
│   │   └── app.js                       # Interactive dashboard & Canvas controller
│   └── charts/                          # High-res static benchmark charts
│       ├── elbow_silhouette_optimization.png
│       ├── pca_cluster_scatter.png
│       └── cluster_format_breakdown.png
└── templates/
    └── index.html                       # Responsive single-page web dashboard
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Training & Generate Visual Artifacts
```bash
python train_pipeline.py
```

### 3. Launch Interactive Web Application
```bash
python app.py
```
Open your browser at **`http://127.0.0.1:8000`** to access the live dashboard.

---

## 🌐 REST API Endpoints

- `GET /api/overview`: Overview metadata, KPI summary, and explained PCA variance.
- `GET /api/clusters`: Statistical profiles for all 6 content personas.
- `GET /api/scatter-points`: 2D/3D PCA latent coordinates for canvas plotting.
- `GET /api/catalog`: Filterable, paginated, and searchable catalog database.
- `POST /api/predict`: Live segment classification and nearest-neighbor title retrieval.
- `GET /api/strategic-insights`: Automated business intelligence and portfolio gap analysis.

---

## 🎨 UI/UX Design Palette
The frontend uses a custom-curated **Midnight Obsidian & Coral Plum** luxury editorial theme:
- **Obsidian Dark Surface**: `#0c1017` / `#131b28`
- **Primary Accent**: Coral Plum / Crimson Rose (`#f43f5e`, `#e11d48`)
- **Secondary Accents**: Electric Iris (`#818cf8`), Emerald Mint (`#10b981`), Warm Amber (`#f59e0b`), Sky Azure (`#38bdf8`), Amethyst (`#a855f7`)
- **Typography**: Outfit & Plus Jakarta Sans
- **Glassmorphism**: Translucent frosted cards (`backdrop-filter: blur(14px)`).

---

## 🔒 Originality & Plagiarism Statement
This project is an **original, custom-engineered machine learning implementation** developed specifically for the **Auspify Technologies Machine Learning Task 4 Assessment**. All feature transformations, mathematical formulas, styling architectures, and API endpoints are built from first principles.
