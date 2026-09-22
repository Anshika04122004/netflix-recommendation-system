# 🎬 Task 1 — Netflix Content Recommendation System
## Complete Pipeline Walkthrough

---

## 🏗️ Architecture Overview

```
Dataset.csv (8,790 titles)
       │
       ▼
 ┌─────────────────────────────────────────────────────┐
 │  STEP 1 — Feature Preparation                       │
 │  genres × 3 + director × 2 + country + type +       │
 │  rating + decade  →  "content_soup" string           │
 └───────────────────────┬─────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────┐
 │  STEP 2 — TF-IDF Vectorisation                      │
 │  TfidfVectorizer(ngram=(1,2), sublinear_tf=True)    │
 │  8,790 × 4,583 sparse matrix                        │
 └───────────────────────┬─────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────┐
 │  STEP 3 — Cosine Similarity                         │
 │  8,790 × 8,790 pairwise similarity matrix           │
 │  mean=0.029  std=0.076  max=1.000                   │
 └───────────────────────┬─────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────┐
 │  STEP 4 — Recommendation Engine                     │
 │  get_recommendations(title, n=10)                   │
 │  → sorted Top-N by cosine score                     │
 └───────────────────────┬─────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────┐
 │  STEP 5 — Quality Evaluation                        │
 │  200 queries → Genre Overlap, ILD, Consistency      │
 └─────────────────────────────────────────────────────┘
```

---

## 📊 Step 1 — Exploratory Data Analysis

![EDA Dashboard](C:/Users/anshi/.gemini/antigravity-ide/brain/7ca40df2-12f6-4c3c-add5-26d77f53b3b5/step1_eda_overview.png)

> **Key Insights:** 69.7% Movies vs 30.3% TV Shows · Top genres: Dramas, Comedies, International · Content peaks 2017–2020 · Dominant ratings: TV-MA, TV-14 · Top countries: USA, India, UK

---

## 🔤 Step 2 — TF-IDF Feature Extraction

![TF-IDF Top Tokens](C:/Users/anshi/.gemini/antigravity-ide/brain/7ca40df2-12f6-4c3c-add5-26d77f53b3b5/step2_tfidf_top_tokens.png)

> **Matrix:** 8,790 titles × 4,583 vocabulary tokens · Sparsity: 99.75% · Top tokens are genre/era/country identifiers (exactly what drives good recommendations)

---

## 🔍 Step 3 — Cosine Similarity Matrix

![Similarity Analysis](C:/Users/anshi/.gemini/antigravity-ide/brain/7ca40df2-12f6-4c3c-add5-26d77f53b3b5/step3_similarity_analysis.png)

> **Full 8,790 × 8,790 matrix** · Most pairs score near 0 (different content) · Similar genre/country/type pairs score 0.7–1.0

---

## 🎯 Step 4 — Recommendations for Demo Titles

![Recommendation Visualisation](C:/Users/anshi/.gemini/antigravity-ide/brain/7ca40df2-12f6-4c3c-add5-26d77f53b3b5/step4_recommendations_viz.png)

### Live Results

| Query Title | #1 Recommendation | Score |
|-------------|-------------------|-------|
| 🩸 **Midnight Mass** | The Haunting of Bly Manor | 0.703 |
| 🔫 **Ganglands** | Dealer | 0.747 |
| 🎬 **Dick Johnson Is Dead** | Giving Voice | **1.000** |
| 👑 **The Crown** | Behind Her Eyes | **1.000** |
| 🦑 **Squid Game** | Kakegurui | **0.976** |

---

## 📈 Step 5 — Recommendation Quality Evaluation

![Evaluation Dashboard](C:/Users/anshi/.gemini/antigravity-ide/brain/7ca40df2-12f6-4c3c-add5-26d77f53b3b5/step5_evaluation_viz.png)

### Quality Scorecard (200-query sample)

| Metric | Score | Rating |
|--------|-------|--------|
| **Genre Overlap Ratio** | **0.943** | 🟢 Excellent — 94.3% genre match |
| **Type Consistency** | **0.997** | 🟢 Near-perfect — 99.7% same content type |
| **Avg Cosine Similarity** | **0.817** | 🟢 High content alignment |
| **Intra-List Diversity** | **0.158** | 🟡 Focused (expected for content-based) |
| **Catalogue Coverage** | **17.27%** | 🟡 1,519 unique titles surfaced |

---

## 📁 Output Files

All saved to `s:\AS ML\Task1_Outputs\`

| File | Size | Content |
|------|------|---------|
| `step1_eda_overview.png` | 165 KB | 5-panel EDA dashboard |
| `step2_tfidf_top_tokens.png` | 146 KB | Top 25 TF-IDF tokens bar chart |
| `step3_similarity_analysis.png` | 55 KB | Similarity distribution + heatmap |
| `step4_recommendations.csv` | 5 KB | Top-10 recs for 5 demo titles |
| `step4_recommendations_viz.png` | 111 KB | Bar + scatter recommendation plots |
| `step5_evaluation_metrics.csv` | 10 KB | Per-query metrics for 200 titles |
| `step5_evaluation_viz.png` | 138 KB | Evaluation dashboard + scorecard |

**Source Script:** `s:\AS ML\task1_netflix_recommendation.py` (465 lines, fully documented)
