# ============================================================
#  Netflix Content-Based Recommendation System  -  Task 1
#  4-Week ML Internship -- Auspify Technologies
# ============================================================
# Step 1: Prepare content-related features
# Step 2: Convert text data into machine-readable format (TF-IDF)
# Step 3: Calculate content similarity scores (Cosine Similarity)
# Step 4: Generate recommendations for selected titles
# Step 5: Evaluate recommendation quality
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATASET_PATH  = r"s:\AS ML\Dataset.csv"
OUTPUT_FOLDER = r"s:\AS ML\Task1_Outputs"
TOP_N         = 10
RANDOM_SEED   = 42
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PALETTE = {
    "bg":      "#0D0D0D",
    "surface": "#1A1A2E",
    "accent1": "#E50914",
    "accent2": "#F5A623",
    "accent3": "#4FC3F7",
    "text":    "#E0E0E0",
    "muted":   "#888888",
}

def _apply_dark(fig, axes=None):
    fig.patch.set_facecolor(PALETTE["bg"])
    if axes is not None:
        items = axes if hasattr(axes, "__iter__") else [axes]
        for ax in items:
            ax.set_facecolor(PALETTE["surface"])
            ax.tick_params(colors=PALETTE["text"])
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["muted"])

# =====================================================================
# STEP 1: Prepare Content-Related Features
# =====================================================================
print("=" * 64)
print("  STEP 1  >  Feature Preparation")
print("=" * 64)

df = pd.read_csv(DATASET_PATH)
print(f"  Dataset loaded: {df.shape[0]:,} titles x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

TEXT_COLS = ["type","title","director","country","rating","duration","listed_in"]
for c in TEXT_COLS:
    df[c] = df[c].fillna("").astype(str).str.strip()
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

def tok_genres(raw):
    raw = raw.lower()
    raw = re.sub(r"\s*&\s*", "_and_", raw)
    raw = re.sub(r",\s*", " ", raw)
    raw = re.sub(r"\s+", "_", raw.strip())
    return raw

def tok_director(raw):
    if not raw:
        return ""
    return "_".join(raw.lower().split()[:2])

df["genre_tokens"]   = df["listed_in"].apply(tok_genres)
df["director_token"] = df["director"].apply(tok_director)
df["country_token"]  = df["country"].str.lower().str.replace(" ", "_").str.split(",").str[0].str.strip()
df["type_token"]     = df["type"].str.lower().str.replace(" ", "_")
df["rating_token"]   = df["rating"].str.lower().str.replace("-", "_")
df["decade_token"]   = (df["release_year"] // 10 * 10).astype(str).apply(lambda x: f"decade_{x}s")

def build_soup(row):
    return (
        (row["genre_tokens"] + " ") * 3
        + (row["director_token"] + " ") * 2
        + row["country_token"] + " "
        + row["type_token"] + " "
        + row["rating_token"] + " "
        + row["decade_token"]
    )

df["content_soup"] = df.apply(build_soup, axis=1)
df = df.reset_index(drop=True)
title_to_idx = pd.Series(df.index, index=df["title"].str.lower())
print("  Feature engineering complete.")
print(f"  Sample soup: {df['content_soup'].iloc[0][:100]}")

# EDA Visualisation
fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
tc  = df["type"].value_counts()
ax0.pie(tc, labels=tc.index, autopct="%1.1f%%",
        colors=[PALETTE["accent1"], PALETTE["accent3"]],
        textprops={"color": PALETTE["text"]}, startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": PALETTE["bg"]})
ax0.set_title("Content Type Split", fontsize=13, color=PALETTE["text"])

all_g = []
for g in df["listed_in"]:
    all_g.extend([x.strip() for x in g.split(",")])
gs_ser = pd.Series(all_g).value_counts().head(15)
ax1 = fig.add_subplot(gs[0, 1:])
ax1.set_facecolor(PALETTE["surface"])
ax1.barh(gs_ser.index[::-1], gs_ser.values[::-1], color=PALETTE["accent1"], edgecolor=PALETTE["bg"])
ax1.set_xlabel("Count", color=PALETTE["text"])
ax1.set_title("Top 15 Genres on Netflix", fontsize=13, color=PALETTE["text"])
ax1.tick_params(colors=PALETTE["text"])
for s in ax1.spines.values(): s.set_edgecolor(PALETTE["muted"])

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(PALETTE["surface"])
ax2.hist(df[df["release_year"] > 1980]["release_year"], bins=30, color=PALETTE["accent2"], edgecolor=PALETTE["bg"])
ax2.set_title("Release Year Distribution", fontsize=13, color=PALETTE["text"])
ax2.set_xlabel("Year", color=PALETTE["text"])
ax2.set_ylabel("Count", color=PALETTE["text"])
ax2.tick_params(colors=PALETTE["text"])
for s in ax2.spines.values(): s.set_edgecolor(PALETTE["muted"])

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(PALETTE["surface"])
rc = df["rating"].value_counts().head(10)
ax3.bar(rc.index, rc.values, color=PALETTE["accent3"], edgecolor=PALETTE["bg"])
ax3.set_title("Top 10 Content Ratings", fontsize=13, color=PALETTE["text"])
ax3.tick_params(colors=PALETTE["text"], axis="x", rotation=45)
for s in ax3.spines.values(): s.set_edgecolor(PALETTE["muted"])

ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor(PALETTE["surface"])
cc = df["country"].str.split(",").str[0].str.strip().value_counts().head(10)
ax4.barh(cc.index[::-1], cc.values[::-1], color="#9B59B6", edgecolor=PALETTE["bg"])
ax4.set_title("Top 10 Production Countries", fontsize=13, color=PALETTE["text"])
ax4.tick_params(colors=PALETTE["text"])
for s in ax4.spines.values(): s.set_edgecolor(PALETTE["muted"])

fig.suptitle("Netflix Dataset - Exploratory Data Analysis", fontsize=16,
             color=PALETTE["accent1"], fontweight="bold")
fig.savefig(os.path.join(OUTPUT_FOLDER, "step1_eda_overview.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  EDA visualisation saved.")

# =====================================================================
# STEP 2: Convert Text Data into Machine-Readable Format (TF-IDF)
# =====================================================================
print("\n" + "=" * 64)
print("  STEP 2  >  TF-IDF Feature Extraction")
print("=" * 64)

vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=2,
    max_df=0.90, sublinear_tf=True, stop_words="english"
)
tfidf_matrix = vectorizer.fit_transform(df["content_soup"])
vocab_size = len(vectorizer.vocabulary_)
print(f"  TF-IDF matrix shape : {tfidf_matrix.shape}")
print(f"  Vocabulary size     : {vocab_size:,} tokens")
print(f"  Matrix sparsity     : {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0]*tfidf_matrix.shape[1]):.4f}")

feature_names = vectorizer.get_feature_names_out()
mean_w = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
top_idx = mean_w.argsort()[::-1][:25]
top_tok = [(feature_names[i], mean_w[i]) for i in top_idx]

fig, ax = plt.subplots(figsize=(13, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["surface"])
tn, tw = zip(*top_tok)
bar_colors = [PALETTE["accent1"] if i % 2 == 0 else PALETTE["accent2"] for i in range(25)]
ax.bar(tn, tw, color=bar_colors, edgecolor=PALETTE["bg"])
ax.set_title("Top 25 TF-IDF Feature Tokens (Mean Corpus Weight)", fontsize=13, color=PALETTE["text"])
ax.set_xlabel("Token", color=PALETTE["text"])
ax.set_ylabel("Mean TF-IDF Weight", color=PALETTE["text"])
ax.tick_params(axis="x", rotation=45, colors=PALETTE["text"])
ax.tick_params(axis="y", colors=PALETTE["text"])
for s in ax.spines.values(): s.set_edgecolor(PALETTE["muted"])
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER, "step2_tfidf_top_tokens.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  TF-IDF visualisation saved.")

# =====================================================================
# STEP 3: Calculate Content Similarity Scores
# =====================================================================
print("\n" + "=" * 64)
print("  STEP 3  >  Cosine Similarity Matrix")
print("=" * 64)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
np.fill_diagonal(cosine_sim, 0)
print(f"  Similarity matrix : {cosine_sim.shape}")
print(f"  Mean similarity   : {cosine_sim.mean():.4f}")
print(f"  Std  similarity   : {cosine_sim.std():.4f}")
print(f"  Max similarity    : {cosine_sim.max():.4f}")

np.random.seed(RANDOM_SEED)
flat_s = cosine_sim.ravel()[np.random.choice(cosine_sim.size, 50000, replace=False)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
_apply_dark(fig, axes)
axes[0].hist(flat_s, bins=60, color=PALETTE["accent1"], edgecolor=PALETTE["bg"], log=True)
axes[0].set_title("Cosine Similarity Distribution (Sample)", fontsize=12)
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Count (log scale)")

rand_idx = np.random.choice(len(df), 40, replace=False)
sub_mat  = cosine_sim[np.ix_(rand_idx, rand_idx)]
sns.heatmap(sub_mat, ax=axes[1], cmap="Reds", linewidths=0,
            xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.7})
axes[1].set_title("Similarity Heatmap (Random 40-Title Sub-Matrix)", fontsize=12)
axes[1].title.set_color(PALETTE["text"])
fig.suptitle("Step 3 - Cosine Similarity Analysis", fontsize=14,
             color=PALETTE["accent1"], fontweight="bold")
fig.savefig(os.path.join(OUTPUT_FOLDER, "step3_similarity_analysis.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Similarity visualisation saved.")

# =====================================================================
# STEP 4: Generate Recommendations for Selected Titles
# =====================================================================
print("\n" + "=" * 64)
print("  STEP 4  >  Recommendation Engine")
print("=" * 64)

def get_recommendations(title, n=TOP_N, type_filter=None):
    """Return top-n most similar Netflix titles for a given query."""
    key = title.lower().strip()
    if key not in title_to_idx.index:
        matches = [t for t in title_to_idx.index if key in t]
        if not matches:
            return pd.DataFrame()
        key = matches[0]
    idx = title_to_idx[key]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    results = []
    for i, sc in scores:
        if len(results) >= n:
            break
        if i == idx:
            continue
        row = df.iloc[i]
        if type_filter and row["type"] != type_filter:
            continue
        results.append({
            "title":            row["title"],
            "type":             row["type"],
            "listed_in":        row["listed_in"],
            "rating":           row["rating"],
            "release_year":     int(row["release_year"]),
            "country":          row["country"],
            "similarity_score": round(float(sc), 4),
        })
    return pd.DataFrame(results)

DEMO_TITLES = ["Midnight Mass", "Ganglands", "Dick Johnson Is Dead", "The Crown", "Squid Game"]
all_recs = {}

for dt in DEMO_TITLES:
    recs = get_recommendations(dt)
    all_recs[dt] = recs
    print(f"\n  Query: '{dt}'  ({len(recs)} recs)")
    if not recs.empty:
        for rank, (_, r) in enumerate(recs.iterrows(), 1):
            t_display = r["title"][:38] + ".." if len(r["title"]) > 40 else r["title"]
            print(f"    {rank:>2}. {t_display:<42} [{r['type']:<8}] {r['similarity_score']:.4f}")

csv_rows = []
for q, r in all_recs.items():
    temp = r.copy()
    temp.insert(0, "query_title", q)
    csv_rows.append(temp)
if csv_rows:
    pd.concat(csv_rows, ignore_index=True).to_csv(
        os.path.join(OUTPUT_FOLDER, "step4_recommendations.csv"), index=False)
    print("\n  Recommendations CSV saved.")

def plot_recs(query_title, recs, out_path):
    if recs.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE["bg"])
    _apply_dark(fig, axes)
    ts = [t[:28] + ".." if len(t) > 30 else t for t in recs["title"]]
    bar_col = [PALETTE["accent1"] if t == "Movie" else PALETTE["accent3"] for t in recs["type"]]
    axes[0].barh(ts[::-1], recs["similarity_score"].values[::-1],
                 color=bar_col[::-1], edgecolor=PALETTE["bg"])
    axes[0].set_xlabel("Cosine Similarity Score")
    axes[0].set_title(f"Top Recommendations for:\n'{query_title}'", fontsize=12)
    lp = [mpatches.Patch(color=PALETTE["accent1"], label="Movie"),
          mpatches.Patch(color=PALETTE["accent3"], label="TV Show")]
    axes[0].legend(handles=lp, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    sc = [PALETTE["accent1"] if t == "Movie" else PALETTE["accent3"] for t in recs["type"]]
    axes[1].scatter(recs["release_year"], recs["similarity_score"],
                    c=sc, s=100, edgecolors=PALETTE["muted"], zorder=3)
    for _, row in recs.iterrows():
        axes[1].annotate(row["title"][:14], (row["release_year"], row["similarity_score"]),
                        xytext=(4, 4), textcoords="offset points",
                        color=PALETTE["text"], fontsize=7)
    axes[1].set_xlabel("Release Year")
    axes[1].set_ylabel("Similarity Score")
    axes[1].set_title("Release Year vs Similarity Score", fontsize=12)
    axes[1].legend(handles=lp, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    fig.suptitle("Step 4 - Recommendations Analysis", fontsize=14,
                 color=PALETTE["accent1"], fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

plot_recs(DEMO_TITLES[0], all_recs[DEMO_TITLES[0]],
          os.path.join(OUTPUT_FOLDER, "step4_recommendations_viz.png"))
print("  Recommendation visualisation saved.")

# =====================================================================
# STEP 5: Evaluate Recommendation Quality
# =====================================================================
print("\n" + "=" * 64)
print("  STEP 5  >  Recommendation Quality Evaluation")
print("=" * 64)

def genre_set(s):
    return {g.strip().lower() for g in s.split(",")}

def genre_overlap(qg, rg):
    if not qg:
        return 0.0
    return len(qg & rg) / len(qg)

def intra_list_diversity(rec_indices):
    """1 minus mean pairwise cosine similarity within the recommendation list."""
    if len(rec_indices) < 2:
        return 1.0
    sub = cosine_sim[np.ix_(rec_indices, rec_indices)]
    np.fill_diagonal(sub, 0)
    n = len(rec_indices)
    return round(1.0 - float(sub.sum() / (n * (n - 1))), 4)

np.random.seed(RANDOM_SEED)
eval_sample = df.sample(min(200, len(df)), random_state=RANDOM_SEED)
metric_rows = []
covered_titles = set()

for _, q_row in eval_sample.iterrows():
    qt    = q_row["title"]
    qg    = genre_set(q_row["listed_in"])
    qtype = q_row["type"]
    qi    = title_to_idx.get(qt.lower())
    if qi is None:
        continue
    recs = get_recommendations(qt, n=TOP_N)
    if recs.empty:
        continue
    rec_idx = [int(title_to_idx[t.lower()]) if not isinstance(title_to_idx[t.lower()], pd.Series)
               else int(title_to_idx[t.lower()].iloc[0])
               for t in recs["title"] if t.lower() in title_to_idx]
    covered_titles.update(recs["title"].tolist())
    gor_vals = [genre_overlap(qg, genre_set(r["listed_in"])) for _, r in recs.iterrows()]
    metric_rows.append({
        "query":                qt,
        "type":                 qtype,
        "avg_genre_overlap":    round(float(np.mean(gor_vals)), 4),
        "intra_list_diversity": intra_list_diversity(rec_idx),
        "avg_similarity":       round(float(recs["similarity_score"].mean()), 4),
        "type_consistency":     round(float((recs["type"] == qtype).mean()), 4),
        "n_recs":               len(recs),
    })

mdf = pd.DataFrame(metric_rows)
cov = round(len(covered_titles) / len(df) * 100, 2)

print(f"  Evaluation sample : {len(mdf)} queries")
print(f"  Catalogue coverage: {cov}%")
print("\n  Aggregate Metrics:")
print(mdf[["avg_genre_overlap","intra_list_diversity","avg_similarity","type_consistency"]].describe().round(4).to_string())
mdf.to_csv(os.path.join(OUTPUT_FOLDER, "step5_evaluation_metrics.csv"), index=False)
print("\n  Evaluation CSV saved.")

fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
metric_cfg = [
    ("avg_genre_overlap",    "Avg Genre Overlap Ratio",    PALETTE["accent1"]),
    ("intra_list_diversity", "Intra-List Diversity (ILD)", PALETTE["accent2"]),
    ("avg_similarity",       "Avg Cosine Similarity",      PALETTE["accent3"]),
    ("type_consistency",     "Type Consistency Rate",       "#9B59B6"),
]
for pos, (col, lbl, clr) in enumerate(metric_cfg):
    ax = fig.add_subplot(gs[pos // 3, pos % 3])
    ax.set_facecolor(PALETTE["surface"])
    ax.hist(mdf[col], bins=30, color=clr, edgecolor=PALETTE["bg"], alpha=0.9)
    mv = mdf[col].mean()
    ax.axvline(mv, color="white", linestyle="--", linewidth=1.5, label=f"Mean: {mv:.3f}")
    ax.set_title(lbl, fontsize=11, color=PALETTE["text"])
    ax.set_xlabel("Value", color=PALETTE["text"])
    ax.set_ylabel("Count", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], fontsize=9)
    for s in ax.spines.values(): s.set_edgecolor(PALETTE["muted"])

ax_c = fig.add_subplot(gs[1, 1:])
ax_c.set_facecolor(PALETTE["surface"])
ax_c.axis("off")
scorecard = [
    ["Metric",             "Mean",                                          "Median",                                            "Std"],
    ["Genre Overlap",      f"{mdf['avg_genre_overlap'].mean():.3f}",        f"{mdf['avg_genre_overlap'].median():.3f}",          f"{mdf['avg_genre_overlap'].std():.3f}"],
    ["ILD",                f"{mdf['intra_list_diversity'].mean():.3f}",     f"{mdf['intra_list_diversity'].median():.3f}",       f"{mdf['intra_list_diversity'].std():.3f}"],
    ["Avg Similarity",     f"{mdf['avg_similarity'].mean():.3f}",           f"{mdf['avg_similarity'].median():.3f}",             f"{mdf['avg_similarity'].std():.3f}"],
    ["Type Consistency",   f"{mdf['type_consistency'].mean():.3f}",         f"{mdf['type_consistency'].median():.3f}",           f"{mdf['type_consistency'].std():.3f}"],
    ["Coverage",           f"{cov}%",                                       "-",                                                 "-"],
]
tbl = ax_c.table(cellText=scorecard[1:], colLabels=scorecard[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(PALETTE["accent1"])
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#222244")
        cell.set_text_props(color=PALETTE["text"])
    else:
        cell.set_facecolor(PALETTE["surface"])
        cell.set_text_props(color=PALETTE["text"])
    cell.set_edgecolor(PALETTE["muted"])
ax_c.set_title("Recommendation Quality Scorecard", fontsize=12, color=PALETTE["text"], pad=10)
fig.suptitle("Step 5 - Recommendation Quality Evaluation", fontsize=15,
             color=PALETTE["accent1"], fontweight="bold")
fig.savefig(os.path.join(OUTPUT_FOLDER, "step5_evaluation_viz.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Evaluation visualisation saved.")

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 64)
print("  TASK 1 PIPELINE COMPLETE")
print("=" * 64)
print(f"  TF-IDF Matrix     : {tfidf_matrix.shape[0]:,} x {tfidf_matrix.shape[1]:,}  ({vocab_size:,} vocab tokens)")
print(f"  Genre Overlap     : {mdf['avg_genre_overlap'].mean():.3f}")
print(f"  ILD               : {mdf['intra_list_diversity'].mean():.3f}")
print(f"  Avg Similarity    : {mdf['avg_similarity'].mean():.3f}")
print(f"  Type Consistency  : {mdf['type_consistency'].mean():.3f}")
print(f"  Catalogue Coverage: {cov}%")
print(f"\n  Output folder: {OUTPUT_FOLDER}")
print("  Files generated:")
for fn in sorted(os.listdir(OUTPUT_FOLDER)):
    fp = os.path.join(OUTPUT_FOLDER, fn)
    print(f"    {fn}  ({os.path.getsize(fp) // 1024} KB)")
