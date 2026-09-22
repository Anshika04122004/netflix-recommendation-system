import os, re, json, warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

DATASET_PATH = r"s:\AS ML\Dataset.csv"
OUT_JSON     = r"s:\AS ML\netflix_recs_data.json"
TOP_N        = 10

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

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
    if not raw: return ""
    return "_".join(raw.lower().split()[:2])

df["genre_tokens"]   = df["listed_in"].apply(tok_genres)
df["director_token"] = df["director"].apply(tok_director)
df["country_token"]  = df["country"].str.lower().str.replace(" ","_").str.split(",").str[0].str.strip()
df["type_token"]     = df["type"].str.lower().str.replace(" ","_")
df["rating_token"]   = df["rating"].str.lower().str.replace("-","_")
df["decade_token"]   = (df["release_year"] // 10 * 10).astype(str).apply(lambda x: f"decade_{x}s")

def build_soup(row):
    return ((row["genre_tokens"]+" ")*3 + (row["director_token"]+" ")*2 +
            row["country_token"]+" "+row["type_token"]+" "+row["rating_token"]+" "+row["decade_token"])

df["content_soup"] = df.apply(build_soup, axis=1)
df = df.reset_index(drop=True)
title_to_idx = pd.Series(df.index, index=df["title"].str.lower())

print("Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.90, sublinear_tf=True, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content_soup"])

print("Computing cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
np.fill_diagonal(cosine_sim, 0)

print("Generating all recommendations...")
all_recs = {}
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing {idx}/{len(df)}...")
    title = row["title"]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    recs = []
    for i, sc in scores:
        if len(recs) >= TOP_N: break
        r = df.iloc[i]
        recs.append({
            "t": r["title"],
            "tp": r["type"],
            "g": r["listed_in"],
            "ra": r["rating"],
            "y": int(r["release_year"]),
            "c": r["country"].split(",")[0].strip(),
            "s": round(float(sc), 3)
        })
    all_recs[title] = recs

# Build titles catalogue for search autocomplete
catalogue = []
for idx, row in df.iterrows():
    catalogue.append({
        "title": row["title"],
        "type":  row["type"],
        "genre": row["listed_in"],
        "year":  int(row["release_year"]),
        "rating": row["rating"],
        "country": row["country"].split(",")[0].strip()
    })

output = {"catalogue": catalogue, "recs": all_recs}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",",":"))

size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
print(f"\nData exported to: {OUT_JSON}  ({size_mb:.1f} MB)")
print(f"Total titles with recs: {len(all_recs)}")
