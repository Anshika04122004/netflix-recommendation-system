"""
Enhanced pre-processing and feature engineering pipeline for the Netflix Audience Rating Classifier.
Uses rich feature engineering: TF-IDF on cast/description, interaction features, 
and SMOTE-aware label grouping for rare classes.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Fixed Top Categories identified during dataset analysis
TOP_COUNTRIES = [
    'United States', 'India', 'United Kingdom', 'Pakistan', 'Canada',
    'Japan', 'South Korea', 'France', 'Spain', 'Mexico',
    'Egypt', 'Australia', 'Turkey', 'Nigeria', 'Germany', 'Not Given'
]

ALL_GENRES = [
    'Action & Adventure', 'Anime Features', 'Anime Series', 'British TV Shows',
    'Children & Family Movies', 'Classic & Cult TV', 'Classic Movies', 'Comedies',
    'Crime TV Shows', 'Cult Movies', 'Documentaries', 'Docuseries', 'Dramas',
    'Faith & Spirituality', 'Horror Movies', 'Independent Movies', 'International Movies',
    'International TV Shows', "Kids' TV", 'Korean TV Shows', 'LGBTQ Movies',
    'Movies', 'Music & Musicals', 'Reality TV', 'Romantic Movies', 'Romantic TV Shows',
    'Sci-Fi & Fantasy', 'Science & Nature TV', 'Spanish-Language TV Shows',
    'Sports Movies', 'Stand-Up Comedy', 'Stand-Up Comedy & Talk Shows',
    'TV Action & Adventure', 'TV Comedies', 'TV Dramas', 'TV Horror',
    'TV Mysteries', 'TV Sci-Fi & Fantasy', 'TV Shows', 'TV Thrillers',
    'Teen TV Shows', 'Thrillers'
]

# Derived content type families that strongly predict rating categories
GENRE_GROUPS = {
    'is_kids_content': ["Children & Family Movies", "Kids' TV", 'Anime Features', 'Anime Series'],
    'is_adult_drama': ['Dramas', 'TV Dramas', 'Crime TV Shows', 'TV Thrillers', 'TV Mysteries'],
    'is_comedy': ['Comedies', 'TV Comedies', 'Stand-Up Comedy', 'Stand-Up Comedy & Talk Shows'],
    'is_horror': ['Horror Movies', 'Cult Movies', 'Thrillers'],
    'is_documentary': ['Documentaries', 'Docuseries', 'Science & Nature TV'],
    'is_international': ['International Movies', 'International TV Shows', 'Korean TV Shows',
                         'Spanish-Language TV Shows', 'British TV Shows'],
    'is_animation': ['Anime Features', 'Anime Series'],
    'is_reality': ['Reality TV', 'Sports Movies'],
    'is_romantic': ['Romantic Movies', 'Romantic TV Shows', 'LGBTQ Movies'],
    'is_scifi': ['Sci-Fi & Fantasy', 'TV Sci-Fi & Fantasy', 'Action & Adventure', 'TV Action & Adventure'],
    'is_family': ['Family Movies', 'Faith & Spirituality'],
    'is_teen': ['Teen TV Shows'],
}

# Rating-to-numeric ordinality (for interaction features)
RATING_ORDINALITY = {
    'G': 1, 'TV-G': 1, 'TV-Y': 2, 'TV-Y7': 3, 'TV-Y7-FV': 3,
    'PG': 4, 'TV-PG': 4, 'PG-13': 5, 'TV-14': 6,
    'R': 7, 'TV-MA': 8, 'NC-17': 9, 'NR': 5, 'UR': 5
}

NUMERIC_FEATURES = [
    'added_year', 'added_month', 'release_year', 'content_age',
    'duration_value', 'country_count', 'genre_count',
    'has_director', 'director_popularity',
    'is_movie', 'is_tv_show',
    # Genre group flags
    'is_kids_content', 'is_adult_drama', 'is_comedy', 'is_horror',
    'is_documentary', 'is_international', 'is_animation', 'is_reality',
    'is_romantic', 'is_scifi', 'is_family', 'is_teen',
    # Interaction features
    'movie_duration_mins', 'season_count',
    'release_decade', 'is_recent',
    'international_content', 'us_primary',
    'india_primary', 'anime_content',
    'cast_size',
]

CATEGORICAL_FEATURES = ['type', 'duration_type', 'primary_country']

# TF-IDF vocab for description
DESC_TFIDF_FEATURES = 50
CAST_TFIDF_FEATURES = 30


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts rich structured features from raw Netflix metadata columns.
    Includes TF-IDF for description and cast, genre group flags, 
    and interaction features.
    """
    def __init__(self):
        self.top_directors = []
        self.director_popularity_map = {}
        self.desc_tfidf = TfidfVectorizer(max_features=DESC_TFIDF_FEATURES,
                                          stop_words='english', ngram_range=(1, 2),
                                          min_df=2)
        self.cast_tfidf = TfidfVectorizer(max_features=CAST_TFIDF_FEATURES,
                                          min_df=2, analyzer='word',
                                          token_pattern=r'[A-Za-z][a-z]+(?:\s[A-Z][a-z]+)*')
        self.top_countries = TOP_COUNTRIES
        self.all_genres = ALL_GENRES
        self._desc_tfidf_fitted = False
        self._cast_tfidf_fitted = False

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Build director popularity map (fraction of data they appear in)
        if 'director' in df.columns:
            dir_series = df['director'].fillna('Not Given').astype(str)
            counts = dir_series.value_counts()
            total = len(dir_series)
            self.director_popularity_map = {d: c/total for d, c in counts.items() if d != 'Not Given'}
            frequent = counts[counts >= 3].index.tolist()
            self.top_directors = [d for d in frequent if d != 'Not Given']
        
        # Fit TF-IDF on description
        if 'description' in df.columns:
            desc_corpus = df['description'].fillna('').astype(str).tolist()
            try:
                self.desc_tfidf.fit(desc_corpus)
                self._desc_tfidf_fitted = True
            except Exception:
                self._desc_tfidf_fitted = False

        # Fit TF-IDF on cast
        if 'cast' in df.columns:
            cast_corpus = df['cast'].fillna('').astype(str).tolist()
            try:
                self.cast_tfidf.fit(cast_corpus)
                self._cast_tfidf_fitted = True
            except Exception:
                self._cast_tfidf_fitted = False

        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        transformed = pd.DataFrame(index=df.index)

        # 1. Content Type (binary)
        type_raw = df.get('type', pd.Series('Movie', index=df.index)).fillna('Movie').astype(str)
        transformed['type'] = type_raw
        transformed['is_movie'] = (type_raw.str.lower() == 'movie').astype(int)
        transformed['is_tv_show'] = (type_raw.str.lower() == 'tv show').astype(int)

        # 2. Date Added features
        date_series = pd.to_datetime(df.get('date_added', pd.Series(None, index=df.index)), errors='coerce')
        transformed['added_year'] = date_series.dt.year.fillna(2019).astype(int)
        transformed['added_month'] = date_series.dt.month.fillna(6).astype(int)

        # 3. Release Year, Content Age & Decade
        rel_year = pd.to_numeric(df.get('release_year', pd.Series(2018, index=df.index)), errors='coerce').fillna(2018).astype(int)
        transformed['release_year'] = rel_year
        transformed['content_age'] = (transformed['added_year'] - rel_year).clip(lower=0)
        transformed['release_decade'] = ((rel_year // 10) * 10).astype(int)
        transformed['is_recent'] = (rel_year >= 2015).astype(int)

        # 4. Duration features
        dur_raw = df.get('duration', pd.Series('90 min', index=df.index)).fillna('90 min').astype(str)
        dur_num = dur_raw.str.extract(r'(\d+)')[0].astype(float)
        is_minutes = dur_raw.str.contains('min', case=False, na=False)
        default_dur = pd.Series(np.where(is_minutes, 90, 1), index=df.index, dtype=float)
        dur_num = dur_num.fillna(default_dur)
        transformed['duration_value'] = dur_num.astype(float)
        transformed['duration_type'] = np.where(is_minutes, 'min', 'Season')
        transformed['movie_duration_mins'] = np.where(is_minutes, dur_num.astype(float), 0)
        transformed['season_count'] = np.where(~is_minutes, dur_num.astype(float), 0)

        # 5. Country features
        country_raw = df.get('country', pd.Series('Not Given', index=df.index)).fillna('Not Given').astype(str)
        transformed['country_count'] = country_raw.apply(
            lambda s: len([c for c in str(s).split(',') if c.strip()]) if s != 'Not Given' else 0
        )
        primary_country = country_raw.apply(lambda s: str(s).split(',')[0].strip() if s and s != 'Not Given' else 'Not Given')
        transformed['primary_country'] = primary_country.apply(
            lambda c: c if c in self.top_countries else 'Other'
        )
        transformed['international_content'] = (~primary_country.isin(['United States', 'Not Given'])).astype(int)
        transformed['us_primary'] = (primary_country == 'United States').astype(int)
        transformed['india_primary'] = (primary_country == 'India').astype(int)

        # 6. Director features
        dir_raw = df.get('director', pd.Series('Not Given', index=df.index)).fillna('Not Given').astype(str)
        transformed['has_director'] = (dir_raw != 'Not Given').astype(int)
        transformed['director_popularity'] = dir_raw.apply(
            lambda d: self.director_popularity_map.get(d, 0.0)
        )

        # 7. Genre features (binary flags per genre)
        genre_raw = df.get('listed_in', pd.Series('', index=df.index)).fillna('').astype(str)
        transformed['genre_count'] = genre_raw.apply(
            lambda s: len([g for g in str(s).split(',') if g.strip()])
        )
        for g in self.all_genres:
            transformed['genre_' + g] = genre_raw.apply(
                lambda s: 1 if g in [x.strip() for x in str(s).split(',')] else 0
            )

        # 8. Genre group flags (aggregated genre semantics)
        for group_name, genres in GENRE_GROUPS.items():
            transformed[group_name] = genre_raw.apply(
                lambda s: 1 if any(g in [x.strip() for x in str(s).split(',')] for g in genres) else 0
            )

        # 9. Anime flag
        transformed['anime_content'] = (
            transformed.get('genre_Anime Features', pd.Series(0, index=df.index)) |
            transformed.get('genre_Anime Series', pd.Series(0, index=df.index))
        ).astype(int)

        # 10. Cast size
        cast_raw = df.get('cast', pd.Series('', index=df.index)).fillna('').astype(str)
        transformed['cast_size'] = cast_raw.apply(
            lambda s: len([c for c in str(s).split(',') if c.strip()]) if s else 0
        )

        return transformed

    def get_tfidf_features(self, X):
        """Returns TF-IDF feature arrays for description and cast."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        results = []
        
        if self._desc_tfidf_fitted and 'description' in df.columns:
            desc_corpus = df['description'].fillna('').astype(str).tolist()
            try:
                desc_mat = self.desc_tfidf.transform(desc_corpus).toarray()
                results.append(desc_mat)
            except Exception:
                results.append(np.zeros((len(df), DESC_TFIDF_FEATURES)))
        else:
            results.append(np.zeros((len(df), DESC_TFIDF_FEATURES)))

        if self._cast_tfidf_fitted and 'cast' in df.columns:
            cast_corpus = df['cast'].fillna('').astype(str).tolist()
            try:
                cast_mat = self.cast_tfidf.transform(cast_corpus).toarray()
                results.append(cast_mat)
            except Exception:
                results.append(np.zeros((len(df), CAST_TFIDF_FEATURES)))
        else:
            results.append(np.zeros((len(df), CAST_TFIDF_FEATURES)))

        return np.hstack(results)


class NetflixPreprocessor:
    """
    Complete Preprocessor handling feature extraction, scaling, and one-hot encoding.
    Includes TF-IDF for text fields (description, cast) for richer semantic features.
    """
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X, y=None):
        self.feature_extractor.fit(X, y)
        X_extracted = self.feature_extractor.transform(X)

        # Scale numerical features
        numeric_cols = [c for c in NUMERIC_FEATURES if c in X_extracted.columns]
        self.scaler.fit(X_extracted[numeric_cols])
        self._numeric_cols = numeric_cols

        # OneHot encode categorical features
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_extracted.columns]
        self.encoder.fit(X_extracted[cat_cols])
        self._cat_cols = cat_cols

        if y is not None:
            self.label_encoder.fit(y)

        # Compute full list of output feature names
        genre_cols = ['genre_' + g for g in ALL_GENRES]
        group_cols = list(GENRE_GROUPS.keys())
        cat_encoded_names = self.encoder.get_feature_names_out(cat_cols).tolist()
        desc_names = [f'desc_tfidf_{i}' for i in range(DESC_TFIDF_FEATURES)]
        cast_names = [f'cast_tfidf_{i}' for i in range(CAST_TFIDF_FEATURES)]

        self.feature_names = (
            numeric_cols + genre_cols + group_cols + cat_encoded_names +
            desc_names + cast_names
        )
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        X_extracted = self.feature_extractor.transform(X)

        num_scaled = self.scaler.transform(X_extracted[self._numeric_cols])
        genre_cols = ['genre_' + g for g in ALL_GENRES]
        genre_vals = X_extracted[genre_cols].values
        group_vals = X_extracted[list(GENRE_GROUPS.keys())].values
        cat_encoded = self.encoder.transform(X_extracted[self._cat_cols])
        tfidf_feats = self.feature_extractor.get_tfidf_features(X)

        X_transformed = np.hstack([num_scaled, genre_vals, group_vals, cat_encoded, tfidf_feats])
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def encode_target(self, y):
        return self.label_encoder.transform(y)

    def decode_target(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)

    def get_classes(self):
        return list(self.label_encoder.classes_)

    def get_feature_names(self):
        return self.feature_names
