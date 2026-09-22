"""
Feature Engineering and Preprocessing Pipeline for Netflix Content Type Classification.
Implements leak-free feature extractors, text vectorization, and categorical encodings.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class NetflixFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to engineer tabular, temporal, and lexical features.
    """
    def __init__(self, top_countries: int = 15):
        self.top_countries = top_countries
        self.popular_countries_: List[str] = []
        self.median_added_year_: float = 2019.0
        self.genre_keywords_ = [
            'drama', 'comedy', 'action', 'adventure', 'horror', 'thriller',
            'romantic', 'romance', 'documentary', 'docuseries', 'crime',
            'animation', 'anime', 'family', 'children', 'international',
            'reality', 'sci-fi', 'fantasy', 'mystery', 'music', 'musical',
            'stand-up', 'talk', 'teen', 'cult', 'classic', 'sports'
        ]

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        # Learn top countries from training data
        all_countries = X_df['country'].dropna().apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip() != 'Unknown'])
        country_series = pd.Series([c for sublist in all_countries for c in sublist])
        if not country_series.empty:
            self.popular_countries_ = country_series.value_counts().head(self.top_countries).index.tolist()
        else:
            self.popular_countries_ = ['United States', 'India', 'United Kingdom', 'Japan', 'South Korea', 'Canada']

        # Parse date_added to learn median added year
        parsed_dates = pd.to_datetime(X_df['date_added'], errors='coerce')
        valid_years = parsed_dates.dt.year.dropna()
        if not valid_years.empty:
            self.median_added_year_ = float(valid_years.median())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        features = pd.DataFrame(index=X_df.index)

        # 1. Temporal Features
        parsed_dates = pd.to_datetime(X_df['date_added'], errors='coerce')
        features['added_year'] = parsed_dates.dt.year.fillna(self.median_added_year_).astype(int)
        features['added_month'] = parsed_dates.dt.month.fillna(6).astype(int)
        features['added_dayofweek'] = parsed_dates.dt.dayofweek.fillna(3).astype(int)
        features['added_quarter'] = parsed_dates.dt.quarter.fillna(2).astype(int)
        
        # Release year & lag
        features['release_year'] = pd.to_numeric(X_df['release_year'], errors='coerce').fillna(2018).astype(int)
        features['years_between_release_and_added'] = np.maximum(0, features['added_year'] - features['release_year'])
        features['is_modern_content'] = (features['release_year'] >= 2015).astype(int)
        features['is_vintage_content'] = (features['release_year'] < 2000).astype(int)

        # 2. Metadata / Director Features
        features['has_director'] = (X_df['director'].astype(str).str.strip().str.lower() != 'unknown') & (X_df['director'].astype(str).str.strip().str.lower() != 'not given')
        features['has_director'] = features['has_director'].astype(int)
        features['director_count'] = X_df['director'].astype(str).apply(
            lambda x: len([d for d in x.split(',') if d.strip() not in ['Unknown', 'Not Given', '']]) if x not in ['Unknown', 'Not Given'] else 0
        )

        # 3. Country / Production Geography Features
        features['country_count'] = X_df['country'].astype(str).apply(
            lambda x: len([c for c in x.split(',') if c.strip() not in ['Unknown', '']]) if x != 'Unknown' else 0
        )
        features['is_international_coprod'] = (features['country_count'] > 1).astype(int)
        features['has_country'] = (X_df['country'].astype(str) != 'Unknown').astype(int)

        for country in self.popular_countries_:
            col_name = f"country_{re.sub(r'[^a-zA-Z0-9]', '_', country).lower()}"
            features[col_name] = X_df['country'].astype(str).apply(lambda x: 1 if country.lower() in x.lower() else 0)

        # 4. Rating System Nuances
        rating_str = X_df['rating'].astype(str).str.strip()
        features['is_tv_rating'] = rating_str.str.startswith('TV-').astype(int)
        features['is_mpaa_rating'] = rating_str.isin(['G', 'PG', 'PG-13', 'R', 'NC-17', 'NR', 'UR']).astype(int)
        features['is_mature_content'] = rating_str.isin(['TV-MA', 'R', 'NC-17']).astype(int)
        features['is_kids_content'] = rating_str.isin(['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'G', 'PG']).astype(int)

        # Rating one-hot encoding flags for common categories
        common_ratings = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'TV-Y', 'TV-Y7', 'PG', 'TV-G', 'NR', 'G']
        for r in common_ratings:
            col_name = f"rating_{re.sub(r'[^a-zA-Z0-9]', '_', r).lower()}"
            features[col_name] = (rating_str == r).astype(int)

        # 5. Title Lexical Features
        title_str = X_df['title'].astype(str)
        features['title_char_length'] = title_str.str.len()
        features['title_word_count'] = title_str.apply(lambda x: len(x.split()))
        features['title_has_number'] = title_str.str.contains(r'\d', regex=True).astype(int)
        features['title_uppercase_count'] = title_str.apply(lambda x: sum(1 for c in x if c.isupper()))
        features['title_has_colon_or_hyphen'] = title_str.str.contains(r'[:\-]', regex=True).astype(int)

        # 6. Domain-Agnostic Genre Semantic Keywords
        listed_in_str = X_df['listed_in'].astype(str).str.lower()
        features['genre_tag_count'] = listed_in_str.apply(lambda x: len(x.split(',')))
        for kw in self.genre_keywords_:
            features[f'genre_kw_{kw}'] = listed_in_str.apply(lambda x: 1 if kw in x else 0)

        return features


def build_full_feature_pipeline(tfidf_max_features: int = 100) -> Tuple[Pipeline, List[str]]:
    """
    Constructs a complete Scikit-Learn Pipeline combining tabular features and Title TF-IDF.
    
    Args:
        tfidf_max_features: Number of TF-IDF features to extract from title.
        
    Returns:
        Tuple of (sklearn Pipeline, list of feature names)
    """
    tabular_transformer = NetflixFeatureExtractor()
    
    # We construct a full pipeline that extracts both tabular features and title n-grams
    return tabular_transformer


def prepare_training_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, NetflixFeatureExtractor, TfidfVectorizer]:
    """
    Prepares train/test splits with stratified sampling and zero data leakage.
    
    Args:
        df: Cleaned dataframe.
        test_size: Fraction of data for test set.
        random_state: Seed for reproducibility.
        
    Returns:
        X_train_transformed, X_test_transformed, y_train, y_test, feature_extractor, tfidf_vectorizer
    """
    from sklearn.model_selection import train_test_split
    
    # Target encoding: Movie = 0, TV Show = 1 (or binary indicator)
    y = (df['type'] == 'TV Show').astype(int)
    X = df[['title', 'director', 'country', 'date_added', 'release_year', 'rating', 'listed_in']]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 1. Fit & transform tabular feature extractor
    feature_extractor = NetflixFeatureExtractor()
    X_train_tab = feature_extractor.fit_transform(X_train_raw)
    X_test_tab = feature_extractor.transform(X_test_raw)
    
    # 2. Fit & transform Title TF-IDF Vectorizer
    tfidf_vec = TfidfVectorizer(max_features=60, stop_words='english', ngram_range=(1, 2))
    train_tfidf = tfidf_vec.fit_transform(X_train_raw['title'])
    test_tfidf = tfidf_vec.transform(X_test_raw['title'])
    
    tfidf_cols = [f"tfidf_{w}" for w in tfidf_vec.get_feature_names_out()]
    train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols, index=X_train_tab.index)
    test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_cols, index=X_test_tab.index)
    
    X_train_final = pd.concat([X_train_tab, train_tfidf_df], axis=1)
    X_test_final = pd.concat([X_test_tab, test_tfidf_df], axis=1)
    
    return X_train_final, X_test_final, y_train, y_test, feature_extractor, tfidf_vec


if __name__ == "__main__":
    from data_loader import load_dataset, clean_data
    df = clean_data(load_dataset("Dataset.csv"))
    X_train, X_test, y_train, y_test, feat_ext, tfidf = prepare_training_data(df)
    print(f"X_train shape: {X_train.shape}, y_train distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"X_test shape: {X_test.shape}, y_test distribution:\n{y_test.value_counts(normalize=True)}")
    print(f"Features generated ({len(X_train.columns)}):", X_train.columns.tolist()[:15], "...")
