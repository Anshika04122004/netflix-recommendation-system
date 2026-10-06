"""
Feature Engineering Module for Netflix Content Segmentation
Handles numerical transformations, categorical multi-label encodings,
scaling, and dimensionality reduction.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

class NetflixFeatureEngineer:
    def __init__(self):
        self.mlb_genres = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.feature_names: List[str] = []
        self.is_fitted = False

    @staticmethod
    def parse_duration(duration_str: str) -> Tuple[int, int, float]:
        """
        Parses duration strings like '90 min' or '2 Seasons'.
        Returns: (is_tv, season_count, estimated_total_minutes)
        """
        d = str(duration_str).lower().strip()
        if 'season' in d:
            parts = d.split()
            seasons = int(parts[0]) if parts and parts[0].isdigit() else 1
            # Average TV season is ~8 episodes of 45 mins = 360 mins
            return 1, seasons, float(seasons * 360)
        elif 'min' in d:
            parts = d.split()
            mins = int(parts[0]) if parts and parts[0].isdigit() else 90
            return 0, 0, float(mins)
        return 0, 0, 90.0

    @staticmethod
    def map_rating_tier(rating_str: str) -> str:
        """Groups MPAA / TV ratings into logical maturity tiers."""
        r = str(rating_str).upper().strip()
        if r in ['TV-MA', 'NC-17', 'R', 'UR']:
            return 'Mature_18Plus'
        elif r in ['TV-14', 'PG-13']:
            return 'Teen_14Plus'
        elif r in ['TV-PG', 'PG']:
            return 'Family_PG'
        elif r in ['TV-Y', 'TV-Y7', 'TV-G', 'G']:
            return 'Kids_Universal'
        return 'General_Unrated'

    @staticmethod
    def map_geographic_region(country_str: str) -> str:
        """Categorizes production countries into global market regions."""
        c = str(country_str).strip()
        if any(term in c for term in ['United States', 'Canada']):
            return 'North_America'
        elif any(term in c for term in ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka']):
            return 'South_Asia'
        elif any(term in c for term in ['Japan', 'South Korea', 'China', 'Taiwan', 'Hong Kong', 'Thailand', 'Indonesia', 'Philippines', 'Singapore']):
            return 'East_SE_Asia'
        elif any(term in c for term in ['United Kingdom', 'France', 'Spain', 'Germany', 'Italy', 'Turkey', 'Russia', 'Poland', 'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Belgium', 'Ireland']):
            return 'Europe'
        elif any(term in c for term in ['Mexico', 'Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru']):
            return 'Latin_America'
        elif any(term in c for term in ['Egypt', 'Nigeria', 'South Africa', 'Saudi Arabia', 'United Arab Emirates', 'Kenya']):
            return 'Middle_East_Africa'
        elif c == 'Not Given':
            return 'Global_NotGiven'
        return 'Other_International'

    def extract_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Transforms raw dataframe into standardized numerical feature vector for clustering.
        """
        # 1. Parse genres
        genre_lists = df['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
        if fit:
            genre_matrix = self.mlb_genres.fit_transform(genre_lists)
        else:
            # Handle unknown genres gracefully during inference
            genre_matrix = np.zeros((len(df), len(self.mlb_genres.classes_)))
            for i, glist in enumerate(genre_lists):
                for g in glist:
                    if g in self.mlb_genres.classes_:
                        idx = list(self.mlb_genres.classes_).index(g)
                        genre_matrix[i, idx] = 1

        genre_cols = [f"genre_{g}" for g in self.mlb_genres.classes_]

        # 2. Extract Duration features
        dur_tuples = [self.parse_duration(d) for d in df['duration']]
        df_dur = pd.DataFrame(dur_tuples, columns=['is_tv', 'season_count', 'est_minutes'])

        # 3. Categorical Rating tiers
        rating_tiers = df['rating'].apply(self.map_rating_tier)
        tier_list = ['Mature_18Plus', 'Teen_14Plus', 'Family_PG', 'Kids_Universal', 'General_Unrated']
        rating_matrix = np.zeros((len(df), len(tier_list)))
        for i, tier in enumerate(rating_tiers):
            if tier in tier_list:
                rating_matrix[i, tier_list.index(tier)] = 1
        rating_cols = [f"rating_{t}" for t in tier_list]

        # 4. Regional Categorization
        regions = df['country'].apply(self.map_geographic_region)
        region_list = ['North_America', 'South_Asia', 'East_SE_Asia', 'Europe', 'Latin_America', 'Middle_East_Africa', 'Other_International', 'Global_NotGiven']
        region_matrix = np.zeros((len(df), len(region_list)))
        for i, reg in enumerate(regions):
            if reg in region_list:
                region_matrix[i, region_list.index(reg)] = 1
        region_cols = [f"region_{r}" for r in region_list]

        # 5. Type indicator
        is_movie_arr = (df['type'] == 'Movie').astype(float).values.reshape(-1, 1)
        is_tv_arr = (df['type'] == 'TV Show').astype(float).values.reshape(-1, 1)

        # 6. Numerical Scaled features
        num_raw = np.column_stack([
            df['release_year'].astype(float).values,
            df_dur['season_count'].values,
            df_dur['est_minutes'].values,
            df.get('addition_lag_years', np.zeros(len(df))).astype(float).values
        ])

        if fit:
            num_scaled = self.scaler.fit_transform(num_raw)
            self.is_fitted = True
        else:
            num_scaled = self.scaler.transform(num_raw)

        num_cols = ['scaled_release_year', 'scaled_seasons', 'scaled_duration', 'scaled_addition_lag']

        # 7. Apply feature weighting for balanced semantic clustering
        # Genres: 1.3 weight, Type: 1.1 weight, Rating: 0.9 weight, Region: 0.8 weight, Numerics: 0.9 weight
        X_matrix = np.hstack([
            genre_matrix * 1.3,
            rating_matrix * 0.9,
            region_matrix * 0.8,
            is_movie_arr * 1.1,
            is_tv_arr * 1.1,
            num_scaled * 0.9
        ])

        self.feature_names = genre_cols + rating_cols + region_cols + ['is_movie', 'is_tv'] + num_cols
        return X_matrix

    def compute_projections(self, X: np.ndarray, fit: bool = True) -> Dict[str, np.ndarray]:
        """Calculates 2D and 3D PCA projections for interactive visualization."""
        if fit:
            pca_2d_coords = self.pca_2d.fit_transform(X)
            pca_3d_coords = self.pca_3d.fit_transform(X)
        else:
            pca_2d_coords = self.pca_2d.transform(X)
            pca_3d_coords = self.pca_3d.transform(X)
            
        return {
            "pca_2d": pca_2d_coords,
            "pca_3d": pca_3d_coords,
            "pca_2d_explained_variance": self.pca_2d.explained_variance_ratio_.tolist(),
            "pca_3d_explained_variance": self.pca_3d.explained_variance_ratio_.tolist()
        }
