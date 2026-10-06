"""
Data Ingestion and Cleaning Module for Netflix Content Segmentation
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class NetflixDataLoader:
    def __init__(self, filepath: str = "Dataset.csv"):
        self.filepath = filepath
        self.df = None

    def load_and_clean(self) -> pd.DataFrame:
        """
        Loads the dataset and performs rigorous data cleaning, missing value handling,
        and standardizing column formats.
        """
        df = pd.read_csv(self.filepath)
        
        # Strip string whitespace
        for col in ['show_id', 'type', 'title', 'director', 'country', 'rating', 'duration', 'listed_in']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
        
        # Impute missing textual metadata
        df['director'] = df['director'].replace('', 'Not Given')
        df['country'] = df['country'].replace('', 'Not Given')
        df['rating'] = df['rating'].replace('', 'Unrated')
        
        # Standardize Rating naming
        rating_clean_map = {
            'UR': 'Unrated',
            'NR': 'Unrated',
            'TV-Y7-FV': 'TV-Y7'
        }
        df['rating'] = df['rating'].replace(rating_clean_map)
        
        # Parse Dates
        df['date_added_dt'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['added_year'] = df['date_added_dt'].dt.year.fillna(df['release_year']).astype(int)
        df['added_month'] = df['date_added_dt'].dt.month.fillna(6).astype(int)
        
        # Calculate catalog retention / addition lag
        df['addition_lag_years'] = np.maximum(0, df['added_year'] - df['release_year'])
        
        self.df = df
        return df

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns statistical overview of the loaded dataset."""
        if self.df is None:
            self.load_and_clean()
            
        return {
            "total_records": len(self.df),
            "total_features": len(self.df.columns),
            "movie_count": int((self.df['type'] == 'Movie').sum()),
            "tv_show_count": int((self.df['type'] == 'TV Show').sum()),
            "unique_countries": int(self.df['country'].nunique()),
            "year_min": int(self.df['release_year'].min()),
            "year_max": int(self.df['release_year'].max()),
            "unique_ratings": int(self.df['rating'].nunique())
        }
