"""
Data Loader Module for Netflix Content Type Prediction.
Handles dataset loading, validation, cleaning, and preprocessing.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_dataset(file_path: str = "Dataset.csv") -> pd.DataFrame:
    """
    Load the Netflix dataset from a CSV file and perform initial validation.
    
    Args:
        file_path: Path to the dataset CSV file.
        
    Returns:
        pd.DataFrame: Cleaned raw dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Standardize column names (strip whitespace and lowercase)
    df.columns = df.columns.str.strip()
    
    # Required columns check
    required_cols = ['show_id', 'type', 'title', 'director', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe:
    - Normalizes target column
    - Imputes missing/unknown values cleanly
    - Normalizes string formatting
    - Removes invalid or corrupted records
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Standardize target 'type'
    df['type'] = df['type'].str.title()
    df = df[df['type'].isin(['Movie', 'Tv Show', 'TV Show'])].copy()
    df['type'] = df['type'].replace({'Tv Show': 'TV Show'})
    
    # Handle missing/placeholder values in categorical columns
    placeholders = ['Not Given', 'Unknown', 'nan', 'None', '', 'NaN']
    
    for col in ['director', 'country', 'rating']:
        df[col] = df[col].replace(placeholders, 'Unknown')
        
    # Ensure release_year is numeric
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df = df.dropna(subset=['release_year'])
    df['release_year'] = df['release_year'].astype(int)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive statistical and schema summary of the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        dict: Summary statistics and distributions.
    """
    target_counts = df['type'].value_counts().to_dict()
    total_records = len(df)
    
    summary = {
        "total_records": total_records,
        "total_features": df.shape[1],
        "target_distribution": {
            k: {
                "count": int(v),
                "percentage": round((v / total_records) * 100, 2)
            } for k, v in target_counts.items()
        },
        "release_year_range": (int(df['release_year'].min()), int(df['release_year'].max())),
        "unique_directors": int(df['director'].nunique()),
        "unique_countries": int(df['country'].nunique()),
        "unique_ratings": sorted(df['rating'].unique().tolist()),
    }
    return summary


if __name__ == "__main__":
    raw_df = load_dataset("Dataset.csv")
    cleaned_df = clean_data(raw_df)
    summary = get_dataset_summary(cleaned_df)
    print("Dataset Summary:")
    print(summary)
