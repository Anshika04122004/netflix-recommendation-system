"""
Prediction & Inference Module for Netflix Content Type Classification.
Supports CLI inference and programmatic prediction for single items and batch data.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

# Add src to path if executed standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import load_pipeline_artifacts


class NetflixPredictor:
    """
    Production-grade predictor for Netflix Content Type inference.
    """
    def __init__(self, artifact_path: str = "models/content_type_pipeline.joblib"):
        bundle = load_pipeline_artifacts(artifact_path)
        self.model = bundle["model"]
        self.feature_extractor = bundle["feature_extractor"]
        self.tfidf_vectorizer = bundle["tfidf_vectorizer"]
        self.classes = bundle.get("classes", ["Movie", "TV Show"])

    def preprocess_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw input dataframe into model feature space.
        """
        df = input_df.copy()
        
        # Ensure all expected columns exist
        defaults = {
            'title': 'Unknown Title',
            'director': 'Unknown',
            'country': 'Unknown',
            'date_added': '1/1/2021',
            'release_year': 2021,
            'rating': 'TV-MA',
            'listed_in': 'Dramas'
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)
                
        # Transform tabular features
        tab_features = self.feature_extractor.transform(df)
        
        # Transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(df['title'].astype(str))
        tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_cols, index=tab_features.index)
        
        final_X = pd.concat([tab_features, tfidf_df], axis=1)
        return final_X

    def predict_single(self, title_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts content type for a single title dict.
        
        Example input:
        {
            "title": "Stranger Things",
            "director": "The Duffer Brothers",
            "country": "United States",
            "release_year": 2016,
            "rating": "TV-14",
            "listed_in": "TV Dramas, TV Sci-Fi & Fantasy, TV Mysteries",
            "date_added": "7/15/2016"
        }
        """
        df = pd.DataFrame([title_data])
        X = self.preprocess_input(df)
        
        # Predict probability
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            prob_movie = float(proba[0])
            prob_tv = float(proba[1])
        else:
            pred_class = int(self.model.predict(X)[0])
            prob_tv = 1.0 if pred_class == 1 else 0.0
            prob_movie = 1.0 - prob_tv

        predicted_label = "TV Show" if prob_tv >= 0.5 else "Movie"
        confidence = prob_tv if predicted_label == "TV Show" else prob_movie

        # Interpretability highlights
        insights = []
        rating = str(title_data.get('rating', ''))
        director = str(title_data.get('director', ''))
        genres = str(title_data.get('listed_in', ''))

        if rating.startswith('TV-'):
            insights.append("Rating follows TV broadcast parental guidelines (TV-MA/TV-14/TV-PG).")
        elif rating in ['PG-13', 'R', 'PG', 'G', 'NC-17']:
            insights.append("Rating is MPAA theatrical standard (R/PG-13), which strongly signals a Movie.")
            
        if director in ['Unknown', 'Not Given', '']:
            insights.append("No primary film director specified, which is common for episodic TV shows.")
        else:
            insights.append(f"Dedicated director listed: '{director}', typical for standalone feature films.")

        if any(kw in genres.lower() for kw in ['docuseries', 'series', 'reality', 'talk', 'anime series']):
            insights.append("Genre tags indicate episodic serialization.")

        return {
            "title": title_data.get("title", "Unknown"),
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Movie": round(prob_movie * 100, 2),
                "TV Show": round(prob_tv * 100, 2)
            },
            "key_factors": insights
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts content type for a DataFrame batch.
        """
        X = self.preprocess_input(df)
        preds = self.model.predict(X)
        
        results_df = df.copy()
        results_df['predicted_type'] = np.where(preds == 1, 'TV Show', 'Movie')
        
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            results_df['prob_movie'] = np.round(probas[:, 0] * 100, 2)
            results_df['prob_tv_show'] = np.round(probas[:, 1] * 100, 2)
            results_df['confidence'] = np.maximum(results_df['prob_movie'], results_df['prob_tv_show'])
            
        return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netflix Content Type Predictor CLI")
    parser.add_argument("--title", type=str, default="Breaking Bad", help="Title name")
    parser.add_argument("--director", type=str, default="Vince Gilligan", help="Director")
    parser.add_argument("--country", type=str, default="United States", help="Country")
    parser.add_argument("--year", type=int, default=2008, help="Release year")
    parser.add_argument("--rating", type=str, default="TV-MA", help="Parental Rating")
    parser.add_argument("--genres", type=str, default="Crime TV Shows, TV Dramas, TV Thrillers", help="Genres")
    args = parser.parse_args()

    sample = {
        "title": args.title,
        "director": args.director,
        "country": args.country,
        "release_year": args.year,
        "rating": args.rating,
        "listed_in": args.genres,
        "date_added": "8/2/2013"
    }

    try:
        predictor = NetflixPredictor("models/content_type_pipeline.joblib")
        result = predictor.predict_single(sample)
        print("\n=== Prediction Result ===")
        print(f"Title: {result['title']}")
        print(f"Prediction: {result['prediction']} ({result['confidence']}% confidence)")
        print(f"Probabilities: {result['probabilities']}")
        print("Key Explanations:")
        for factor in result['key_factors']:
            print(f" - {factor}")
    except Exception as e:
        print(f"Note: Train the model first by running main.py. Error: {e}")
