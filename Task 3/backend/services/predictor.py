"""
Prediction service for the Netflix Audience Rating Classifier API.
Loads trained models, applies real-time preprocessing, and returns predictions with confidence.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Detailed ratings taxonomy metadata
RATING_INFO = {
    'TV-MA': {
        'name': 'TV-MA (Mature Audience)',
        'badge': '18+',
        'level': 'Adults Only',
        'color': '#E50914',
        'description': 'Designed specifically for mature adult audiences. May contain intense violence, explicit language, and adult themes.'
    },
    'TV-14': {
        'name': 'TV-14 (Parents Strongly Cautioned)',
        'badge': '14+',
        'level': 'Teens 14+',
        'color': '#F59E0B',
        'description': 'Contains material that many parents would find unsuitable for children under 14 years of age.'
    },
    'TV-PG': {
        'name': 'TV-PG (Parental Guidance Suggested)',
        'badge': 'PG',
        'level': 'Older Kids / Family',
        'color': '#3B82F6',
        'description': 'Parental guidance suggested; program may contain material unsuitable for younger children.'
    },
    'R': {
        'name': 'R (Restricted)',
        'badge': '17+',
        'level': 'Restricted / Adults',
        'color': '#DC2626',
        'description': 'Requires accompanying parent or adult guardian for viewers under 17 years old.'
    },
    'PG-13': {
        'name': 'PG-13 (Parents Strongly Cautioned)',
        'badge': '13+',
        'level': 'Teens 13+',
        'color': '#8B5CF6',
        'description': 'Some material may be inappropriate for pre-teen children under the age of 13.'
    },
    'TV-Y7': {
        'name': 'TV-Y7 (Directed to Older Children)',
        'badge': '7+',
        'level': 'Kids 7+',
        'color': '#10B981',
        'description': 'Specifically designed for children age 7 and above; may include mild comedic or fantasy violence.'
    },
    'TV-Y': {
        'name': 'TV-Y (All Children)',
        'badge': 'Kids',
        'level': 'All Children',
        'color': '#06B6D4',
        'description': 'Appropriate for all children, including very young viewers ages 2 through 6.'
    },
    'PG': {
        'name': 'PG (Parental Guidance Suggested)',
        'badge': 'PG',
        'level': 'Parental Guidance',
        'color': '#6366F1',
        'description': 'Some material may not be suitable for young children; parental discretion advised.'
    },
    'TV-G': {
        'name': 'TV-G (General Audience)',
        'badge': 'All',
        'level': 'General Audience',
        'color': '#14B8A6',
        'description': 'Suitable for all ages. Little or no violence, mild humor, and family-friendly narrative.'
    },
    'NR': {
        'name': 'NR (Not Rated)',
        'badge': 'NR',
        'level': 'Unrated',
        'color': '#6B7280',
        'description': 'Content has not received a formal MPAA or TV Parental Guidelines board rating.'
    },
    'G': {
        'name': 'G (General Audiences)',
        'badge': 'G',
        'level': 'All Ages',
        'color': '#22C55E',
        'description': 'Admits all ages; contains no offensive material or inappropriate themes.'
    },
    'TV-Y7-FV': {
        'name': 'TV-Y7-FV (Directed to Older Children - Fantasy Violence)',
        'badge': '7+ FV',
        'level': 'Kids 7+ Fantasy',
        'color': '#059669',
        'description': 'Content suitable for children 7+, featuring distinct fantasy action or comedic combat.'
    },
    'NC-17': {
        'name': 'NC-17 (Adults Only)',
        'badge': '18+',
        'level': 'Explicit Adult',
        'color': '#991B1B',
        'description': 'Strictly adults only. No one 17 and under admitted due to graphic content.'
    },
    'UR': {
        'name': 'UR (Unrated)',
        'badge': 'UR',
        'level': 'Unrated / Extended',
        'color': '#4B5563',
        'description': 'Unrated version or international cut released outside standard rating classifications.'
    }
}


class RatingPredictorService:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        model_path = os.path.join(self.models_dir, 'best_model.pkl')
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')

        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.is_loaded = True
        else:
            self.is_loaded = False

    def predict(self, input_data: dict) -> dict:
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("Models are not yet trained or loaded. Please run the training pipeline first.")

        # Build raw input DataFrame (pass all relevant fields; TF-IDF on description/cast enriches predictions)
        raw_df = pd.DataFrame([{
            'type': input_data.get('type', 'Movie'),
            'director': input_data.get('director', 'Not Given'),
            'cast': input_data.get('cast', ''),
            'country': input_data.get('country', 'United States'),
            'date_added': input_data.get('date_added', '2021-09-25'),
            'release_year': input_data.get('release_year', 2020),
            'duration': input_data.get('duration', '90 min'),
            'listed_in': input_data.get('listed_in', 'Documentaries'),
            'description': input_data.get('description', ''),
        }])

        # Transform features
        X_trans = self.preprocessor.transform(raw_df)

        # Predict encoded class
        pred_encoded = self.model.predict(X_trans)[0]
        pred_class = self.preprocessor.decode_target([pred_encoded])[0]

        # Compute probability distribution across all classes if model supports predict_proba
        probabilities = {}
        confidence = 0.5
        classes = self.preprocessor.get_classes()

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_trans)[0]
            for cls_name, prob in zip(classes, probs):
                probabilities[cls_name] = round(float(prob), 4)
            confidence = round(float(np.max(probs)), 4)
        else:
            confidence = 1.0
            probabilities[pred_class] = 1.0

        # Sort probabilities descending
        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

        # Identify top active features for transparency
        top_features = []
        genres_input = [g.strip() for g in str(input_data.get('listed_in', '')).split(',') if g.strip()]
        for g in genres_input:
            top_features.append({'feature': f'Genre: {g}', 'impact': 'Active Category'})

        top_features.append({'feature': f"Content Type: {input_data.get('type', 'Movie')}", 'impact': 'Format'})
        top_features.append({'feature': f"Duration: {input_data.get('duration', '90 min')}", 'impact': 'Runtime'})
        top_features.append({'feature': f"Country: {input_data.get('country', 'United States')}", 'impact': 'Regional Context'})

        rating_details = RATING_INFO.get(pred_class, {
            'name': pred_class,
            'badge': pred_class,
            'level': 'General',
            'color': '#4B5563',
            'description': 'Audience rating category classified by model.'
        })

        return {
            'predicted_rating': pred_class,
            'confidence': confidence,
            'model_used': type(self.model).__name__,
            'rating_category_info': rating_details,
            'probabilities': sorted_probs,
            'top_contributing_features': top_features,
            'input_received': input_data
        }
