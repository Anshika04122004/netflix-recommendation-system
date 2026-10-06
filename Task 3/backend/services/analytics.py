"""
Analytics service computing dataset insights and serving model performance analytics.
Auspify Technologies - Machine Learning Internship | Task 3
"""

import os
import json
import pandas as pd
import numpy as np


class AnalyticsService:
    def __init__(self, data_path='data/Dataset.csv', metrics_path='models/metrics.json'):
        self.data_path = data_path
        self.metrics_path = metrics_path
        self.dataset_stats = {}
        self.metrics_data = {}
        self.load_data()

    def load_data(self):
        # 1. Load Dataset
        actual_path = self.data_path
        if not os.path.exists(actual_path) and os.path.exists('Dataset.csv'):
            actual_path = 'Dataset.csv'

        if os.path.exists(actual_path):
            df = pd.read_csv(actual_path)
            self._compute_dataset_stats(df)

        # 2. Load Evaluation Metrics
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                self.metrics_data = json.load(f)

    def _compute_dataset_stats(self, df: pd.DataFrame):
        total_records = len(df)
        type_counts = df['type'].value_counts().to_dict()
        rating_counts = df['rating'].value_counts().to_dict()

        # Rating by content type cross-tabulation
        cross_tab = pd.crosstab(df['rating'], df['type']).to_dict()
        rating_by_type = []
        for r, count in rating_counts.items():
            rating_by_type.append({
                'rating': r,
                'Movie': int(cross_tab.get('Movie', {}).get(r, 0)),
                'TV Show': int(cross_tab.get('TV Show', {}).get(r, 0)),
                'total': int(count)
            })

        # Release year distribution (binned or recent decades)
        year_counts = df['release_year'].value_counts().sort_index()
        recent_years = year_counts[year_counts.index >= 2000].to_dict()
        release_trends = [{'year': int(y), 'count': int(c)} for y, c in recent_years.items()]

        # Top Countries
        country_counts = df['country'].replace('Not Given', 'Unknown').value_counts().head(10).to_dict()
        top_countries = [{'country': c, 'count': int(cnt)} for c, cnt in country_counts.items()]

        # Top Genres
        genre_dict = {}
        for item in df['listed_in'].dropna():
            for g in item.split(','):
                g = g.strip()
                if g:
                    genre_dict[g] = genre_dict.get(g, 0) + 1
        sorted_genres = sorted(genre_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        top_genres = [{'genre': g, 'count': c} for g, c in sorted_genres]

        # Duration analysis
        movies_df = df[df['type'] == 'Movie']
        movie_durations = movies_df['duration'].str.extract(r'(\d+)')[0].dropna().astype(int)
        
        tv_df = df[df['type'] == 'TV Show']
        tv_seasons = tv_df['duration'].str.extract(r'(\d+)')[0].dropna().astype(int)

        duration_stats = {
            'movie_avg_duration_min': round(float(movie_durations.mean()), 1) if len(movie_durations) else 0,
            'movie_median_duration_min': int(movie_durations.median()) if len(movie_durations) else 0,
            'tv_avg_seasons': round(float(tv_seasons.mean()), 1) if len(tv_seasons) else 0,
            'tv_max_seasons': int(tv_seasons.max()) if len(tv_seasons) else 0,
        }

        self.dataset_stats = {
            'total_records': total_records,
            'type_distribution': [{'name': k, 'value': int(v)} for k, v in type_counts.items()],
            'rating_distribution': [{'rating': k, 'count': int(v)} for k, v in rating_counts.items()],
            'rating_by_type': rating_by_type,
            'release_trends': release_trends,
            'top_countries': top_countries,
            'top_genres': top_genres,
            'duration_stats': duration_stats
        }

    def get_analytics(self) -> dict:
        if not self.dataset_stats or not self.metrics_data:
            self.load_data()

        # Prepare model comparison summary table
        model_comparison = []
        models = self.metrics_data.get('models', {})
        for name, data in models.items():
            model_comparison.append({
                'model_name': name,
                'accuracy': data.get('accuracy', 0),
                'precision_weighted': data.get('precision_weighted', 0),
                'recall_weighted': data.get('recall_weighted', 0),
                'f1_weighted': data.get('f1_weighted', 0),
                'f1_macro': data.get('f1_macro', 0)
            })

        # Pre vs Post Tuning summary
        tuning_comparison = [
            {
                'model': 'Decision Tree',
                'baseline_acc': models.get('Decision Tree (Baseline)', {}).get('accuracy', 0),
                'tuned_acc': models.get('Decision Tree (Tuned)', {}).get('accuracy', 0),
                'baseline_f1': models.get('Decision Tree (Baseline)', {}).get('f1_weighted', 0),
                'tuned_f1': models.get('Decision Tree (Tuned)', {}).get('f1_weighted', 0),
                'improvement_acc': round((models.get('Decision Tree (Tuned)', {}).get('accuracy', 0) - models.get('Decision Tree (Baseline)', {}).get('accuracy', 0)) * 100, 2),
                'improvement_f1': round((models.get('Decision Tree (Tuned)', {}).get('f1_weighted', 0) - models.get('Decision Tree (Baseline)', {}).get('f1_weighted', 0)) * 100, 2)
            },
            {
                'model': 'Random Forest',
                'baseline_acc': models.get('Random Forest (Baseline)', {}).get('accuracy', 0),
                'tuned_acc': models.get('Random Forest (Tuned)', {}).get('accuracy', 0),
                'baseline_f1': models.get('Random Forest (Baseline)', {}).get('f1_weighted', 0),
                'tuned_f1': models.get('Random Forest (Tuned)', {}).get('f1_weighted', 0),
                'improvement_acc': round((models.get('Random Forest (Tuned)', {}).get('accuracy', 0) - models.get('Random Forest (Baseline)', {}).get('accuracy', 0)) * 100, 2),
                'improvement_f1': round((models.get('Random Forest (Tuned)', {}).get('f1_weighted', 0) - models.get('Random Forest (Baseline)', {}).get('f1_weighted', 0)) * 100, 2)
            }
        ]

        best_model_name = self.metrics_data.get('best_model_name', 'Random Forest (Tuned)')
        best_model_metrics = models.get(best_model_name, {})

        return {
            'dataset': self.dataset_stats,
            'best_model_name': best_model_name,
            'best_model_metrics': best_model_metrics,
            'model_comparison': model_comparison,
            'tuning_comparison': tuning_comparison,
            'tuning_parameters': self.metrics_data.get('tuning_parameters', {}),
            'feature_importances': self.metrics_data.get('feature_importances', []),
            'classes': self.metrics_data.get('dataset_info', {}).get('classes', [])
        }
