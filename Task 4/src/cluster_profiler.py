"""
Cluster Profiler & Business Intelligence Persona Generator
Interprets statistical traits of each cluster and creates executive personas and strategic insights.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from collections import Counter

class NetflixClusterProfiler:
    def __init__(self):
        # Bespoke luxury editorial aesthetic colors for cluster visuals
        self.cluster_meta = {
            0: {
                "name": "Mainstream Comedy & Family Features",
                "tagline": "Feel-Good, High-Engagement Domestic & International Comedies",
                "color": "#f43f5e", # Rose Ruby
                "badge": "Comedy & Family",
                "icon": "masks-theater",
                "target_demographic": "General Audience, Family Co-Viewing, Teen & Young Adults",
                "business_strategy": "High viewer retention; bundle with holiday release campaigns and cross-family syndication."
            },
            1: {
                "name": "Global Serialized Dramas & K-Dramas",
                "tagline": "Multi-Season International Storytelling & High-Tension TV Sagas",
                "color": "#818cf8", # Indigo Violet
                "badge": "International Series",
                "icon": "tv",
                "target_demographic": "Binge-Watchers, Subtitled Foreign Content Enthusiasts, K-Drama Community",
                "business_strategy": "Highest subscriber loyalty driver; invest heavily in multi-language dubbing and local production hubs."
            },
            2: {
                "name": "Heritage Cinema & Classic Era Archives",
                "tagline": "20th Century Masterpieces, Vintage Action, and Timeless Classics",
                "color": "#f59e0b", # Warm Amber
                "badge": "Vintage & Cult",
                "icon": "film",
                "target_demographic": "Cinephiles, Nostalgia Viewers, Film Scholars & Classic Buffs",
                "business_strategy": "Low licensing acquisition cost; provides high perceived catalog prestige and archival depth."
            },
            3: {
                "name": "Real-World Documentaries & Stand-Up Specials",
                "tagline": "In-Depth Investigative Series, Biographical Docs & Stand-Up Comedy",
                "color": "#10b981", # Emerald Teal
                "badge": "Docs & Specials",
                "icon": "microphone",
                "target_demographic": "Adult Learners, Non-Fiction Seekers, Comedy Club Fans",
                "business_strategy": "Viral social discussion generator; strong cultural zeitgeist momentum and awards eligibility."
            },
            4: {
                "name": "Youth Animation & Western Episodic Shows",
                "tagline": "Animated Adventures, Teen Sitcoms & Family-Friendly Docuseries",
                "color": "#38bdf8", # Sky Cyan
                "badge": "Kids & Youth TV",
                "icon": "shapes",
                "target_demographic": "Children (Toddlers to Pre-teens), Parents, Saturday-Morning Viewers",
                "business_strategy": "Lowest churn rate segment; crucial for family household retention and merchandise tie-ins."
            },
            5: {
                "name": "Global Indie Cinema & Emotional Dramas",
                "tagline": "Award-Winning International Storytelling, Arthouse & Deep Narratives",
                "color": "#a855f7", # Violet Amethyst
                "badge": "Indie & Arthouse",
                "icon": "clapperboard",
                "target_demographic": "Discerning Movie Enthusiasts, Global Festival Followers, Dramatic Art Fans",
                "business_strategy": "Key differentiator against rival streaming platforms; drives film festival accolades and critical acclaim."
            }
        }

    def profile_all_clusters(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Calculates comprehensive statistical profiles for each discovered content segment.
        """
        profiles = {}
        total_records = len(df)
        
        for c_id in sorted(df['cluster'].unique()):
            sub = df[df['cluster'] == c_id]
            count = len(sub)
            percentage = round((count / total_records) * 100, 2)
            
            # Genres breakdown
            all_genres = [g.strip() for glist in sub['listed_in'] for g in str(glist).split(',') if g.strip()]
            genre_counts = Counter(all_genres)
            top_genres = [{"genre": g, "count": cnt, "percent": round((cnt/count)*100, 1)} 
                          for g, cnt in genre_counts.most_common(5)]
            
            # Format breakdown
            type_counts = sub['type'].value_counts().to_dict()
            movie_pct = round((type_counts.get('Movie', 0) / count) * 100, 1)
            tv_pct = round((type_counts.get('TV Show', 0) / count) * 100, 1)
            
            # Rating breakdown
            rating_counts = sub['rating'].value_counts().head(4).to_dict()
            rating_dist = [{"rating": r, "count": cnt, "percent": round((cnt/count)*100, 1)}
                           for r, cnt in rating_counts.items()]
            
            # Top Countries
            country_counts = sub['country'].value_counts().head(5).to_dict()
            country_dist = [{"country": c, "count": cnt, "percent": round((cnt/count)*100, 1)}
                            for c, cnt in country_counts.items()]
            
            # Exemplar titles (representative sample)
            exemplar_titles = sub[['title', 'type', 'release_year', 'rating', 'duration', 'listed_in', 'country']].head(8).to_dict(orient='records')
            
            # Metadata
            meta = self.cluster_meta.get(c_id, {
                "name": f"Segment Cluster {c_id}",
                "tagline": "Discovered Unsupervised Content Archetype",
                "color": "#6366f1",
                "badge": "Content Segment",
                "icon": "layer-group",
                "target_demographic": "General Streaming Audience",
                "business_strategy": "Optimize catalog exposure and discoverability."
            })

            profiles[int(c_id)] = {
                "cluster_id": int(c_id),
                "name": meta["name"],
                "tagline": meta["tagline"],
                "color": meta["color"],
                "badge": meta["badge"],
                "icon": meta["icon"],
                "target_demographic": meta["target_demographic"],
                "business_strategy": meta["business_strategy"],
                "total_titles": count,
                "percentage_of_catalog": percentage,
                "movie_percent": movie_pct,
                "tv_show_percent": tv_pct,
                "mean_release_year": round(float(sub['release_year'].mean()), 1),
                "median_release_year": int(sub['release_year'].median()),
                "min_release_year": int(sub['release_year'].min()),
                "max_release_year": int(sub['release_year'].max()),
                "top_genres": top_genres,
                "rating_distribution": rating_dist,
                "top_countries": country_dist,
                "exemplar_titles": exemplar_titles
            }
            
        return profiles
