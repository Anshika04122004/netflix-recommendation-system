"""
Standalone Pipeline Execution & Artifact Generation Script for Task 4
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.pipeline import NetflixSegmentationPipeline

def generate_static_visualizations(pipeline: NetflixSegmentationPipeline, out_dir: str = "static/charts"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Set sophisticated palette and style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'figure.facecolor': '#0f172a',
        'axes.facecolor': '#1e293b',
        'axes.edgecolor': '#334155',
        'axes.labelcolor': '#e2e8f0',
        'text.color': '#f8fafc',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'grid.color': '#334155',
        'grid.alpha': 0.6
    })

    # 1. Elbow & Silhouette Metric Chart
    k_res = pipeline.k_optim
    ks = [r['k'] for r in k_res]
    inertias = [r['inertia'] for r in k_res]
    silhouettes = [r['silhouette_score'] for r in k_res]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = '#38bdf8'
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12, fontweight='bold')
    l1 = ax1.plot(ks, inertias, color=color, marker='o', linewidth=2.5, markersize=7, label='Inertia (Elbow)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color2 = '#f43f5e'
    ax2.set_ylabel('Silhouette Score', color=color2, fontsize=12, fontweight='bold')
    l2 = ax2.plot(ks, silhouettes, color=color2, marker='s', linewidth=2.5, markersize=7, linestyle='--', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Highlight optimal K=6
    ax1.axvline(x=6, color='#fbbf24', linestyle=':', linewidth=2, label='Optimal Selection (K=6)')

    lines = l1 + l2 + [plt.Line2D([0], [0], color='#fbbf24', linestyle=':', linewidth=2)]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.8, facecolor='#1e293b', edgecolor='#475569')
    plt.title('K-Means Optimization: Elbow Curve & Silhouette Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbow_silhouette_optimization.png"), dpi=300)
    plt.close()

    # 2. PCA 2D Cluster Scatter Plot
    palette = ['#f43f5e', '#818cf8', '#f59e0b', '#10b981', '#38bdf8', '#a855f7']
    plt.figure(figsize=(11, 7))
    for c_id in range(6):
        sub = pipeline.df[pipeline.df['cluster'] == c_id]
        p_name = pipeline.profiles[c_id]['name']
        plt.scatter(
            sub['pca_x'], sub['pca_y'],
            c=palette[c_id % len(palette)],
            label=f"C{c_id}: {p_name}",
            alpha=0.65,
            s=28,
            edgecolors='none'
        )
    plt.title('Netflix Content Segmentation: 2D PCA Latent Space', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"Principal Component 1 ({pipeline.projections['explained_variance_2d'][0]*100:.1f}% Variance)", fontsize=11, fontweight='bold')
    plt.ylabel(f"Principal Component 2 ({pipeline.projections['explained_variance_2d'][1]*100:.1f}% Variance)", fontsize=11, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.8, facecolor='#1e293b', edgecolor='#475569', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_cluster_scatter.png"), dpi=300)
    plt.close()

    # 3. Cluster Size & Composition Bar Chart
    cluster_names = [f"C{c}: " + pipeline.profiles[c]['name'].split('&')[0].strip() for c in range(6)]
    movie_counts = []
    tv_counts = []
    for c in range(6):
        sub = pipeline.df[pipeline.df['cluster'] == c]
        movie_counts.append((sub['type'] == 'Movie').sum())
        tv_counts.append((sub['type'] == 'TV Show').sum())
        
    plt.figure(figsize=(10, 5.5))
    x_indices = np.arange(len(cluster_names))
    width = 0.38
    plt.bar(x_indices - width/2, movie_counts, width, label='Movies', color='#f43f5e', alpha=0.9, edgecolor='#fda4af')
    plt.bar(x_indices + width/2, tv_counts, width, label='TV Shows', color='#38bdf8', alpha=0.9, edgecolor='#7dd3fc')
    plt.xticks(x_indices, cluster_names, rotation=20, ha='right', fontsize=9.5)
    plt.ylabel('Number of Titles', fontsize=11, fontweight='bold')
    plt.title('Content Volume and Format Distribution Across Clusters', fontsize=14, fontweight='bold', pad=15)
    plt.legend(framealpha=0.8, facecolor='#1e293b', edgecolor='#475569')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_format_breakdown.png"), dpi=300)
    plt.close()

    print(f" Visualization charts exported to {out_dir}/")

def main():
    print("=" * 65)
    print("  NETFLIX CONTENT SEGMENTATION PIPELINE (TASK 4)")
    print("=" * 65)
    
    pipeline = NetflixSegmentationPipeline(data_path="Dataset.csv", optimal_k=6)
    results = pipeline.run_full_pipeline()
    
    # Save artifacts
    pipeline.save_artifacts("models")
    
    # Generate static charts
    generate_static_visualizations(pipeline, "static/charts")
    
    print("\n--- Pipeline Execution Summary ---")
    print(f"Total Titles Segmented: {results['summary_stats']['total_records']}")
    print(f"Optimal Clusters: {len(results['cluster_profiles'])}")
    print(f"Silhouette Score: {results['evaluation_metrics']['silhouette_score']}")
    print(f"Davies-Bouldin Score: {results['evaluation_metrics']['davies_bouldin_score']}")
    print("=" * 65)

if __name__ == "__main__":
    main()
