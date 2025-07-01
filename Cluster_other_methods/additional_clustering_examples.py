# -*- coding: utf-8 -*-
"""
Additional Clustering Methods Examples
This script demonstrates how to use various clustering methods with your chemical process data.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, SpectralClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing functions
from main import get_slopes, cluster_files

def dbscan_clustering_example(feature_vectors, file_names):
    """
    Example of DBSCAN clustering for outlier detection
    """
    print("=== DBSCAN Clustering Example ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_vectors)
    
    # Try different eps values
    eps_values = [0.3, 0.5, 0.7, 1.0]
    results = {}
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate silhouette score (excluding noise points)
        if n_clusters > 1:
            non_noise_mask = labels != -1
            if sum(non_noise_mask) > 1:
                silhouette = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        results[eps] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette
        }
        
        print(f"eps={eps}: {n_clusters} clusters, {n_noise} noise points, silhouette={silhouette:.3f}")
    
    # Plot best result
    best_eps = max(results.keys(), key=lambda x: results[x]['silhouette'])
    best_labels = results[best_eps]['labels']
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.title(f'DBSCAN Clustering (eps={best_eps})\nClusters: {results[best_eps]["n_clusters"]}, Noise: {results[best_eps]["n_noise"]}')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.savefig('dbscan_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def gmm_clustering_example(feature_vectors, file_names):
    """
    Example of Gaussian Mixture Model clustering
    """
    print("\n=== Gaussian Mixture Model Clustering Example ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_vectors)
    
    # Try different covariance types
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    results = {}
    
    for cov_type in covariance_types:
        gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=42)
        labels = gmm.fit_predict(X_scaled)
        
        # Get probabilities
        probabilities = gmm.predict_proba(X_scaled)
        max_prob = np.max(probabilities, axis=1)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, labels)
        
        results[cov_type] = {
            'labels': labels,
            'probabilities': probabilities,
            'max_prob': max_prob,
            'silhouette': silhouette,
            'aic': gmm.aic(X_scaled),
            'bic': gmm.bic(X_scaled)
        }
        
        print(f"{cov_type}: silhouette={silhouette:.3f}, AIC={gmm.aic(X_scaled):.1f}, BIC={gmm.bic(X_scaled):.1f}")
    
    # Plot best result
    best_cov = max(results.keys(), key=lambda x: results[x]['silhouette'])
    best_labels = results[best_cov]['labels']
    best_probs = results[best_cov]['max_prob']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.title(f'GMM Clustering ({best_cov})\nSilhouette: {results[best_cov]["silhouette"]:.3f}')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.colorbar(scatter)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_probs, 
                         cmap='plasma', alpha=0.7, s=50)
    plt.title('Cluster Assignment Confidence')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig('gmm_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def spectral_clustering_example(feature_vectors, file_names):
    """
    Example of Spectral clustering
    """
    print("\n=== Spectral Clustering Example ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_vectors)
    
    # Try different affinity metrics
    affinity_metrics = ['rbf', 'nearest_neighbors', 'cosine']
    results = {}
    
    for affinity in affinity_metrics:
        try:
            spectral = SpectralClustering(n_clusters=2, affinity=affinity, random_state=42)
            labels = spectral.fit_predict(X_scaled)
            
            silhouette = silhouette_score(X_scaled, labels)
            
            results[affinity] = {
                'labels': labels,
                'silhouette': silhouette
            }
            
            print(f"{affinity}: silhouette={silhouette:.3f}")
        except Exception as e:
            print(f"{affinity}: Error - {e}")
            results[affinity] = {'labels': None, 'silhouette': -1}
    
    # Plot best result
    best_affinity = max(results.keys(), key=lambda x: results[x]['silhouette'])
    if results[best_affinity]['labels'] is not None:
        best_labels = results[best_affinity]['labels']
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, 
                             cmap='viridis', alpha=0.7, s=50)
        plt.title(f'Spectral Clustering ({best_affinity})\nSilhouette: {results[best_affinity]["silhouette"]:.3f}')
        plt.xlabel('Feature 1 (Standardized)')
        plt.ylabel('Feature 2 (Standardized)')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.savefig('spectral_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results

def optics_clustering_example(feature_vectors, file_names):
    """
    Example of OPTICS clustering
    """
    print("\n=== OPTICS Clustering Example ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_vectors)
    
    # OPTICS clustering
    optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
    labels = optics.fit_predict(X_scaled)
    
    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"OPTICS: {n_clusters} clusters, {n_noise} noise points")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.title(f'OPTICS Clustering\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.savefig('optics_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'labels': labels, 'n_clusters': n_clusters, 'n_noise': n_noise}

def compare_all_methods(feature_vectors, file_names):
    """
    Compare all clustering methods
    """
    print("\n=== Comparing All Clustering Methods ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_vectors)
    
    # Run all methods
    results = {}
    
    # K-Means (from your original code)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    results['KMeans'] = {
        'labels': kmeans_labels,
        'silhouette': silhouette_score(X_scaled, kmeans_labels)
    }
    
    # Agglomerative
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(X_scaled)
    results['Agglomerative'] = {
        'labels': agg_labels,
        'silhouette': silhouette_score(X_scaled, agg_labels)
    }
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    if len(set(dbscan_labels)) > 1:
        non_noise_mask = dbscan_labels != -1
        if sum(non_noise_mask) > 1:
            dbscan_silhouette = silhouette_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
        else:
            dbscan_silhouette = -1
    else:
        dbscan_silhouette = -1
    results['DBSCAN'] = {
        'labels': dbscan_labels,
        'silhouette': dbscan_silhouette
    }
    
    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    results['GMM'] = {
        'labels': gmm_labels,
        'silhouette': silhouette_score(X_scaled, gmm_labels)
    }
    
    # Spectral
    spectral = SpectralClustering(n_clusters=2, random_state=42, affinity='rbf')
    spectral_labels = spectral.fit_predict(X_scaled)
    results['Spectral'] = {
        'labels': spectral_labels,
        'silhouette': silhouette_score(X_scaled, spectral_labels)
    }
    
    # Print comparison
    print("\nMethod Comparison:")
    print("-" * 50)
    for method, result in results.items():
        print(f"{method:12}: Silhouette = {result['silhouette']:.3f}")
    
    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['silhouette'])
    print(f"\nBest method: {best_method} (Silhouette = {results[best_method]['silhouette']:.3f})")
    
    return results

def main():
    """
    Main function to run all clustering examples
    """
    print("Additional Clustering Methods Examples")
    print("=" * 50)
    
    # Load your data (you'll need to modify this path)
    data_dir = "Data"
    
    # This would need to be integrated with your existing data loading
    # For now, we'll assume you have feature_vectors and file_names
    
    print("Note: This script demonstrates the clustering methods.")
    print("To use with your data, integrate the functions with your existing data loading code.")
    
    # Example usage:
    # feature_vectors, file_names = load_your_data()
    # dbscan_results = dbscan_clustering_example(feature_vectors, file_names)
    # gmm_results = gmm_clustering_example(feature_vectors, file_names)
    # spectral_results = spectral_clustering_example(feature_vectors, file_names)
    # optics_results = optics_clustering_example(feature_vectors, file_names)
    # all_results = compare_all_methods(feature_vectors, file_names)

if __name__ == "__main__":
    main() 