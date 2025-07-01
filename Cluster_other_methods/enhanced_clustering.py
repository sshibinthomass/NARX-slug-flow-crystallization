# -*- coding: utf-8 -*-
"""
Enhanced Clustering Analysis for Chemical Process Data
This script implements multiple clustering methods beyond basic K-means and Agglomerative clustering.
"""

import os
import numpy as np
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')

def get_slopes(data):
    """Extract slope features from time-series data"""
    slopes = []
    x = np.arange(data.shape[0]).reshape(-1, 1)
    for i in range(data.shape[1]):
        y = data[:, i].reshape(-1, 1)
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmean(y))
        model = LinearRegression()
        model.fit(x, y)
        slopes.append(model.coef_[0][0])
    return slopes

def get_advanced_features(data):
    """Extract additional features beyond just slopes"""
    features = []
    
    # Slopes (existing method)
    slopes = get_slopes(data)
    features.extend(slopes)
    
    # Additional statistical features
    for i in range(data.shape[1]):
        col_data = data[:, i]
        if np.isnan(col_data).any():
            col_data = np.nan_to_num(col_data, nan=np.nanmean(col_data))
        
        # Statistical moments
        features.append(np.mean(col_data))
        features.append(np.std(col_data))
        features.append(np.var(col_data))
        features.append(np.median(col_data))
        
        # Range and percentiles
        features.append(np.max(col_data) - np.min(col_data))
        features.append(np.percentile(col_data, 25))
        features.append(np.percentile(col_data, 75))
        
        # Trend features
        features.append(np.corrcoef(np.arange(len(col_data)), col_data)[0, 1])
    
    return features

def load_and_process_data(data_dir):
    """Load and process all data files"""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
    
    if not files:
        print("No data files found in the directory.")
        return None, None, None

    feature_vectors = []
    advanced_feature_vectors = []
    file_names = []

    for file_path in files:
        try:
            data = np.genfromtxt(file_path, delimiter="\t", skip_header=1)
            
            if data.size == 0 or np.isnan(data).all():
                print(f"Warning: Could not read data from {os.path.basename(file_path)}. Skipping.")
                continue

            # Basic slope features
            feature_vector = get_slopes(data)
            if np.isnan(feature_vector).any():
                feature_vector = np.nan_to_num(feature_vector, nan=0)
            
            # Advanced features
            advanced_features = get_advanced_features(data)
            if np.isnan(advanced_features).any():
                advanced_features = np.nan_to_num(advanced_features, nan=0)

            feature_vectors.append(feature_vector)
            advanced_feature_vectors.append(advanced_features)
            file_names.append(os.path.basename(file_path))

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")

    if not feature_vectors:
        print("No valid data to cluster.")
        return None, None, None

    return np.array(feature_vectors), np.array(advanced_feature_vectors), file_names

def evaluate_clustering(X, labels, method_name):
    """Evaluate clustering quality using multiple metrics"""
    if len(np.unique(labels)) < 2:
        return {"silhouette": -1, "calinski_harabasz": -1, "davies_bouldin": -1}
    
    try:
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        return {
            "silhouette": silhouette,
            "calinski_harabasz": calinski_harabasz,
            "davies_bouldin": davies_bouldin
        }
    except:
        return {"silhouette": -1, "calinski_harabasz": -1, "davies_bouldin": -1}

def perform_clustering_analysis(feature_vectors, advanced_features, file_names):
    """Perform comprehensive clustering analysis"""
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_vectors)
    X_advanced_scaled = scaler.fit_transform(advanced_features)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE for non-linear dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    results = {}
    
    # 1. K-Means Clustering
    print("Performing K-Means clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    results['KMeans'] = {
        'labels': kmeans_labels,
        'metrics': evaluate_clustering(X_scaled, kmeans_labels, 'KMeans')
    }
    
    # 2. Agglomerative Clustering with different linkage methods
    print("Performing Agglomerative clustering...")
    linkage_methods = ['ward', 'complete', 'average', 'single']
    for method in linkage_methods:
        agg = AgglomerativeClustering(n_clusters=2, linkage=method)
        agg_labels = agg.fit_predict(X_scaled)
        results[f'Agglomerative_{method}'] = {
            'labels': agg_labels,
            'metrics': evaluate_clustering(X_scaled, agg_labels, f'Agglomerative_{method}')
        }
    
    # 3. DBSCAN
    print("Performing DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    results['DBSCAN'] = {
        'labels': dbscan_labels,
        'metrics': evaluate_clustering(X_scaled, dbscan_labels, 'DBSCAN')
    }
    
    # 4. Spectral Clustering
    print("Performing Spectral clustering...")
    spectral = SpectralClustering(n_clusters=2, random_state=42, affinity='rbf')
    spectral_labels = spectral.fit_predict(X_scaled)
    results['Spectral'] = {
        'labels': spectral_labels,
        'metrics': evaluate_clustering(X_scaled, spectral_labels, 'Spectral')
    }
    
    # 5. Gaussian Mixture Models
    print("Performing Gaussian Mixture Model clustering...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    results['GMM'] = {
        'labels': gmm_labels,
        'metrics': evaluate_clustering(X_scaled, gmm_labels, 'GMM')
    }
    
    # 6. Bayesian Gaussian Mixture Models
    print("Performing Bayesian Gaussian Mixture Model clustering...")
    bgmm = BayesianGaussianMixture(n_components=2, random_state=42)
    bgmm_labels = bgmm.fit_predict(X_scaled)
    results['BGMM'] = {
        'labels': bgmm_labels,
        'metrics': evaluate_clustering(X_scaled, bgmm_labels, 'BGMM')
    }
    
    # 7. Advanced features clustering
    print("Performing clustering with advanced features...")
    kmeans_advanced = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_advanced_labels = kmeans_advanced.fit_predict(X_advanced_scaled)
    results['KMeans_Advanced'] = {
        'labels': kmeans_advanced_labels,
        'metrics': evaluate_clustering(X_advanced_scaled, kmeans_advanced_labels, 'KMeans_Advanced')
    }
    
    return results, X_pca, X_tsne

def create_comprehensive_visualization(results, X_pca, X_tsne, file_names):
    """Create comprehensive visualization of all clustering methods"""
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    methods = list(results.keys())
    
    for i, method in enumerate(methods):
        if i >= len(axes):
            break
            
        labels = results[method]['labels']
        metrics = results[method]['metrics']
        
        # Use PCA for visualization
        scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                cmap='viridis', alpha=0.7, s=50)
        axes[i].set_title(f'{method}\nSilhouette: {metrics["silhouette"]:.3f}')
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(methods), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('comprehensive_clustering_results.png', dpi=300, bbox_inches='tight')
    print("Comprehensive clustering visualization saved as 'comprehensive_clustering_results.png'")
    
    # t-SNE visualization
    plt.figure(figsize=(15, 5))
    
    # Show t-SNE with best performing method
    best_method = max(results.keys(), key=lambda x: results[x]['metrics']['silhouette'])
    best_labels = results[best_method]['labels']
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=best_labels, cmap='viridis', alpha=0.7, s=50)
    plt.title(f't-SNE: {best_method} (Best Silhouette Score)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis', alpha=0.7, s=50)
    plt.title(f'PCA: {best_method}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig('best_clustering_visualization.png', dpi=300, bbox_inches='tight')
    print("Best clustering visualization saved as 'best_clustering_visualization.png'")

def create_comparison_table(results, file_names):
    """Create comprehensive comparison table"""
    
    # Create DataFrame with all results
    df_data = {'File': file_names}
    
    for method, result in results.items():
        df_data[f'{method}_Cluster'] = result['labels']
        df_data[f'{method}_Silhouette'] = result['metrics']['silhouette']
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    df.to_csv('clustering_comparison_results.csv', index=False)
    print("Detailed comparison table saved as 'clustering_comparison_results.csv'")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CLUSTERING METHODS COMPARISON SUMMARY")
    print("="*80)
    
    summary_data = []
    for method, result in results.items():
        metrics = result['metrics']
        summary_data.append({
            'Method': method,
            'Silhouette Score': f"{metrics['silhouette']:.3f}",
            'Calinski-Harabasz': f"{metrics['calinski_harabasz']:.1f}",
            'Davies-Bouldin': f"{metrics['davies_bouldin']:.3f}",
            'Clusters Found': len(np.unique(result['labels']))
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['metrics']['silhouette'])
    print(f"\nBest performing method: {best_method} (Silhouette Score: {results[best_method]['metrics']['silhouette']:.3f})")
    
    return df

def create_detailed_cluster_tables(results, file_names):
    """Create detailed tables showing which files belong to which cluster for each method"""
    
    print("\n" + "="*80)
    print("DETAILED CLUSTER ASSIGNMENTS BY METHOD")
    print("="*80)
    
    # Create separate tables for each method
    for method, result in results.items():
        labels = result['labels']
        metrics = result['metrics']
        
        # Create DataFrame for this method
        method_df = pd.DataFrame({
            'File': file_names,
            'Cluster': labels
        })
        
        # Sort by cluster and then by filename
        method_df = method_df.sort_values(['Cluster', 'File'])
        
        # Count files in each cluster
        cluster_counts = method_df['Cluster'].value_counts().sort_index()
        
        print(f"\n{method} Clustering (Silhouette Score: {metrics['silhouette']:.3f})")
        print("-" * 60)
        print(f"Total clusters found: {len(np.unique(labels))}")
        
        # Print cluster distribution
        print("\nCluster Distribution:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} files")
        
        # Print detailed file assignments
        print(f"\nDetailed File Assignments:")
        print(method_df.to_string(index=False, header=['File', 'Cluster']))
        
        # Save individual method table
        method_df.to_csv(f'cluster_assignments_{method.lower().replace(" ", "_")}.csv', index=False)
        print(f"\nDetailed assignments saved to: cluster_assignments_{method.lower().replace(' ', '_')}.csv")
        
        # Export to TXT file for K-Means specifically
        if method == 'KMeans':
            export_kmeans_to_txt(method_df, metrics)
        
        print("\n" + "-" * 60)
    
    # Create a comprehensive table with all methods
    print("\n" + "="*80)
    print("COMPREHENSIVE CLUSTER ASSIGNMENT COMPARISON")
    print("="*80)
    
    comprehensive_df = pd.DataFrame({'File': file_names})
    
    for method, result in results.items():
        comprehensive_df[f'{method}_Cluster'] = result['labels']
    
    # Sort by first method's clusters, then by filename
    first_method = list(results.keys())[0]
    comprehensive_df = comprehensive_df.sort_values([f'{first_method}_Cluster', 'File'])
    
    print("\nComprehensive comparison (first 20 files shown):")
    print(comprehensive_df.head(20).to_string(index=False))
    
    if len(comprehensive_df) > 20:
        print(f"\n... and {len(comprehensive_df) - 20} more files")
    
    # Save comprehensive table
    comprehensive_df.to_csv('comprehensive_cluster_assignments.csv', index=False)
    print(f"\nComprehensive comparison saved to: comprehensive_cluster_assignments.csv")
    
    return comprehensive_df

def export_kmeans_to_txt(kmeans_df, metrics):
    """Export K-Means cluster assignments to a formatted TXT file"""
    
    # Create a formatted text file
    with open('kmeans_cluster_assignments.txt', 'w', encoding='utf-8') as f:
        f.write("K-MEANS CLUSTERING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Write summary information
        f.write(f"Silhouette Score: {metrics['silhouette']:.3f}\n")
        f.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.1f}\n")
        f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}\n")
        f.write(f"Total Clusters: {len(kmeans_df['Cluster'].unique())}\n\n")
        
        # Write cluster distribution
        cluster_counts = kmeans_df['Cluster'].value_counts().sort_index()
        f.write("CLUSTER DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for cluster, count in cluster_counts.items():
            f.write(f"Cluster {cluster}: {count} files\n")
        f.write("\n")
        
        # Write detailed file assignments
        f.write("DETAILED FILE ASSIGNMENTS:\n")
        f.write("-" * 30 + "\n")
        f.write("File\t\t\tCluster\n")
        f.write("-" * 30 + "\n")
        
        for _, row in kmeans_df.iterrows():
            f.write(f"{row['File']:<20}\t{row['Cluster']}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("End of K-Means Clustering Results\n")
    
    print("K-Means results exported to: kmeans_cluster_assignments.txt")

def analyze_cluster_consistency(results, file_names):
    """Analyze consistency of cluster assignments across different methods"""
    
    print("\n" + "="*80)
    print("CLUSTER ASSIGNMENT CONSISTENCY ANALYSIS")
    print("="*80)
    
    # Create DataFrame with all cluster assignments
    df = pd.DataFrame({'File': file_names})
    for method, result in results.items():
        df[f'{method}_Cluster'] = result['labels']
    
    # Find files with consistent assignments across all methods
    cluster_columns = [f'{method}_Cluster' for method in results.keys()]
    
    # Check if all methods produce the same cluster assignment for each file
    df['All_Same'] = df[cluster_columns].nunique(axis=1) == 1
    
    consistent_files = df[df['All_Same'] == True]
    inconsistent_files = df[df['All_Same'] == False]
    
    print(f"\nFiles with consistent cluster assignments across all methods: {len(consistent_files)}")
    print(f"Files with different cluster assignments: {len(inconsistent_files)}")
    
    if len(consistent_files) > 0:
        print("\nConsistent files:")
        print(consistent_files[['File'] + cluster_columns].to_string(index=False))
    
    if len(inconsistent_files) > 0:
        print("\nInconsistent files (first 10 shown):")
        print(inconsistent_files[['File'] + cluster_columns].head(10).to_string(index=False))
        if len(inconsistent_files) > 10:
            print(f"... and {len(inconsistent_files) - 10} more files")
    
    # Analyze pairwise consistency between methods
    print("\nPairwise Method Consistency:")
    print("-" * 50)
    
    method_names = list(results.keys())
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method1 = method_names[i]
            method2 = method_names[j]
            
            col1 = f'{method1}_Cluster'
            col2 = f'{method2}_Cluster'
            
            same_assignments = (df[col1] == df[col2]).sum()
            total_files = len(df)
            consistency_percent = (same_assignments / total_files) * 100
            
            print(f"{method1:15} vs {method2:15}: {same_assignments:3d}/{total_files:3d} ({consistency_percent:5.1f}%)")
    
    return df

def export_clusters_as_python_lists(results, file_names, output_file='cluster_lists.py'):
    """Export cluster assignments as Python lists for each method and cluster."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Auto-generated cluster assignments as Python lists\n\n')
        for method, result in results.items():
            labels = result['labels']
            method_name = method.replace(' ', '_').replace('-', '_')
            unique_clusters = sorted(set(labels))
            for cluster in unique_clusters:
                files_in_cluster = [file_names[i] for i, lbl in enumerate(labels) if lbl == cluster]
                pylist = ', '.join([repr(x) for x in files_in_cluster])
                f.write(f'{method_name}_cluster{cluster} = [{pylist}]\n')
        f.write('\n# End of cluster assignments\n')
    print(f"Cluster assignments exported as Python lists to: {output_file}")

def main():
    """Main function to run comprehensive clustering analysis"""
    
    # Update this path to match your data directory
    data_dir = "Data"  # Assuming Data folder is in current directory
    
    print("Loading and processing data...")
    feature_vectors, advanced_features, file_names = load_and_process_data(data_dir)
    
    if feature_vectors is None:
        return
    
    print(f"Processed {len(file_names)} files successfully")
    print(f"Basic feature vector shape: {feature_vectors.shape}")
    print(f"Advanced feature vector shape: {advanced_features.shape}")
    
    print("\nPerforming comprehensive clustering analysis...")
    results, X_pca, X_tsne = perform_clustering_analysis(feature_vectors, advanced_features, file_names)
    
    print("\nCreating visualizations...")
    create_comprehensive_visualization(results, X_pca, X_tsne, file_names)
    
    print("\nCreating comparison table...")
    df = create_comparison_table(results, file_names)
    
    print("\nCreating detailed cluster assignment tables...")
    detailed_df = create_detailed_cluster_tables(results, file_names)
    
    print("\nAnalyzing cluster assignment consistency...")
    consistency_df = analyze_cluster_consistency(results, file_names)
    
    # Export as Python lists
    export_clusters_as_python_lists(results, file_names)
    
    print("\nAnalysis complete! Check the generated files:")
    print("- comprehensive_clustering_results.png")
    print("- best_clustering_visualization.png") 
    print("- clustering_comparison_results.csv")
    print("- comprehensive_cluster_assignments.csv")
    print("- cluster_assignments_[method].csv (individual method files)")
    print("- cluster_lists.py (Python lists for all clusters)")

if __name__ == "__main__":
    main() 