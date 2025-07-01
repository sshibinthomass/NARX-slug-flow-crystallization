# -*- coding: utf-8 -*-
"""
Cluster Assignment Table Generator
This script creates detailed tables showing which files belong to which cluster.
Can be used with existing clustering results or as a standalone tool.
"""

import pandas as pd
import numpy as np
import os

def create_cluster_table_from_results(file_names, cluster_labels, method_name="Clustering"):
    """
    Create a detailed table showing which files belong to which cluster
    
    Parameters:
    - file_names: List of file names
    - cluster_labels: Array of cluster labels
    - method_name: Name of the clustering method
    """
    
    # Create DataFrame
    df = pd.DataFrame({
        'File': file_names,
        'Cluster': cluster_labels
    })
    
    # Sort by cluster and then by filename
    df = df.sort_values(['Cluster', 'File'])
    
    # Count files in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    print(f"\n{method_name} Results")
    print("=" * 60)
    print(f"Total clusters found: {len(np.unique(cluster_labels))}")
    
    # Print cluster distribution
    print("\nCluster Distribution:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} files")
    
    # Print detailed file assignments
    print(f"\nDetailed File Assignments:")
    print(df.to_string(index=False, header=['File', 'Cluster']))
    
    # Save to CSV
    output_file = f'cluster_assignments_{method_name.lower().replace(" ", "_")}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed assignments saved to: {output_file}")
    
    return df

def create_comparison_table(file_names, results_dict):
    """
    Create a comparison table showing cluster assignments across multiple methods
    
    Parameters:
    - file_names: List of file names
    - results_dict: Dictionary with method names as keys and cluster labels as values
    """
    
    # Create DataFrame
    df = pd.DataFrame({'File': file_names})
    
    for method_name, cluster_labels in results_dict.items():
        df[f'{method_name}_Cluster'] = cluster_labels
    
    # Sort by first method's clusters, then by filename
    first_method = list(results_dict.keys())[0]
    df = df.sort_values([f'{first_method}_Cluster', 'File'])
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CLUSTER ASSIGNMENT COMPARISON")
    print("="*80)
    
    print("\nComprehensive comparison (first 20 files shown):")
    print(df.head(20).to_string(index=False))
    
    if len(df) > 20:
        print(f"\n... and {len(df) - 20} more files")
    
    # Save comprehensive table
    df.to_csv('comprehensive_cluster_assignments.csv', index=False)
    print(f"\nComprehensive comparison saved to: comprehensive_cluster_assignments.csv")
    
    return df

def analyze_consistency(file_names, results_dict):
    """
    Analyze consistency of cluster assignments across different methods
    
    Parameters:
    - file_names: List of file names
    - results_dict: Dictionary with method names as keys and cluster labels as values
    """
    
    # Create DataFrame with all cluster assignments
    df = pd.DataFrame({'File': file_names})
    for method_name, cluster_labels in results_dict.items():
        df[f'{method_name}_Cluster'] = cluster_labels
    
    # Find files with consistent assignments across all methods
    cluster_columns = [f'{method_name}_Cluster' for method_name in results_dict.keys()]
    
    # Check if all methods produce the same cluster assignment for each file
    df['All_Same'] = df[cluster_columns].nunique(axis=1) == 1
    
    consistent_files = df[df['All_Same'] == True]
    inconsistent_files = df[df['All_Same'] == False]
    
    print("\n" + "="*80)
    print("CLUSTER ASSIGNMENT CONSISTENCY ANALYSIS")
    print("="*80)
    
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
    
    method_names = list(results_dict.keys())
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

def example_usage():
    """
    Example of how to use this script with your existing clustering results
    """
    print("Cluster Assignment Table Generator")
    print("=" * 50)
    print("\nThis script can be used to create detailed cluster assignment tables.")
    print("\nExample usage:")
    print("1. Run your clustering analysis (e.g., main.py)")
    print("2. Use the results to create detailed tables:")
    print("   - create_cluster_table_from_results(file_names, kmeans_labels, 'KMeans')")
    print("   - create_comparison_table(file_names, results_dict)")
    print("   - analyze_consistency(file_names, results_dict)")
    
    print("\nExample with dummy data:")
    
    # Example data
    example_files = ['file_001.txt', 'file_002.txt', 'file_003.txt', 'file_004.txt', 'file_005.txt']
    kmeans_labels = [0, 0, 1, 1, 0]
    agg_labels = [0, 0, 1, 1, 1]
    
    # Create individual table
    print("\n--- Individual Method Table ---")
    create_cluster_table_from_results(example_files, kmeans_labels, "KMeans")
    
    # Create comparison table
    print("\n--- Comparison Table ---")
    results_dict = {
        'KMeans': kmeans_labels,
        'Agglomerative': agg_labels
    }
    create_comparison_table(example_files, results_dict)
    
    # Analyze consistency
    print("\n--- Consistency Analysis ---")
    analyze_consistency(example_files, results_dict)

def integrate_with_main():
    """
    Function to integrate with your existing main.py clustering results
    """
    print("\nTo integrate with your existing main.py:")
    print("1. Import this module in your main.py")
    print("2. After running clustering, call:")
    print("   create_cluster_table_from_results(file_names, kmeans_labels, 'KMeans')")
    print("   create_cluster_table_from_results(file_names, agg_labels, 'Agglomerative')")
    print("3. For comparison:")
    print("   results_dict = {'KMeans': kmeans_labels, 'Agglomerative': agg_labels}")
    print("   create_comparison_table(file_names, results_dict)")
    print("   analyze_consistency(file_names, results_dict)")

if __name__ == "__main__":
    example_usage()
    integrate_with_main() 