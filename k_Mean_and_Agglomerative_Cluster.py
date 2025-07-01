import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Extract slope features from the time-series data
def get_slopes(data):
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

def cluster_files():
    data_dir = "D:\\Projects\\MLME_Test\\Data"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not files:
        print("No data files found in the directory.")
        return

    feature_vectors = []
    file_names = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            lines = lines[1:]

            num_columns = -1
            for i, line in enumerate(lines):
                parts = line.strip().split('\t')
                if i == 0:
                    num_columns = len(parts)
                elif len(parts) != num_columns:
                    print(f"Warning: Inconsistent number of columns in {os.path.basename(file_path)}. Skipping.")
                    continue

            data = np.genfromtxt(file_path, delimiter="\t", skip_header=1)
            
            if data.size == 0 or np.isnan(data).all():
                print(f"Warning: Could not read data from {os.path.basename(file_path)}. Skipping.")
                continue

            feature_vector = get_slopes(data)

            if np.isnan(feature_vector).any():
                print(f"Warning: NaN values in feature vector for {os.path.basename(file_path)}. Replacing with 0.")
                feature_vector = np.nan_to_num(feature_vector)

            feature_vectors.append(feature_vector)
            file_names.append(os.path.basename(file_path))

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")

    if not feature_vectors:
        print("No valid data to cluster.")
        return

    feature_vectors = np.array(feature_vectors)

    # --- K-Means Clustering ---
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans_labels = kmeans.fit_predict(feature_vectors)
    kmeans_silhouette = silhouette_score(feature_vectors, kmeans_labels)

    # --- Agglomerative Clustering ---
    agg_clustering = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg_clustering.fit_predict(feature_vectors)
    agg_labels = 1 - agg_labels  # Inverting labels as requested
    agg_silhouette = silhouette_score(feature_vectors, agg_labels)

    # --- PCA and K-Means on PCA ---
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(feature_vectors)
    kmeans_pca = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans_pca_labels = kmeans_pca.fit_predict(reduced_features)
    kmeans_pca_silhouette = silhouette_score(reduced_features, kmeans_pca_labels)


    # --- Diagnostic Information ---
    print(f"\n--- Diagnostic Information ---")
    print(f"Original feature vectors shape: {feature_vectors.shape}")
    print(f"PCA reduced features shape: {reduced_features.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

    # Check if K-Means and K-Means PCA produce same results
    kmeans_vs_pca_same = np.array_equal(kmeans_labels, kmeans_pca_labels)
    print(f"K-Means and K-Means PCA produce same results: {kmeans_vs_pca_same}")

    if kmeans_vs_pca_same:
        print("This is normal! Your data clusters well in both original and PCA space.")
        print("The clusters are well-separated enough that dimensionality reduction preserves the structure.")

    # Show cluster centers comparison
    print(f"\nK-Means cluster centers (original space):")
    print(kmeans.cluster_centers_)
    print(f"\nK-Means PCA cluster centers (reduced space):")
    print(kmeans_pca.cluster_centers_)

    # --- Plotting ---
    plt.figure(figsize=(18, 6))

    # K-Means Plot
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=kmeans_labels, palette='viridis', s=100, alpha=0.7)
    plt.title(f"K-Means Clustering (Silhouette Score: {kmeans_silhouette:.2f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    # Agglomerative Clustering Plot
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=agg_labels, palette='viridis', s=100, alpha=0.7)
    plt.title(f"Agglomerative Clustering (Silhouette Score: {agg_silhouette:.2f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    # K-Means on PCA Plot
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=kmeans_pca_labels, palette='viridis', s=100, alpha=0.7)
    plt.title(f"K-Means on PCA (Silhouette Score: {kmeans_pca_silhouette:.2f})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("cluster_comparison.png")
    print("\nPlots saved to cluster_comparison.png")

    # --- Comparison Table ---
    df = pd.DataFrame({
        'File': file_names,
        'KMeans_Cluster': kmeans_labels,
        'Agglomerative_Cluster': agg_labels,
        'KMeans_PCA_Cluster': kmeans_pca_labels
    })

    print("\n--- Clustering Comparison Table ---")
    print(df.to_string())

    # --- Consistency and Interchange Analysis ---
    same_all = df[(df['KMeans_Cluster'] == df['Agglomerative_Cluster']) & (df['KMeans_Cluster'] == df['KMeans_PCA_Cluster'])]
    num_same = same_all.shape[0]
    print(f"\nNumber of files with the same cluster in all 3 methods: {num_same} out of {df.shape[0]}")
    if num_same > 0:
        print("Files with same cluster in all methods:")
        print(same_all[['File', 'KMeans_Cluster']].to_string(index=False, header=['File', 'Cluster']))

    # Files with different assignments (interchanged)
    interchanged = df[(df['KMeans_Cluster'] != df['Agglomerative_Cluster']) | (df['KMeans_Cluster'] != df['KMeans_PCA_Cluster']) | (df['Agglomerative_Cluster'] != df['KMeans_PCA_Cluster'])]
    num_interchanged = interchanged.shape[0]
    print(f"\nNumber of files with different cluster assignments (interchanged): {num_interchanged}")
    if num_interchanged > 0:
        print("Files with different assignments:")
        print(interchanged[['File', 'KMeans_Cluster', 'Agglomerative_Cluster', 'KMeans_PCA_Cluster']].to_string(index=False))

if __name__ == "__main__":
    cluster_files()