
# Project: Clustering of Chemical Process Data

This project focuses on clustering a dataset of 98 files, each representing a unique chemical process. The goal is to group these files into two distinct clusters based on the trends observed in the process variables.

## File Format

The data is stored in 98 separate text files within the `Data/` directory. Each file follows a Tab-Separated Values (TSV) format and contains 1,000 data points, representing the steps in the chemical process.

The files have a header row with the following 13 columns:
- `c`
- `T_PM`
- `d50`
- `d90`
- `d10`
- `T_TM`
- `mf_PM`
- `mf_TM`
- `Q_g`
- `w_crystal`
- `c_in`
- `T_PM_in`
- `T_TM_in`

## Clustering Methodology

To group the files, we explored three different clustering approaches. Given that each file represents a time-series of a chemical process, we used the rate of change (slope) of each variable as the primary feature for clustering.

### 1. Feature Extraction for Time-Series Data

Standard clustering algorithms are not designed to work directly with time-series data. To address this, we first needed to extract meaningful features that capture the temporal behavior of the chemical processes. We chose to represent each file by the rate of change of its variables over time.

For each of the 98 files, we performed the following steps:
- The data was read, and the header row was skipped.
- For each of the 13 columns, a linear regression model was fitted with the row number (from 1 to 1,000) as the independent variable (time) and the column's data as the dependent variable (process value).
- The slope of the regression line was calculated for each column. This slope represents the average rate of change of that variable over the entire process, providing a concise summary of its trend.
- This resulted in a feature vector of 13 slopes for each file, which was used for the clustering analysis.

This approach was chosen because it effectively captures the dynamic behavior of each process in a way that can be understood by traditional clustering algorithms.

### 2. Clustering Algorithms

We applied three different clustering methods to the extracted feature vectors:

- **K-Means Clustering**: This method partitions the data into *k* clusters by minimizing the variance within each cluster. We used *k*=2 to group the files into two clusters. It is a robust and widely used algorithm that is effective at identifying spherical clusters.

- **Agglomerative Clustering**: This hierarchical clustering method starts with each file as its own cluster and iteratively merges the closest clusters until only two remain. This method is useful because it does not assume spherical clusters and can reveal a hierarchical structure in the data.

- **K-Means with PCA**: To visualize the data and explore a different clustering perspective, we first applied Principal Component Analysis (PCA) to reduce the 13-dimensional feature vectors to 2 dimensions. We then applied the K-Means algorithm to this reduced dataset. This helps in visualizing the clusters and can sometimes improve clustering performance by removing noise.

### 3. Cluster Validation

To assess the quality of the clusters produced by each method, we used the **Silhouette Score**. This metric evaluates how well-separated the clusters are, with a score ranging from -1 to 1. A score closer to 1 indicates that the clusters are dense and well-defined.

## Results

The clustering results are presented in two forms:

- **Comparison Table**: A detailed table has been generated to show the cluster assignment for each file across the three different clustering methods. This allows for a direct comparison of how each file is categorized by the different algorithms.

- **Visualizations**: A PNG image named `cluster_comparison.png` has been created, which includes three scatter plots. Each plot visualizes the clusters for one of the three methods, with the Silhouette Score included in the title for easy validation.

This comprehensive approach provides a clear and validated grouping of the chemical process data, offering insights into the similarities and differences between the files.
