# -*- coding: utf-8 -*-
"""
Export clusters from main.py output as Python lists.
This script reads the clustering assignments from main.py and writes a .py file with lists like:
K_Mean_cluster1 = [...]
K_Mean_cluster2 = [...]
K_MeanPCA_cluster1 = [...]
K_MeanPCA_cluster2 = [...]
Agglomerative_cluster1 = [...]
Agglomerative_cluster2 = [...]
"""

import pandas as pd

# Change this if you use a different file name
CLUSTER_CSV = 'clustering_comparison_results.csv'
OUTPUT_PY = 'main_clusters_lists.py'

def export_main_clusters_as_python_lists(csv_file=CLUSTER_CSV, output_file=OUTPUT_PY):
    df = pd.read_csv(csv_file)
    method_map = {
        'KMeans_Cluster': 'K_Mean',
        'KMeans_PCA_Cluster': 'K_MeanPCA',
        'Agglomerative_Cluster': 'Agglomerative',
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Auto-generated cluster assignments from main.py\n\n')
        for col, pyname in method_map.items():
            for cluster in sorted(df[col].unique()):
                files = df.loc[df[col] == cluster, 'File'].tolist()
                pylist = ', '.join([repr(x) for x in files])
                f.write(f'{pyname}_cluster{cluster+1} = [{pylist}]\n')
        f.write('\n# End of cluster assignments\n')
    print(f"Cluster assignments exported as Python lists to: {output_file}")

if __name__ == '__main__':
    export_main_clusters_as_python_lists() 