import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import math
from sklearn.cluster import SpectralCoclustering

from statsmodels.stats.multitest import multipletests

def find_enriched_clusters_with_gene_names(data_path, output_path, min_clusters=2, max_clusters=10, top_x_clusters=3, min_samples_group_1=5, significant_only=True, alpha=0.05, max_genes_per_cluster=1500):
    df = pd.read_csv(data_path, index_col=0)

    expression_data = df.drop('Target', axis=1)
    targets = df['Target']
    gene_names = expression_data.columns.tolist()

    enriched_clusters = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        model = SpectralBiclustering(n_clusters=n_clusters, random_state=0)
        model.fit(expression_data)

        for cluster_idx in range(n_clusters):
            row_indices = np.where(model.row_labels_ == cluster_idx)[0]
            col_indices = np.where(model.column_labels_ == cluster_idx)[0]
            # Filter clusters based on the number of genes
            if len(col_indices) > max_genes_per_cluster:
                continue  # Skip this cluster if it has more genes than specified


            cluster_targets = targets.iloc[row_indices]
            gene_names_in_cluster = [gene_names[i] for i in col_indices]

            a = np.sum(cluster_targets == 1)  # ASD responders in cluster
            b = np.sum(cluster_targets == 0)  # ASD non-responders in cluster
            c = np.sum(targets == 1) - a      # ASD responders not in the cluster
            d = np.sum(targets == 0) - b      # ASD non-responders not in the cluster

            if a >= min_samples_group_1:
                odds_ratio, p_value = fisher_exact([[a, b], [c, d]], 'greater')
                se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_low = math.exp(math.log(odds_ratio) - 1.96 * se_log_or)
                ci_upp = math.exp(math.log(odds_ratio) + 1.96 * se_log_or)

                enriched_clusters.append({
                    'n_clusters': n_clusters,
                    'cluster_idx': cluster_idx,
                    'p_value': p_value,
                    'adj_p_value': None,  # Placeholder for adjusted p-value
                    'odds_ratio': odds_ratio,
                    'ci_95_low': ci_low,
                    'ci_95_upp': ci_upp,
                    'num_responders_in_cluster': a,
                    'num_non_responders_in_cluster': b,
                    'total_samples_in_cluster': a + b,
                    'total_responders':  np.sum(targets == 1),
                    'total_non_responders': np.sum(targets == 0),
                    'number_of_genes': len(col_indices),
                    'gene_names': '\t'.join(gene_names_in_cluster)
                })

    # Adjust p-values for multiple comparisons, if there are any clusters to adjust
    if enriched_clusters:
        p_values = [cluster['p_value'] for cluster in enriched_clusters]
        adj_p_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[1]

        for cluster, adj_p_val in zip(enriched_clusters, adj_p_values):
            cluster['adj_p_value'] = adj_p_val

    # Filter, sort, and select top clusters
    enriched_clusters = [cluster for cluster in enriched_clusters if (significant_only and cluster['p_value'] <= alpha) or not significant_only]
    enriched_clusters.sort(key=lambda x: x['p_value'])
    top_clusters = enriched_clusters[:top_x_clusters]

    results_df = pd.DataFrame(top_clusters)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

# Example usage
# data_path = r"G:\My Drive\XGBoost_Fever_effect\limma_edgeR\new5\LogCounts_CPM_BeforeVoom_Allsamples_Target.csv"
data_path = r"C:\Users\talri\Documents\fever_effect_paper\counts_LogCPM_GIsamples_Target.csv"
output_path = "G:/My Drive/XGBoost_Fever_effect/biclustering/enriched_cluster_SpectralBiclustering_maxclustersRandom1.csv"
find_enriched_clusters_with_gene_names(data_path, output_path, min_clusters=3, max_clusters=20, top_x_clusters=5, min_samples_group_1=40, significant_only=True, alpha=1, max_genes_per_cluster=500)
