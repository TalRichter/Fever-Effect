import pandas as pd
import numpy as np
import scipy
import os
import glob
from scipy.stats import zscore, chi2
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, fisher_exact
from datetime import datetime
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2


def read_pathways_from_gmt(gmt_file_path, min_genes=5, max_genes=200):
    """
    Reads pathways from a GMT file and filters them based on gene count criteria and specific keywords in their names.

    Parameters:
    - gmt_file_path: Path to the GMT file.
    - min_genes: Minimum number of genes for a pathway to be included.
    - max_genes: Maximum number of genes for a pathway to be included.

    Returns:
    - A dictionary with pathway names as keys and lists of genes as values
    """
    pathways = {}
    with open(gmt_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            pathway_name = parts[0]
            gene_list = parts[2:]
            if min_genes <= len(gene_list) <= max_genes:
                pathways[pathway_name] = gene_list
    return pathways


def calculate_aggregate_mahalanobis_distances(data, pathways):
    """
    Calculates aggregate Mahalanobis distances for each sample across specified pathways.

    Parameters:
    - data: DataFrame containing gene expression data.
    - pathways: Dictionary of pathways and their associated genes.

    Returns:
    - A Series containing the mean Mahalanobis distance for each sample across all pathways.
    """
    distances = []

    for pathway_name, genes in pathways.items():
        filtered_genes = [gene for gene in genes if gene in data.columns]
        if not filtered_genes:
            continue

        pathway_data = data[filtered_genes]
        if pathway_data.empty:
            continue
        # Calculate covariance matrix and its inverse
        cov_matrix = np.cov(pathway_data.T)  # Ensure correct orientation for np.cov
        inv_cov_matrix = scipy.linalg.pinv(cov_matrix)  # Use pseudo-inverse for robustness

        # Calculate the Mahalanobis distance for each sample in the pathway
        centroid = np.mean(pathway_data, axis=0)
        pathway_distances = [mahalanobis(sample, centroid, inv_cov_matrix) for sample in pathway_data.values]
        distances.append(pathway_distances)

    # Calculate the mean distance across all pathways for each sample
    if distances:
        mean_distances = np.mean(np.array(distances, dtype=object), axis=0)
    else:
        mean_distances = np.array([])

    return pd.Series(mean_distances, index=data.index)


def calculate_zscore_outliers(data):
    """
    Identifies outliers in a dataset based on Z-scores.

    Parameters:
    - data: DataFrame of gene expression data.

    Returns:
    - A Series indicating outliers (1) and non-outliers (0) for each sample.
    """
    z_scores = np.abs(zscore(data, axis=0, nan_policy='omit'))
    outliers = (z_scores > 3).any(axis=1)
    return outliers.astype(int)


def calculate_mahalanobis_outliers(data, use_pca=True, pca_variance_threshold=0.90, outlier_threshold_p_value=0.99):
    """
    Calculates outliers based on Mahalanobis distance, optionally using PCA-reduced data.
    - data: DataFrame of gene expression data.
    - use_pca: Whether to use PCA reduction before calculating Mahalanobis distances.
    - pca_variance_threshold: Proportion of variance to maintain in PCA reduction (used if use_pca is True).
    - outlier_threshold_p_value: p-value threshold for determining outliers.
    Returns:
    - A Series indicating outliers (1) and non-outliers (0) for each sample.

    """

    if len(data) < 2 or len(data.columns) < 2:
        print("Insufficient data for Mahalanobis calculation.")
        return pd.Series([], dtype=int)  # Return an empty Series to indicate no calculation was done

    try:
        if use_pca:
            pca = PCA(n_components=pca_variance_threshold).fit(data)
            reduced_data = pca.transform(data)
            n_components = pca.n_components_
        else:
            reduced_data = data.values
            n_components = data.shape[1]

        centroid = np.mean(reduced_data, axis=0)
        cov_matrix = np.cov(reduced_data.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix) if np.linalg.cond(cov_matrix) < 1 / np.finfo(cov_matrix.dtype).eps else scipy.linalg.pinv(cov_matrix)
        mahalanobis_distances = np.array([mahalanobis(sample, centroid, inv_cov_matrix) for sample in reduced_data])
        outlier_threshold = np.sqrt(chi2.ppf(outlier_threshold_p_value, df=n_components))
        outliers = mahalanobis_distances > outlier_threshold
        return pd.Series(outliers.astype(int), index=data.index)
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error during Mahalanobis outlier detection: {e}")
        return pd.Series([], dtype=int)  # Return an empty Series as a fallbac



def identify_and_remove_total_outliers(data, pathways, condition_column='Target'):
    """
    Identifies and removes total outliers from the dataset based on Mahalanobis distances.

    Parameters:
    - data: DataFrame containing gene expression data and a condition column.
    - pathways: Dictionary of pathways and their associated genes.
    - condition_column: Name of the column indicating the condition (e.g., 'Target').

    Returns:
    - The cleaned DataFrame with total outliers removed.
    """
    # Calculate the aggregate Mahalanobis distances
    mean_distances = calculate_aggregate_mahalanobis_distances(data, pathways)

    # Determine the threshold for the top 90th percentile
    threshold = np.percentile(mean_distances.dropna(), 90)

    # Identify total outliers as those above the threshold
    total_outliers = mean_distances > threshold

    # Identify affected and non-affected outliers before removal
    affected_outliers = data.loc[total_outliers & (data[condition_column] == 1)]
    non_affected_outliers = data.loc[total_outliers & (data[condition_column] == 0)]

    # Print the number of affected and non-affected samples removed
    print(f"Number of affected samples removed as total outliers: {affected_outliers.shape[0]}")
    print(f"Number of non-affected samples removed as total outliers: {non_affected_outliers.shape[0]}")

    # Remove total outliers from the dataset
    clean_data = data.loc[~total_outliers]

    return clean_data


def detect_outliers(method, data, use_pca, pca_variance_threshold, outlier_threshold_p_value):
    """
    Detects outliers using the specified method ('mahalanobis' or 'zscore').

    Parameters:
    - method: Outlier detection method ('mahalanobis' or 'zscore').
    - data: DataFrame of gene expression data.
    - use_pca: Whether to use PCA reduction for 'mahalanobis' method.

    Returns:
    - A Series indicating outliers (1) and non-outliers (0) for each sample.
    """
    if method == 'mahalanobis':
        return calculate_mahalanobis_outliers(data, use_pca, pca_variance_threshold, outlier_threshold_p_value)
    elif method == 'zscore':
        return calculate_zscore_outliers(data)
    else:
        raise ValueError("Unsupported outlier detection method")


def compare_outlier_proportions(affected_outliers, not_affected_outliers):
    """
    Compares the proportions of outliers between affected and not affected samples.

    Parameters:
    - affected_outliers: Series indicating outliers in affected samples.
    - not_affected_outliers: Series indicating outliers in not affected samples.

    Returns:
    - The p-value from the statistical test comparing outlier proportions.
    """
    table = np.array([
        [affected_outliers.sum(), len(affected_outliers) - affected_outliers.sum()],
        [not_affected_outliers.sum(), len(not_affected_outliers) - not_affected_outliers.sum()]
    ])
    if np.any(table < 0):
        raise ValueError("Contingency table contains negative values, which is not valid.")
    if table.min() < 5:
        _, p_value = fisher_exact(table, alternative='two-sided')
    else:
        _, p_value, _, _ = chi2_contingency(table)
    return p_value



def aggregate_pathways_from_directory(directory_path):
    """
    Aggregates pathways from all GMT files within a directory.

    Parameters:
    - directory_path: Path to the directory containing GMT files.

    Returns:
    - A dictionary with aggregated pathways.
    """
    aggregated_pathways = {}
    # List all .gmt files in the directory
    gmt_files = glob.glob(f"{directory_path}/*.gmt")
    for gmt_file in gmt_files:
        pathways = read_pathways_from_gmt(gmt_file)
        aggregated_pathways.update(pathways)
    return aggregated_pathways


def perform_analysis_for_dataset(data_path, gmt_folder_path, output_folder_path, use_pca=True, outlier_threshold_p_value=0.99, pca_variance_threshold=0.90):
    print(f"Processing dataset: {data_path}")
    data = pd.read_csv(data_path)
    pathways = aggregate_pathways_from_directory(gmt_folder_path)
    print(f"Total pathways after filtering: {len(pathways)}")
    cleaned_data = identify_and_remove_total_outliers(data, pathways)

    results = []  # Initialize an empty list to store results

    print("Processing pathways...")
    for pathway_name, relevant_genes in pathways.items():
        filtered_genes = [gene for gene in relevant_genes if gene in cleaned_data.columns]
        if not filtered_genes:
            print(f"Skipping {pathway_name}: No matching genes.")
            continue

        filtered_data = cleaned_data[['Target'] + filtered_genes]
        if filtered_data.empty or len(filtered_genes) < 2:
            print(f"Skipping {pathway_name}: Insufficient data.")
            continue

        affected_data = filtered_data[filtered_data['Target'] == 1].drop(columns=['Target'])
        not_affected_data = filtered_data[filtered_data['Target'] == 0].drop(columns=['Target'])

        affected_data_numeric = affected_data.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
        not_affected_data_numeric = not_affected_data.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

        if affected_data_numeric.shape[1] < 2 or not_affected_data_numeric.shape[1] < 2:
            print(f"Skipping {pathway_name}: Not enough data after cleaning.")
            continue

        try:
            affected_outliers = detect_outliers('mahalanobis', affected_data_numeric, use_pca=use_pca, pca_variance_threshold=pca_variance_threshold, outlier_threshold_p_value=outlier_threshold_p_value)
            not_affected_outliers = detect_outliers('mahalanobis', not_affected_data_numeric, use_pca=use_pca, pca_variance_threshold=pca_variance_threshold, outlier_threshold_p_value=outlier_threshold_p_value)
        except np.linalg.LinAlgError as e:
            print(f"Error during outlier detection for {pathway_name}: {e}")
            continue

        if affected_outliers.empty or not_affected_outliers.empty:
            print(f"Skipping {pathway_name}: Outlier detection failed.")
            continue

        p_value = compare_outlier_proportions(affected_outliers, not_affected_outliers)

        results.append({
            'Pathway': pathway_name,
            'Proportion Affected Outliers': np.mean(affected_outliers),
            'Proportion Not Affected Outliers': np.mean(not_affected_outliers),
            'P-Value': p_value
        })

    p_values = [result['P-Value'] for result in results]
    _, p_adj_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    for result, p_adj in zip(results, p_adj_values):
        result['P-Adj'] = p_adj

    results_df = pd.DataFrame(results)
    filtered_results_df = results_df[
        (results_df['Proportion Affected Outliers'] > results_df['Proportion Not Affected Outliers']) &
        (results_df['P-Adj'] <= 0.05)
        ].sort_values('P-Adj', ascending=True)

    print("Filtered results:")
    print(filtered_results_df)
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{base_name}_results_{timestamp}.csv"
    output_csv_path = os.path.join(output_folder_path, output_file_name)

    header_info = f"Original File: {data_path} Use PCA: {use_pca} Outlier Threshold P-Value: {outlier_threshold_p_value} PCA Variance Threshold: {pca_variance_threshold}"

    with open(output_csv_path, 'a') as f:
        f.write(header_info + '\n')
    filtered_results_df.to_csv(output_csv_path, mode='a', index=False)
    print(f"Analysis completed for {data_path}. Results are saved to {output_csv_path}")


def main(dataset_folder_path, gmt_folder_path, output_folder_path, use_pca=True):
    """
    Main function to process all gene expression datasets within a specified folder.

    Parameters:
    - dataset_folder_path: Path to the folder containing gene expression data files.
    - gmt_folder_path: Path to the directory containing GMT files.
    - output_folder_path: Path to the directory where all results will be saved.
    - use_pca: Boolean indicating whether PCA should be used in outlier detection.
    """
    # Define the outlier p-value and PCA variance threshold
    outlier_p_value = 0.99 # You can change this to the p-value you want to use
    pca_variance_threshold = 0.90  # Set the PCA variance threshold
    data_paths = glob.glob(os.path.join(dataset_folder_path, "*.csv"))
    i=0
    # Process each data file
    for data_path in data_paths:
        perform_analysis_for_dataset(data_path, gmt_folder_path, output_folder_path, use_pca, outlier_p_value, pca_variance_threshold)


if __name__ == "__main__":
    dataset_folder_path = r"G:\My Drive\XGBoost_Fever_effect\mahalanobis_distance_analysis_pathways\files\"
    gmt_folder_path = r"G:\My Drive\XGBoost_Fever_effect\mahalanobis_distance_analysis_pathways\gmt_files\c5\bp"
    output_folder_path = r"G:\My Drive\XGBoost_Fever_effect\mahalanobis_distance_analysis_pathways\results\"
    use_pca = True 
    main(dataset_folder_path, gmt_folder_path, output_folder_path, use_pca)
