
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    roc_curve, auc, balanced_accuracy_score, average_precision_score, matthews_corrcoef)
import shap
from sklearn.utils import resample
import json
import sys
import os
import argparse



# Function Definitions
def load_data(gene_expression_file, label_file):
    """Load gene expression and label data."""
    gene_expression = pd.read_csv(gene_expression_file, index_col=0, sep="\t")
    label_group_all = pd.read_csv(label_file, index_col=0)
    return gene_expression, label_group_all

def preprocess_data(gene_expression, label_group_all):
    """Preprocess data by filtering, encoding, and removing rows with missing values."""
    label_group_all['cluster_verbal_iq'] = label_group_all['verbal_iq'].apply(lambda x: 'high' if x >= 70 else 'low')
    ind_info = label_group_all[['sex', 'race', 'any_gastro_disorders_proband',
                                'any_neurological_disorders_proband',
                                'any_neurological_disorders_in_family',
                                'any_autoimmune_disease_proband',
                                'any_autoimmune_disease_in_family',
                                'any_chronic_illnesses_in_family',
                                'any_maternal_infections_in_preg',
                                'any_autoimmune_disease_mat',
                                'any_language_disorders_in_family',
                                'cluster_verbal_iq', 'nonverbal_iq', 'verbal_iq']]
    ind_info_cleaned = ind_info.dropna()
    row_names = ind_info_cleaned.index
    gene_expression_cleaned = gene_expression.loc[row_names].T
    return gene_expression_cleaned, ind_info_cleaned

def convert_target_to_binary(label_group_all, target_column='Target'):
    """Convert the target column to a binary format."""
    if target_column not in label_group_all.columns:
        raise KeyError(f"Column '{target_column}' not found in the dataframe.")
    Target = label_group_all.loc[:, target_column]
    Target = pd.DataFrame(Target.replace(to_replace=['no', 'yes'], value=[0, 1]))
    Target_np = np.array(Target)
    return Target, Target_np

def feature_engineering(gene_expression, ind_info):
    """Feature engineering and selection."""
    X_encoded = pd.get_dummies(ind_info)
    gene_expression = variance_threshold_selector(gene_expression.T, threshold=0.1)
    X_encoded = X_encoded.reindex(gene_expression.index)
    combined = gene_expression.merge(X_encoded, left_index=True, right_index=True)
    return combined

def variance_threshold_selector(data, threshold=0.2):
    """Remove features with low variance."""
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def train_test_split_and_scale(combined, target):
    """Split the data into training and testing sets and scale features."""
    X_train, X_test, y_train, y_test = train_test_split(combined, target, test_size=0.2, random_state=0)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test

def kmeans_clustering(X_train, X_test, n_clusters=30, n_init=10):
    """Apply KMeans clustering to the dataset."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    X_train['Cluster'] = kmeans.fit_predict(X_train)
    X_test['Cluster'] = kmeans.predict(X_test)
    return X_train, X_test

def feature_selection(X_train, y_train, n_features):
    """Select important features using Lasso."""
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    important_genes = np.abs(lasso.coef_).argsort()[-n_features:][::-1]
    return X_train.iloc[:, important_genes], important_genes


def train_xgboost(X_train, y_train, params, cv_method='grid', cv_folds=5, n_jobs=10, n_iter=50):
    """Train an XGBoost model using specified cross-validation method."""
    model = XGBClassifier(objective='binary:logistic', seed=1, colsample_bytree=0.5, use_label_encoder=False)
    if cv_method == 'grid':
        optimal_params = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='roc_auc',
            verbose=0,
            n_jobs=n_jobs,
            cv=cv_folds
        )
    elif cv_method == 'random':
        optimal_params = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            scoring='roc_auc',
            verbose=0,
            n_jobs=n_jobs,
            cv=cv_folds,
            n_iter=n_iter
        )
    else:
        raise ValueError("cv_method must be either 'grid' or 'random'")
    
    optimal_params.fit(X_train, y_train)
    return optimal_params

def evaluate_model(model, X_test, y_test, plot=False):
    """Evaluate the performance of the model."""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
    average_precision = average_precision_score(y_test, y_probs)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    return {
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': roc_auc,
    }

def multiple_downsampling(data, target_column, iterations=5):
    """Perform multiple downsampling iterations of the majority class."""
    downsampled_datasets = []
    majority_class = data[data[target_column] == 0]
    minority_class = data[data[target_column] == 1]
    for i in range(iterations):
        majority_downsampled = resample(majority_class,
                                        replace=False,
                                        n_samples=len(minority_class),
                                        random_state=i)
        downsampled_data = pd.concat([majority_downsampled, minority_class])
        downsampled_datasets.append(downsampled_data)
    return downsampled_datasets

def parse_feature_selection_options(option_str):
    # Convert a comma-separated string to a list of integers
    return [int(x.strip()) for x in option_str.split(',')]



# Main Script
if __name__ == "__main__":
    print(sys.argv)

    parser = argparse.ArgumentParser(description='Process model parameters.')

    # Define arguments
    parser.add_argument('-FS', '--feature_selection', default='10,20', help='Feature selection options (comma-separated)')
    parser.add_argument('-DS', '--downsampling', type=int, default=5, help='Number of downsampling iterations')
    parser.add_argument('-TF', '--top_features', type=int, default=10, help='Number of top features')
    parser.add_argument('-OD', '--output_directory', default='/gpfs0/alal/users/richtert/XGBoost_Fever_effect/output/', help='Directory for saving models')
    parser.add_argument('-OF', '--output_file', default='/gpfs0/alal/users/richtert/XGBoost_Fever_effect/summary/output.csv', help='File for saving evaluations')
    parser.add_argument('-GE', '--gene_expression_file', default="/gpfs0/alal/users/richtert/XGBoost_Fever_effect/files/symbol_gi_log.txt", help='Gene expression data file')
    parser.add_argument('-LF', '--label_file', default="/gpfs0/alal/users/richtert/XGBoost_Fever_effect/files/sampleTable_GI.csv", help='Label file')
    parser.add_argument('-CVM', '--cv_method', default='grid', choices=['grid', 'random'], help='Cross-validation method: grid or random')


    # Parse arguments
    args = parser.parse_args()

    # Use arguments
    feature_selection_options = parse_feature_selection_options(args.feature_selection)
    n_downsampling = args.downsampling
    top_features = args.top_features
    output_file = args.output_file
    gene_expression_file = args.gene_expression_file
    label_file = args.label_file
    cv_method = args.cv_method
    output_directory = args.output_directory

    # Directories for storing models, parameters, and results
    models_directory = os.path.join(args.output_directory, 'models')
    params_directory = os.path.join(args.output_directory, 'params')
    results_directory = os.path.join(args.output_directory, 'results')
    shap_directory = os.path.join(args.output_directory, 'shap_results')


    # Create directories if they don't exist
    for directory in [models_directory, params_directory, results_directory, shap_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)



    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    # Load data
    gene_expression, label_group_all = load_data(gene_expression_file, label_file)
    Target, Target_np = convert_target_to_binary(label_group_all)
    gene_expression, ind_info = preprocess_data(gene_expression, label_group_all)
    combined = feature_engineering(gene_expression, ind_info)
    Target = Target.reindex(combined.index)
    combined_with_target = combined.join(Target)

    # Define the hyperparameter grid for XGBoost
    params = {
        'learning_rate': [0.03, 0.04, 0.05],
        'max_depth': [8, 9, 10],
        'colsample_bylevel': [0.4, 0.5, 0.6],
        'lambda': [1, 2 ,3],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.2, 0.5],
        'subsample': [ 0.7, 0.8, 1],
        'reg_alpha': [0, 0.5, 1],
    }

    # Feature selection options and results storage
    all_results = []  # To store aggregated results
    models = {}
    shap_results = {}
    SHAP_mean_and_corr = {}

    # Multiple downsampling
    downsampled_datasets = multiple_downsampling(combined_with_target, 'Target', iterations=n_downsampling)

    for n_features in feature_selection_options:
        for i, dataset in enumerate(downsampled_datasets):
            X_downsampled = dataset.drop('Target', axis=1)
            y_downsampled = dataset['Target']
            X_train, X_test, y_train, y_test = train_test_split_and_scale(X_downsampled, y_downsampled)
            X_train, X_test = kmeans_clustering(X_train, X_test)

            # Feature selection
            X_train_fs, important_genes = feature_selection(X_train, y_train, n_features)
            X_test_fs = X_test.iloc[:, important_genes]

            # Cross-validation setup
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_train_fs, y_train)):
                X_train_fold, X_test_fold = X_train_fs.iloc[train_idx], X_train_fs.iloc[test_idx]
                y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

                # Train the model
                grid_search = train_xgboost(X_train_fold, y_train_fold, params, cv_method=cv_method)
                model = grid_search.best_estimator_  # Extract the best estimator
                

                # Save the model in the specified directory
                model_file_path = os.path.join(models_directory, f'model_{n_features}_{i}_{fold_idx}.json')
                model.save_model(model_file_path)  # Save the model

    
                # Get the model parameters
                model_params = model.get_xgb_params()
                model_params.update({'n_features': n_features, 'iteration': i, 'fold_idx': fold_idx})
    
    
                # Save the model parameters in the specified directory
                params_file_path = os.path.join(params_directory, f'model_params_{n_features}_{i}_{fold_idx}.json')
                with open(params_file_path, 'w') as f:
                    json.dump(model_params, f)

                # # Store model for later retrieval
                # models[(n_features, i, fold_idx)] = model

                # Evaluate model
                results = evaluate_model(model, X_test_fold, y_test_fold)
                results.update({'Num_Features': n_features, 'Subsample_Iteration': i+1, 'Fold_Index': fold_idx})
                fold_results.append(results)         

                # SHAP values
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test_fold).values

                # Convert SHAP values to DataFrame and store
                shap_df = pd.DataFrame(shap_values, columns=X_test_fold.columns)
                shap_results[(n_features, i, fold_idx)] = shap_df
                shap_df = pd.DataFrame(shap_values, columns=X_test_fold.columns)

                # Save SHAP values to file
                shap_file_path = os.path.join(shap_directory, f'shap_values_{n_features}_{i}_{fold_idx}.csv')
                shap_df.to_csv(shap_file_path, index=False)
                
                
                # Compute the mean absolute SHAP value for each feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_summary = pd.DataFrame({
                    'Feature': X_train_fold.columns,
                    'Mean Abs SHAP Value': mean_abs_shap
                })

                # Correlation
                feature_values = X_test_fold.reset_index(drop=True)  # Reset index to align with shap_values
                correlations = pd.Series(index=X_train_fold.columns, dtype=float)
                for feature in X_train_fold.columns:
                    correlations[feature] = shap_df[feature].corr(feature_values[feature])

                # Prepare the final summary DataFrame
                final_summary_df = shap_summary
                final_summary_df['Correlation with SHAP'] = correlations.values

                key = (n_features, i, fold_idx)
                SHAP_mean_and_corr[key] = final_summary_df.sort_values(by='Mean Abs SHAP Value', ascending=False)

            # Aggregate results for this downsampled dataset
            iteration_results = {
                'Num_Features': n_features,
                'Subsample_Iteration': i+1
            }
            for metric in ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']:
                iteration_results[f'{metric}_Mean'] = np.mean([r[metric] for r in fold_results])
                iteration_results[f'{metric}_Std'] = np.std([r[metric] for r in fold_results])
            
            # Convert the aggregated results to a DataFrame and save to CSV
            all_results.append(iteration_results)

    # Calculate the overall average for each metric
    overall_averages = {}
    metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']
    for metric in metrics:
        overall_averages[f'{metric}_Overall_Mean'] = np.mean([result[f'{metric}_Mean'] for result in all_results])
        overall_averages[f'{metric}_Overall_Std'] = np.mean([result[f'{metric}_Std'] for result in all_results])

    
    # Convert the aggregated results to a DataFrame and save to CSV
    summary_results_df = pd.DataFrame(all_results)
    summary_results_df.to_csv(os.path.join(results_directory, 'summary_evaluation_results.csv'), index=False)
    
    
    # Save the overall averages to a separate file
    overall_averages_df = pd.DataFrame([overall_averages])
    overall_averages_df.to_csv(os.path.join(results_directory, 'overall_averages.csv'), index=False)

    
    # Print the summary
    print("\nModel Performance Summary:")
    print(summary_results_df)
    print("\nOverall Model Performance Summary:")
    print(overall_averages_df)

    # To access a specific model's SHAP values, use the `shap_results` dictionary
    # Example: SHAP values for the model trained with 10 features, on 2nd subsample, 3rd fold
    # specific_shap_values = shap_results[(10, 2, 3)]

    # Aggregate SHAP values and correlations across all models
    aggregated_shap_values = {}
    aggregated_correlations = {}
    
    # Calculate and append mean and standard deviation for each metric across all folds
    for key, shap_df in shap_results.items():
        for feature in shap_df.columns:
            if feature not in aggregated_shap_values:
                aggregated_shap_values[feature] = []
                aggregated_correlations[feature] = []
            aggregated_shap_values[feature].append(np.abs(shap_df[feature]).mean())
            aggregated_correlations[feature].append(SHAP_mean_and_corr[key][SHAP_mean_and_corr[key]['Feature'] == feature]['Correlation with SHAP'].iloc[0])

    # Calculate the mean of the absolute SHAP values and the mean correlation for each feature
    mean_shap_values = {feature: np.mean(values) for feature, values in aggregated_shap_values.items()}
    mean_correlations = {feature: np.mean(values) for feature, values in aggregated_correlations.items()}

    # Sort features based on the mean absolute SHAP values
    sorted_features = sorted(mean_shap_values, key=mean_shap_values.get, reverse=True)

    # Extract the top X features and their directions
    top_features = sorted_features[:top_features]
    top_features_directions = {feature: 'Positive' if mean_correlations[feature] > 0 else 'Negative' for feature in top_features}

    # Display the top features and their directions
    print("Top Features and Their Directions:")
    for feature in top_features:
        print(f"{feature}: {top_features_directions[feature]}")
    

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    roc_curve, auc, balanced_accuracy_score, average_precision_score, matthews_corrcoef)
import shap
from sklearn.utils import resample
import json
import sys
import os
import argparse



# Function Definitions
def load_data(gene_expression_file, label_file):
    """Load gene expression and label data."""
    gene_expression = pd.read_csv(gene_expression_file, index_col=0, sep="\t")
    label_group_all = pd.read_csv(label_file, index_col=0)
    return gene_expression, label_group_all

def preprocess_data(gene_expression, label_group_all):
    """Preprocess data by filtering, encoding, and removing rows with missing values."""
    label_group_all['cluster_verbal_iq'] = label_group_all['verbal_iq'].apply(lambda x: 'high' if x >= 70 else 'low')
    ind_info = label_group_all[['sex', 'race', 'any_gastro_disorders_proband',
                                'any_neurological_disorders_proband',
                                'any_neurological_disorders_in_family',
                                'any_autoimmune_disease_proband',
                                'any_autoimmune_disease_in_family',
                                'any_chronic_illnesses_in_family',
                                'any_maternal_infections_in_preg',
                                'any_autoimmune_disease_mat',
                                'any_language_disorders_in_family',
                                'cluster_verbal_iq', 'nonverbal_iq', 'verbal_iq']]
    ind_info_cleaned = ind_info.dropna()
    row_names = ind_info_cleaned.index
    gene_expression_cleaned = gene_expression.loc[row_names].T
    return gene_expression_cleaned, ind_info_cleaned

def convert_target_to_binary(label_group_all, target_column='Target'):
    """Convert the target column to a binary format."""
    if target_column not in label_group_all.columns:
        raise KeyError(f"Column '{target_column}' not found in the dataframe.")
    Target = label_group_all.loc[:, target_column]
    Target = pd.DataFrame(Target.replace(to_replace=['no', 'yes'], value=[0, 1]))
    Target_np = np.array(Target)
    return Target, Target_np

def feature_engineering(gene_expression, ind_info):
    """Feature engineering and selection."""
    X_encoded = pd.get_dummies(ind_info)
    gene_expression = variance_threshold_selector(gene_expression.T, threshold=0.1)
    X_encoded = X_encoded.reindex(gene_expression.index)
    combined = gene_expression.merge(X_encoded, left_index=True, right_index=True)
    return combined

def variance_threshold_selector(data, threshold=0.2):
    """Remove features with low variance."""
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def train_test_split_and_scale(combined, target):
    """Split the data into training and testing sets and scale features."""
    X_train, X_test, y_train, y_test = train_test_split(combined, target, test_size=0.2, random_state=0)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test

def kmeans_clustering(X_train, X_test, n_clusters=30, n_init=10):
    """Apply KMeans clustering to the dataset."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    X_train['Cluster'] = kmeans.fit_predict(X_train)
    X_test['Cluster'] = kmeans.predict(X_test)
    return X_train, X_test

def feature_selection(X_train, y_train, n_features):
    """Select important features using Lasso."""
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    important_genes = np.abs(lasso.coef_).argsort()[-n_features:][::-1]
    return X_train.iloc[:, important_genes], important_genes


def train_xgboost(X_train, y_train, params, cv_method='grid', cv_folds=5, n_jobs=10, n_iter=50):
    """Train an XGBoost model using specified cross-validation method."""
    model = XGBClassifier(objective='binary:logistic', seed=1, colsample_bytree=0.5, use_label_encoder=False)
    if cv_method == 'grid':
        optimal_params = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='roc_auc',
            verbose=0,
            n_jobs=n_jobs,
            cv=cv_folds
        )
    elif cv_method == 'random':
        optimal_params = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            scoring='roc_auc',
            verbose=0,
            n_jobs=n_jobs,
            cv=cv_folds,
            n_iter=n_iter
        )
    else:
        raise ValueError("cv_method must be either 'grid' or 'random'")

    optimal_params.fit(X_train, y_train)
    return optimal_params

def evaluate_model(model, X_test, y_test, plot=False):
    """Evaluate the performance of the model."""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
    average_precision = average_precision_score(y_test, y_probs)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    return {
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': roc_auc,
    }

def multiple_downsampling(data, target_column, iterations=5):
    """Perform multiple downsampling iterations of the majority class."""
    downsampled_datasets = []
    majority_class = data[data[target_column] == 0]
    minority_class = data[data[target_column] == 1]
    for i in range(iterations):
        majority_downsampled = resample(majority_class,
                                        replace=False,
                                        n_samples=len(minority_class),
                                        random_state=i)
        downsampled_data = pd.concat([majority_downsampled, minority_class])
        downsampled_datasets.append(downsampled_data)
    return downsampled_datasets

def parse_feature_selection_options(option_str):
    # Convert a comma-separated string to a list of integers
    return [int(x.strip()) for x in option_str.split(',')]



# Main Script
if __name__ == "__main__":
    print(sys.argv)

    parser = argparse.ArgumentParser(description='Process model parameters.')

    # Define arguments
    parser.add_argument('-FS', '--feature_selection', default='10,20', help='Feature selection options (comma-separated)')
    parser.add_argument('-DS', '--downsampling', type=int, default=5, help='Number of downsampling iterations')
    parser.add_argument('-TF', '--top_features', type=int, default=10, help='Number of top features')
    parser.add_argument('-OD', '--output_directory', default='/gpfs0/alal/users/richtert/XGBoost_Fever_effect/output/', help='Directory for saving models')
    parser.add_argument('-OF', '--output_file', default='/gpfs0/alal/users/richtert/XGBoost_Fever_effect/summary/output.csv', help='File for saving evaluations')
    parser.add_argument('-GE', '--gene_expression_file', default="/gpfs0/alal/users/richtert/XGBoost_Fever_effect/files/symbol_gi_log.txt", help='Gene expression data file')
    parser.add_argument('-LF', '--label_file', default="/gpfs0/alal/users/richtert/XGBoost_Fever_effect/files/sampleTable_GI.csv", help='Label file')
    parser.add_argument('-CVM', '--cv_method', default='grid', choices=['grid', 'random'], help='Cross-validation method: grid or random')


    # Parse arguments
    args = parser.parse_args()

    # Use arguments
    feature_selection_options = parse_feature_selection_options(args.feature_selection)
    n_downsampling = args.downsampling
    top_features = args.top_features
    output_file = args.output_file
    gene_expression_file = args.gene_expression_file
    label_file = args.label_file
    cv_method = args.cv_method
    output_directory = args.output_directory

    # Directories for storing models, parameters, and results
    models_directory = os.path.join(args.output_directory, 'models')
    params_directory = os.path.join(args.output_directory, 'params')
    results_directory = os.path.join(args.output_directory, 'results')
    shap_directory = os.path.join(args.output_directory, 'shap_results')


    # Create directories if they don't exist
    for directory in [models_directory, params_directory, results_directory, shap_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)



    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    # Load data
    gene_expression, label_group_all = load_data(gene_expression_file, label_file)
    Target, Target_np = convert_target_to_binary(label_group_all)
    gene_expression, ind_info = preprocess_data(gene_expression, label_group_all)
    combined = feature_engineering(gene_expression, ind_info)
    Target = Target.reindex(combined.index)
    combined_with_target = combined.join(Target)

    # Define the hyperparameter grid for XGBoost
    params = {
        'learning_rate': [0.03, 0.04, 0.05],
        'max_depth': [8, 9, 10],
        'colsample_bylevel': [0.4, 0.5, 0.6],
        'lambda': [1, 2 ,3],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.2, 0.5],
        'subsample': [ 0.7, 0.8, 1],
        'reg_alpha': [0, 0.5, 1],
    }

    # Feature selection options and results storage
    all_results = []  # To store aggregated results
    models = {}
    shap_results = {}
    SHAP_mean_and_corr = {}

    # Multiple downsampling
    downsampled_datasets = multiple_downsampling(combined_with_target, 'Target', iterations=n_downsampling)

    for n_features in feature_selection_options:
        for i, dataset in enumerate(downsampled_datasets):
            X_downsampled = dataset.drop('Target', axis=1)
            y_downsampled = dataset['Target']
            X_train, X_test, y_train, y_test = train_test_split_and_scale(X_downsampled, y_downsampled)
            X_train, X_test = kmeans_clustering(X_train, X_test)

            # Feature selection
            X_train_fs, important_genes = feature_selection(X_train, y_train, n_features)
            X_test_fs = X_test.iloc[:, important_genes]

            # Cross-validation setup
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_train_fs, y_train)):
                X_train_fold, X_test_fold = X_train_fs.iloc[train_idx], X_train_fs.iloc[test_idx]
                y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

                # Train the model
                grid_search = train_xgboost(X_train_fold, y_train_fold, params, cv_method=cv_method)
                model = grid_search.best_estimator_  # Extract the best estimator


                # Save the model in the specified directory
                model_file_path = os.path.join(models_directory, f'model_{n_features}_{i}_{fold_idx}.json')
                model.save_model(model_file_path)  # Save the model


                # Get the model parameters
                model_params = model.get_xgb_params()
                model_params.update({'n_features': n_features, 'iteration': i, 'fold_idx': fold_idx})


                # Save the model parameters in the specified directory
                params_file_path = os.path.join(params_directory, f'model_params_{n_features}_{i}_{fold_idx}.json')
                with open(params_file_path, 'w') as f:
                    json.dump(model_params, f)

                # # Store model for later retrieval
                # models[(n_features, i, fold_idx)] = model

                # Evaluate model
                results = evaluate_model(model, X_test_fold, y_test_fold)
                results.update({'Num_Features': n_features, 'Subsample_Iteration': i+1, 'Fold_Index': fold_idx})
                fold_results.append(results)

                # SHAP values
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test_fold).values

                # Convert SHAP values to DataFrame and store
                shap_df = pd.DataFrame(shap_values, columns=X_test_fold.columns)
                shap_results[(n_features, i, fold_idx)] = shap_df
                shap_df = pd.DataFrame(shap_values, columns=X_test_fold.columns)

                # Save SHAP values to file
                shap_file_path = os.path.join(shap_directory, f'shap_values_{n_features}_{i}_{fold_idx}.csv')
                shap_df.to_csv(shap_file_path, index=False)


                # Compute the mean absolute SHAP value for each feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_summary = pd.DataFrame({
                    'Feature': X_train_fold.columns,
                    'Mean Abs SHAP Value': mean_abs_shap
                })

                # Correlation
                feature_values = X_test_fold.reset_index(drop=True)  # Reset index to align with shap_values
                correlations = pd.Series(index=X_train_fold.columns, dtype=float)
                for feature in X_train_fold.columns:
                    correlations[feature] = shap_df[feature].corr(feature_values[feature])

                # Prepare the final summary DataFrame
                final_summary_df = shap_summary
                final_summary_df['Correlation with SHAP'] = correlations.values

                key = (n_features, i, fold_idx)
                SHAP_mean_and_corr[key] = final_summary_df.sort_values(by='Mean Abs SHAP Value', ascending=False)

            # Aggregate results for this downsampled dataset
            iteration_results = {
                'Num_Features': n_features,
                'Subsample_Iteration': i+1
            }
            for metric in ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']:
                iteration_results[f'{metric}_Mean'] = np.mean([r[metric] for r in fold_results])
                iteration_results[f'{metric}_Std'] = np.std([r[metric] for r in fold_results])

            # Convert the aggregated results to a DataFrame and save to CSV
            all_results.append(iteration_results)

    # Calculate the overall average for each metric
    overall_averages = {}
    metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']
    for metric in metrics:
        overall_averages[f'{metric}_Overall_Mean'] = np.mean([result[f'{metric}_Mean'] for result in all_results])
        overall_averages[f'{metric}_Overall_Std'] = np.mean([result[f'{metric}_Std'] for result in all_results])


    # Convert the aggregated results to a DataFrame and save to CSV
    summary_results_df = pd.DataFrame(all_results)
    summary_results_df.to_csv(os.path.join(results_directory, 'summary_evaluation_results.csv'), index=False)


    # Save the overall averages to a separate file
    overall_averages_df = pd.DataFrame([overall_averages])
    overall_averages_df.to_csv(os.path.join(results_directory, 'overall_averages.csv'), index=False)


    # Print the summary
    print("\nModel Performance Summary:")
    print(summary_results_df)
    print("\nOverall Model Performance Summary:")
    print(overall_averages_df)

    # To access a specific model's SHAP values, use the `shap_results` dictionary
    # Example: SHAP values for the model trained with 10 features, on 2nd subsample, 3rd fold
    # specific_shap_values = shap_results[(10, 2, 3)]

    # Aggregate SHAP values and correlations across all models
    aggregated_shap_values = {}
    aggregated_correlations = {}

    # Calculate and append mean and standard deviation for each metric across all folds
    for key, shap_df in shap_results.items():
        for feature in shap_df.columns:
            if feature not in aggregated_shap_values:
                aggregated_shap_values[feature] = []
                aggregated_correlations[feature] = []
            aggregated_shap_values[feature].append(np.abs(shap_df[feature]).mean())
            aggregated_correlations[feature].append(SHAP_mean_and_corr[key][SHAP_mean_and_corr[key]['Feature'] == feature]['Correlation with SHAP'].iloc[0])

    # Calculate the mean of the absolute SHAP values and the mean correlation for each feature
    mean_shap_values = {feature: np.mean(values) for feature, values in aggregated_shap_values.items()}
    mean_correlations = {feature: np.mean(values) for feature, values in aggregated_correlations.items()}

    # Sort features based on the mean absolute SHAP values
    sorted_features = sorted(mean_shap_values, key=mean_shap_values.get, reverse=True)

    # Extract the top X features and their directions
    top_features = sorted_features[:top_features]
    top_features_directions = {feature: 'Positive' if mean_correlations[feature] > 0 else 'Negative' for feature in top_features}

    # Display the top features and their directions
    print("Top Features and Their Directions:")
    for feature in top_features:
        print(f"{feature}: {top_features_directions[feature]}")

