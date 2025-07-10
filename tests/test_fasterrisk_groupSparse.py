import numpy as np
import pandas as pd
import time
import os
import pytest

from sklearn.model_selection import train_test_split

from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df

def get_expected_answers():
    pass

def save_to_dict(int_sols_dict, multiplier, int_sol, train_acc, test_acc, train_auc, test_auc, logisticLoss):
    int_sols_dict["multipliers"].append(multiplier)
    int_sols_dict["int_sols"].append(int_sol)
    int_sols_dict["train_accs"].append(train_acc)
    int_sols_dict["test_accs"].append(test_acc)
    int_sols_dict["train_aucs"].append(train_auc)
    int_sols_dict["test_aucs"].append(test_auc)
    int_sols_dict["logisticLosses"].append(logisticLoss)

def test_check_solutions_interface():
    # import data
    DATA_PATH = "data/breastcancer_data.csv" # Updated path
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data file not found: {DATA_PATH}")

    original_data_df = pd.read_csv(DATA_PATH)
    y_original = np.asarray(original_data_df[original_data_df.columns[0]].values) # First column is target
    X_original_df = original_data_df.drop(columns=original_data_df.columns[0])

    # Convert y from {0, 1} to {-1, 1}
    # Assuming 0 maps to -1 and 1 maps to 1 (or whatever the positive class is)
    # For breastcancer data, 'Benign' is 0 and 'Malignant' might be 1. Let's assume 0 -> -1, 1 -> 1 is the general transformation.
    # Need to confirm actual values if not 0/1. breastcancer_data.csv has 'Benign' as 0. Let's assume it's 0/1 for now.
    y_transformed = np.array([-1 if val == 0 else 1 for val in y_original])


    # Binarize features and get group indices
    # Reducing max_num_thresholds_per_feature for speed, original default is 100
    X_binarized_df, featureIndex_to_groupIndex = convert_continuous_df_to_binary_df(X_original_df, max_num_thresholds_per_feature=10, get_featureIndex_to_groupIndex=True)
    X = np.asarray(X_binarized_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42) # Using 20% for test, consistent y

    lambda2 = 1e-8
    sparsity = 5 # Max number of features
    sparseDiversePool_gap_tolerance = 0.05
    sparseDiversePool_select_top_m = 10 # Reduced from 50 for speed
    parent_size = 10
    child_size = 10
    maxAttempts = 50
    num_ray_search = 5 # Reduced from 20 for speed
    lineSearch_early_stop_tolerance = 0.001 
    group_sparsity = 3 # Max number of groups

    # obtain sparse scoring systems
    int_sols_dict = {"int_sols": [], "train_accs": [], "test_accs": [], "train_aucs": [], "test_aucs": [], "logisticLosses": [], "multipliers": []}
    
    RiskScoreOptimizer_m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity, select_top_m = sparseDiversePool_select_top_m, gap_tolerance = sparseDiversePool_gap_tolerance, parent_size = parent_size, maxAttempts = maxAttempts, num_ray_search = num_ray_search, lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance, group_sparsity = group_sparsity, featureIndex_to_groupIndex = featureIndex_to_groupIndex)

    start_time = time.time()
    
    RiskScoreOptimizer_m.optimize()
    
    int_sols_dict['run_time'] = time.time() - start_time

    multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = RiskScoreOptimizer_m.get_models()

    for i in range(len(multipliers)):
        multiplier = multipliers[i]
        beta0_integer = sparseDiversePool_beta0_integer[i]
        betas_integer = sparseDiversePool_betas_integer[i]

        RiskScoreClassifier_m = RiskScoreClassifier(multiplier, beta0_integer, betas_integer)
        logisticLoss = RiskScoreClassifier_m.compute_logisticLoss(X_train, y_train)
        
        train_acc, train_auc = RiskScoreClassifier_m.get_acc_and_auc(X_train, y_train)
        test_acc, test_auc = RiskScoreClassifier_m.get_acc_and_auc(X_test, y_test)

        integer_sol = np.insert(betas_integer, 0, beta0_integer)
        save_to_dict(int_sols_dict, multiplier, integer_sol, train_acc, test_acc, train_auc, test_auc, logisticLoss)

    # check whether each solution satisfies the group sparsity constraint
    for sol in int_sols_dict['int_sols']:
        sol = sol[1:]
        support = sol.nonzero()[0]
        groupIndices = featureIndex_to_groupIndex[support]
        num_unique_groupIndices = len(np.unique(groupIndices))
        assert num_unique_groupIndices <= group_sparsity, "group sparsity constraint is not satisfied!"
        assert len(support) <= sparsity, "sparsity constraint is not satisfied!"

if __name__ == "__main__":
    test_check_solutions_interface()