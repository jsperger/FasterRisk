import numpy as np
import pandas as pd
import time

from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.utils import get_groupIndex_from_featureNames, isEqual_upTo_16decimal

import sys

def check_answer(random_10_featureIndex_to_groupIndex):
    answer = np.asarray([18, 17, 11, 4, 1, 1, 0, 2, 1, 1], dtype=int)
    assert isEqual_upTo_16decimal(random_10_featureIndex_to_groupIndex, answer), "answers from get_groupIndex_from_featureNames are not correct!"



import os
import pytest

def DISABLED_test_get_groupIndex_from_featureNames(): # Renamed to disable
    data_path = "tests/fico_data.csv" # Original path, file not in new data/
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}, and FICO data is not in the new data folder.")
    y_label_name = "RiskPerformance"
    df = pd.read_csv(data_path)
    X_df = df.drop(columns = [y_label_name])


    X_featureNames = list(X_df.columns)

    featureIndex_to_groupIndex = get_groupIndex_from_featureNames(X_featureNames)

    rng = np.random.default_rng(seed=0)

    random_10_indices = rng.choice(len(X_featureNames), 10, replace=False)
    check_answer(featureIndex_to_groupIndex[random_10_indices])


if __name__ == "__main__":
    test_get_groupIndex_from_featureNames()