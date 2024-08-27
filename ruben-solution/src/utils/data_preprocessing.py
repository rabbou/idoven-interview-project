from os.path import join
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from models.tscnn import SlidingWindowGenerator


def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    """
    Load raw ECG data from the PTB-XL dataset.

    Parameters:
    - df (pd.DataFrame): DataFrame containing metadata of ECG records, including file paths.
    - sampling_rate (int): The desired sampling rate for the ECG data. PTB-XL provides data at two sampling rates: 100 Hz and 500 Hz.
    - path (str): The base path to the directory containing the ECG files.

    Returns:
    - np.ndarray: A 3D array where each entry corresponds to the ECG signal of a single record.
      The dimensions of the array are (number of records, number of samples, number of leads).
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dic: dict, agg_df: pd.DataFrame, col: str, threshold: float = 80.0) -> str:
    """
    Aggregate diagnostic codes based on their likelihoods, including only those with a likelihood of 80% or higher.

    Parameters:
    - y_dic (dict): Dictionary containing diagnostic codes as keys and their respective probabilities or confidences as values.
    - agg_df (pd.DataFrame): DataFrame that maps diagnostic codes to their corresponding superclasses, subclasses, or other aggregate categories.
    - col (str): The column name in agg_df that specifies the category (e.g., superclass, subclass) to which the diagnostic codes should be aggregated.
    - threshold (float): The minimum likelihood required for a diagnostic code to be included in the aggregation.

    Returns:
    - str: A string containing the aggregated diagnostic categories separated by the "|" character.
    """
    tmp = []
    for scp_code, likelihood in y_dic.items():
        if scp_code in agg_df.index and likelihood >= threshold:
            tmp.append(agg_df.loc[scp_code][col])
    return "|".join(set(tmp))


def create_generators_or_datasets(
    y: pd.DataFrame,
    window_size: Optional[int],
    step_size: Optional[int],
    batch_size: int,
    sample_size: float,
    sampling_rate: int,
    data_path: str,
    return_generators: bool,
    valid_fold: int = 9,
    test_fold: int = 10,
    random_seed: int = 42,
) -> Union[
    Tuple[SlidingWindowGenerator, SlidingWindowGenerator, SlidingWindowGenerator],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Sample data, load raw data, and either create generators or return datasets for training, validation, and testing.

    Parameters:
    - y (pd.DataFrame): The DataFrame containing labels and fold information.
    - window_size (int, optional): The size of the window for the sliding window generator. Required if return_generators is True.
    - step_size (int, optional): The step size for the sliding window generator. Required if return_generators is True.
    - batch_size (int): The batch size for the sliding window generator.
    - sample_size (float): The fraction of the data to sample.
    - sampling_rate (int): The sampling rate for loading the raw data.
    - data_path (str): The path to the raw data directory.
    - return_generators (bool): If True, return sliding window generators; if False, return raw datasets.
    - valid_fold (int): The fold number to use for validation. Defaults to 9.
    - test_fold (int): The fold number to use for testing. Defaults to 10.
    - random_seed (Optional[int]): Seed for the random number generator. Defaults to 42.

    Returns:
    - Union[Tuple[SlidingWindowGenerator, SlidingWindowGenerator, SlidingWindowGenerator],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
      Depending on return_generators, either a tuple of generators (train_generator, val_generator, test_generator) or a tuple of datasets (X_train, y_train, X_val, y_val, X_test, y_test).
    """

    np.random.seed(random_seed)

    sample_indices = np.random.choice(y.shape[0], size=int(sample_size * y.shape[0]), replace=False)
    y_sub = y.iloc[sample_indices]
    X_sub = load_raw_data(y_sub, sampling_rate, join(data_path, "raw"))

    X_train = X_sub[np.where(~y_sub.strat_fold.isin([valid_fold, test_fold]))]
    y_train = y_sub.loc[(~y_sub.strat_fold.isin([valid_fold, test_fold]))].LABEL.to_numpy()

    X_val = X_sub[np.where(y_sub.strat_fold == valid_fold)]
    y_val = y_sub.loc[y_sub.strat_fold == valid_fold].LABEL.to_numpy()

    X_test = X_sub[np.where(y_sub.strat_fold == test_fold)]
    y_test = y_sub.loc[y_sub.strat_fold == test_fold].LABEL.to_numpy()

    if return_generators:
        train_generator = SlidingWindowGenerator(X_train, y_train, window_size, step_size, batch_size)
        val_generator = SlidingWindowGenerator(X_val, y_val, window_size, step_size, batch_size)
        test_generator = SlidingWindowGenerator(X_test, y_test, window_size, step_size, batch_size)
        return train_generator, val_generator, test_generator
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
