from logging import Logger
from typing import Tuple, Union
import warnings

import numpy as np


def split_dataset(y: Union[list, tuple, np.array], test_part: float, n_restarts: int=10,
                  logger: Union[Logger, None]=None) -> Tuple[np.ndarray, np.ndarray]:
    if n_restarts < 2:
        raise ValueError('{0} is too small value of restarts number. It must be greater than 1.'.format(n_restarts))
    n_samples = len(y)
    if n_samples < 2:
        raise ValueError('There are too few samples in the data set! Minimal number of samples is 2.')
    n_test = int(round(test_part * n_samples))
    n_train = n_samples - n_test
    if n_test < 1:
        raise ValueError('{0} is too small value of the test part! There are no samples for '
                         'testing subset!'.format(test_part))
    if n_train < 1:
        raise ValueError('{0} is too large value of the test part! There are no samples for '
                         'training subset!'.format(test_part))
    indices = np.arange(0, n_samples, 1, dtype=np.int32)
    np.random.shuffle(indices)
    set_of_classes_for_training = set()
    set_of_classes_for_testing = set()
    for idx in indices[0:n_train]:
        set_of_classes_for_training |= set(y[idx].keys())
    for idx in indices[n_train:]:
        set_of_classes_for_testing |= set(y[idx].keys())
    if set_of_classes_for_training == set_of_classes_for_testing:
        train_index = indices[0:n_train]
        test_index = indices[n_train:]
    else:
        if set_of_classes_for_testing < set_of_classes_for_training:
            best_indices = np.copy(indices)
        else:
            best_indices = None
        for restart in range(1, n_restarts):
            np.random.shuffle(indices)
            set_of_classes_for_training = set()
            set_of_classes_for_testing = set()
            for idx in indices[0:n_train]:
                set_of_classes_for_training |= set(y[idx].keys())
            for idx in indices[n_train:]:
                set_of_classes_for_testing |= set(y[idx].keys())
            if set_of_classes_for_training == set_of_classes_for_testing:
                best_indices = np.copy(indices)
                break
            if set_of_classes_for_testing < set_of_classes_for_training:
                best_indices = np.copy(indices)
        if best_indices is None:
            if logger is None:
                warnings.warn('Data set cannot be splitted by stratified folds.')
            else:
                logger.warning('Data set cannot be splitted by stratified folds.')
            train_index = indices[0:n_train]
            test_index = indices[n_train:]
        else:
            set_of_classes_for_training = set()
            set_of_classes_for_testing = set()
            for idx in best_indices[0:n_train]:
                set_of_classes_for_training |= set(y[idx].keys())
            for idx in best_indices[n_train:]:
                set_of_classes_for_testing |= set(y[idx].keys())
            if set_of_classes_for_training != set_of_classes_for_testing:
                if logger is None:
                    warnings.warn('Data set cannot be splitted by stratified folds.')
                else:
                    logger.warning('Data set cannot be splitted by stratified folds.')
            train_index = best_indices[0:n_train]
            test_index = best_indices[n_train:]
    return np.sort(train_index), np.sort(test_index)
