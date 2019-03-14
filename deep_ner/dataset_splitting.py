from typing import Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset(X: Union[list, tuple, np.array], y_tokenized: np.ndarray, test_part: float, n_restarts: int=3,
                  random_seed: Union[int, None]=None) -> Tuple[np.ndarray, np.ndarray]:
    if n_restarts < 2:
        raise ValueError('{0} is too small value of restarts number. It must be greater than 1.'.format(n_restarts))
    n_samples = y_tokenized.shape[0]
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
    y_full = []
    frequencies_of_labels = dict()
    for sample_idx in range(n_samples):
        y_full.append('_'.join(sorted(list(set(['{0}'.format(val) for val in y_tokenized[sample_idx]])))))
        for token_idx in range(y_tokenized.shape[1]):
            class_idx = y_tokenized[sample_idx][token_idx]
            frequencies_of_labels[class_idx] = frequencies_of_labels.get(class_idx, 0) + 1
    set_of_classes = set(frequencies_of_labels.keys())
    frequencies_of_labels = sorted(
        [(class_idx, frequencies_of_labels[class_idx]) for class_idx in frequencies_of_labels],
        key=lambda it: (it[1], it[0])
    )
    y_full = np.array(y_full, dtype=np.str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=random_seed)
    try:
        train_index, test_index = next(sss.split(X, y_full))
    except:
        train_index = None
        test_index = None
    if (train_index is None) or (test_index is None):
        y_reduced = []
        for _ in range(n_restarts):
            if len(frequencies_of_labels) == 1:
                break
            frequencies_of_labels = frequencies_of_labels[1:]
            del set_of_classes
            set_of_classes = set([it[0] for it in frequencies_of_labels])
            del y_reduced
            y_reduced = []
            for sample_idx in range(n_samples):
                y_reduced.append(
                    '_'.join(
                        sorted(list(set(
                            [
                                '{0}'.format(val) for val in
                                filter(lambda class_idx: class_idx in set_of_classes, y_tokenized[sample_idx])
                            ]
                        )))
                    )
                )
            y_reduced = np.array(y_reduced, dtype=np.str)
            try:
                train_index, test_index = next(sss.split(X, y_reduced))
            except:
                train_index = None
                test_index = None
            if (train_index is not None) and (test_index is not None):
                break
        if (train_index is None) or (test_index is None):
            indices = np.arange(0, n_samples, 1, dtype=np.int32)
            np.random.shuffle(indices)
            train_index = indices[0:n_train]
            test_index = indices[n_train:]
            del indices
        else:
            train_index = set(train_index.tolist())
            test_index = set(test_index.tolist())
            for reduced_class_label in sorted(list(set(y_reduced))):
                indices_and_labels = []
                n = 0
                for sample_idx in sorted(list(train_index)):
                    if y_reduced[sample_idx] == reduced_class_label:
                        indices_and_labels.append((sample_idx, y_full[sample_idx]))
                        train_index.remove(sample_idx)
                        n += 1
                for sample_idx in sorted(list(test_index)):
                    if y_reduced[sample_idx] == reduced_class_label:
                        indices_and_labels.append((sample_idx, y_full[sample_idx]))
                        test_index.remove(sample_idx)
                indices_and_labels.sort(key=lambda it: (len(it[1].split('_')), len(it[1]), it[1], it[0]), reverse=True)
                for idx in range(len(indices_and_labels)):
                    if idx < n:
                        train_index.add(indices_and_labels[idx][0])
                    else:
                        test_index.add(indices_and_labels[idx][0])
                del indices_and_labels
            train_index = np.array(list(train_index), dtype=np.int32)
            test_index = np.array(list(test_index), dtype=np.int32)
    del sss, y_full, frequencies_of_labels
    return np.sort(train_index), np.sort(test_index)
