import os
import re
import sys
import unittest

import numpy as np


try:
    from deep_ner.dataset_splitting import split_dataset
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from deep_ner.dataset_splitting import split_dataset


class TestDatasetSplitting(unittest.TestCase):
    def test_positive01(self):
        X = np.array(
            ['01abc', '02def', '03ghi', '04jkl', '05mno', '06pqr', '07stu', '08vwx', '09yza', '10bcd', '11efg',
             '12hij'],
            dtype=np.str
        )
        y_tokenized = np.array(
            [
                [0, 0, 2, 1, 1, 0, 0, 0, 2, 0, 4, 3, 0],  # 0 1 2 3 4 # 0 2 3 4
                [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 2 6     # 0
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0         # 0
                [0, 0, 0, 4, 0, 0, 4, 3, 0, 0, 0, 0, 0],  # 0 3 4     # 0 3 4
                [4, 3, 0, 4, 3, 3, 0, 0, 0, 0, 0, 2, 1],  # 0 1 2 3 4 # 0 2 3 4
                [0, 0, 0, 2, 1, 0, 0, 0, 4, 0, 0, 0, 0],  # 0 1 2 4   # 0 2 4
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0         # 0
                [0, 0, 0, 6, 5, 5, 0, 0, 0, 0, 0, 0, 0],  # 0 5 6     # 0
                [0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 0, 0, 0],  # 0 3 4 5 6 # 0 3 4
                [0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 2, 0, 0],  # 0 2 4     # 0 2 4
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0],  # 0 3 4     # 0 3 4
                [0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 2, 1, 1],  # 0 1 2 4   # 0 2 4
            ],
            dtype=np.int32
        )
        # 0: 120
        # 1: 6
        # 2: 7
        # 3: 7
        # 4: 11
        # 5: 3
        # 6: 3
        true_indices_for_training = np.array([1, 4, 5, 6, 7, 8, 10, 11], dtype=np.int32)
        true_indices_for_testing = np.array([0, 2, 3, 9], dtype=np.int32)
        calc_indices_for_training, calc_indices_for_testing = split_dataset(X, y_tokenized, 0.3333,
                                                                            n_restarts=4, random_seed=0)
        self.assertIsInstance(calc_indices_for_training, np.ndarray)
        self.assertIsInstance(calc_indices_for_testing, np.ndarray)
        self.assertEqual(true_indices_for_training.tolist(), calc_indices_for_training.tolist())
        self.assertEqual(true_indices_for_testing.tolist(), calc_indices_for_testing.tolist())

    def test_negative01(self):
        X = np.array(
            ['01abc',],
            dtype=np.str
        )
        y_tokenized = np.array(
            [
                [0, 0, 2, 1, 1, 0, 0, 0, 2, 0, 4, 3, 0],
            ],
            dtype=np.int32
        )
        true_err_msg = re.escape('There are too few samples in the data set! Minimal number of samples is 2.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(X, y_tokenized, 0.3333, n_restarts=4, random_seed=0)

    def test_negative02(self):
        X = np.array(
            ['01abc', '02def', '03ghi', '04jkl', '05mno', '06pqr', '07stu', '08vwx', '09yza', '10bcd', '11efg',
             '12hij'],
            dtype=np.str
        )
        y_tokenized = np.array(
            [
                [0, 0, 2, 1, 1, 0, 0, 0, 2, 0, 4, 3, 0],
                [2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 4, 3, 0, 0, 0, 0, 0],
                [4, 3, 0, 4, 3, 3, 0, 0, 0, 0, 0, 2, 1],
                [0, 0, 0, 2, 1, 0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 6, 5, 5, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 0, 0, 0],
                [0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 2, 1, 1],
            ],
            dtype=np.int32
        )
        true_err_msg = re.escape('{0} is too small value of the test part! '
                                 'There are no samples for testing subset!'.format(0.01))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(X, y_tokenized, 0.01, n_restarts=4, random_seed=0)

    def test_negative03(self):
        X = np.array(
            ['01abc', '02def', '03ghi', '04jkl', '05mno', '06pqr', '07stu', '08vwx', '09yza', '10bcd', '11efg',
             '12hij'],
            dtype=np.str
        )
        y_tokenized = np.array(
            [
                [0, 0, 2, 1, 1, 0, 0, 0, 2, 0, 4, 3, 0],
                [2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 4, 3, 0, 0, 0, 0, 0],
                [4, 3, 0, 4, 3, 3, 0, 0, 0, 0, 0, 2, 1],
                [0, 0, 0, 2, 1, 0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 6, 5, 5, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 0, 0, 0],
                [0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 2, 1, 1],
            ],
            dtype=np.int32
        )
        true_err_msg = re.escape('{0} is too large value of the test part! '
                                 'There are no samples for training subset!'.format(0.99))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(X, y_tokenized, 0.99, n_restarts=4, random_seed=0)

    def test_negative04(self):
        X = np.array(
            ['01abc', '02def', '03ghi', '04jkl', '05mno', '06pqr', '07stu', '08vwx', '09yza', '10bcd', '11efg',
             '12hij'],
            dtype=np.str
        )
        y_tokenized = np.array(
            [
                [0, 0, 2, 1, 1, 0, 0, 0, 2, 0, 4, 3, 0],
                [2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 4, 3, 0, 0, 0, 0, 0],
                [4, 3, 0, 4, 3, 3, 0, 0, 0, 0, 0, 2, 1],
                [0, 0, 0, 2, 1, 0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 6, 5, 5, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 0, 0, 0],
                [0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 2, 1, 1],
            ],
            dtype=np.int32
        )
        true_err_msg = re.escape('1 is too small value of restarts number. It must be greater than 1.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(X, y_tokenized, 0.3333, n_restarts=1, random_seed=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
