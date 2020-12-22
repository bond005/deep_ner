import os
import re
import sys
import unittest

import numpy as np


try:
    from deep_ner.dataset_splitting import split_dataset, sample_from_dataset
    from deep_ner.utils import load_dataset_from_json, set_total_seed
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from deep_ner.utils import load_dataset_from_json, set_total_seed
    from deep_ner.dataset_splitting import split_dataset, sample_from_dataset


class TestDatasetSplitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_total_seed(0)

    def test_split_dataset_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        _, y = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        train_index, test_index = split_dataset(y, 0.3, 10)
        self.assertIsInstance(train_index, np.ndarray)
        self.assertIsInstance(test_index, np.ndarray)
        self.assertEqual(len(y), len(train_index) + len(test_index))
        self.assertEqual(len(train_index), len(set(train_index.tolist())))
        self.assertEqual(len(test_index), len(set(test_index.tolist())))
        self.assertEqual(0, len(set(train_index.tolist()) & set(test_index.tolist())))
        true_set_of_classes = {'ORG', 'PERSON', 'LOCATION'}
        set_of_classes_for_training = set()
        for idx in train_index:
            set_of_classes_for_training |= set(y[idx].keys())
        set_of_classes_for_testing = set()
        for idx in test_index:
            set_of_classes_for_testing |= set(y[idx].keys())
        self.assertEqual(set_of_classes_for_training, set_of_classes_for_testing)
        self.assertEqual(true_set_of_classes, set_of_classes_for_training)
        self.assertEqual(true_set_of_classes, set_of_classes_for_testing)

    def test_split_dataset_negative01(self):
        y = [
            {"LOCATION": [(55, 63), (66, 84), (87, 93)], "PERSON": [(281, 289)]}
        ]
        true_err_msg = re.escape('There are too few samples in the data set! Minimal number of samples is 2.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(y, 0.3333, n_restarts=4)

    def test_split_dataset_negative02(self):
        y = [
            {"LOCATION": [(55, 63), (66, 84), (87, 93)], "PERSON": [(281, 289)]},
            {"PERSON": [(33, 44)], "LOCATION": [(198, 204), (189, 197), (168, 185)], "ORG": [(230, 249)]},
            {"PERSON": [(87, 98)], "ORG": [(18, 42), (18, 56)]},
            {"LOCATION": [(151, 157)], "PERSON": [(130, 140)]}
        ]
        true_err_msg = re.escape('{0} is too small value of the test part! '
                                 'There are no samples for testing subset!'.format(0.01))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(y, 0.01, n_restarts=10)

    def test_split_dataset_negative03(self):
        y = [
            {"LOCATION": [(55, 63), (66, 84), (87, 93)], "PERSON": [(281, 289)]},
            {"PERSON": [(33, 44)], "LOCATION": [(198, 204), (189, 197), (168, 185)], "ORG": [(230, 249)]},
            {"PERSON": [(87, 98)], "ORG": [(18, 42), (18, 56)]},
            {"LOCATION": [(151, 157)], "PERSON": [(130, 140)]}
        ]
        true_err_msg = re.escape('{0} is too large value of the test part! '
                                 'There are no samples for training subset!'.format(0.99))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(y, 0.99, n_restarts=4)

    def test_split_dataset_negative04(self):
        y = [
            {"LOCATION": [(55, 63), (66, 84), (87, 93)], "PERSON": [(281, 289)]},
            {"PERSON": [(33, 44)], "LOCATION": [(198, 204), (189, 197), (168, 185)], "ORG": [(230, 249)]},
            {"PERSON": [(87, 98)], "ORG": [(18, 42), (18, 56)]},
            {"LOCATION": [(151, 157)], "PERSON": [(130, 140)]}
        ]
        true_err_msg = re.escape('1 is too small value of restarts number. It must be greater than 1.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = split_dataset(y, 0.3333, n_restarts=1)

    def test_sample_from_dataset_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        _, y = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        subset_index = sample_from_dataset(y=y, n=3, n_restarts=10)
        self.assertIsInstance(subset_index, np.ndarray)
        self.assertEqual(3, len(subset_index))
        self.assertGreater(len(y), len(subset_index))
        self.assertEqual(len(subset_index), len(set(subset_index.tolist())))
        true_set_of_classes = {'ORG', 'PERSON', 'LOCATION'}
        subset_of_classes = set()
        for idx in subset_index:
            subset_of_classes |= set(y[idx].keys())
        self.assertEqual(true_set_of_classes, subset_of_classes)

    def test_sample_from_dataset_positive02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        _, y = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        subset_index = sample_from_dataset(y=y, n=(len(y) + 1) * 2, n_restarts=10)
        self.assertIsInstance(subset_index, np.ndarray)
        self.assertEqual(len(y), len(subset_index))
        self.assertEqual(set(range(len(y))), set(subset_index.tolist()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
