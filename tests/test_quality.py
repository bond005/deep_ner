import os
import sys
import unittest


try:
    from deep_ner.utils import load_dataset
    from deep_ner.quality import calculate_prediction_quality, calc_similarity_between_entities
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from deep_ner.utils import load_dataset
    from deep_ner.quality import calculate_prediction_quality, calc_similarity_between_entities


class TestQuality(unittest.TestCase):
    def test_calc_similarity_between_entities_positive01(self):
        gold_entity = (3, 9)
        predicted_entity = (3, 9)
        true_similarity = 1.0
        true_tp = 6
        true_fp = 0
        true_fn = 0
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calc_similarity_between_entities_positive02(self):
        gold_entity = (4, 8)
        predicted_entity = (3, 9)
        true_similarity = 0.666666667
        true_tp = 4
        true_fp = 2
        true_fn = 0
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calc_similarity_between_entities_positive03(self):
        gold_entity = (3, 9)
        predicted_entity = (4, 8)
        true_similarity = 0.666666667
        true_tp = 4
        true_fp = 0
        true_fn = 2
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calc_similarity_between_entities_positive04(self):
        gold_entity = (3, 9)
        predicted_entity = (2, 8)
        true_similarity = 0.714285714
        true_tp = 5
        true_fp = 1
        true_fn = 1
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calc_similarity_between_entities_positive05(self):
        gold_entity = (2, 8)
        predicted_entity = (3, 9)
        true_similarity = 0.714285714
        true_tp = 5
        true_fp = 1
        true_fn = 1
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calc_similarity_between_entities_positive06(self):
        gold_entity = (3, 9)
        predicted_entity = (10, 16)
        true_similarity = 0.0
        true_tp = 0
        true_fp = 6
        true_fn = 6
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calc_similarity_between_entities_positive07(self):
        gold_entity = (3, 9)
        predicted_entity = (0, 2)
        true_similarity = 0.0
        true_tp = 0
        true_fp = 2
        true_fn = 6
        similarity, tp, fp, fn = calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_calculate_prediction_quality(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        X_true, y_true = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        X_pred, y_pred = load_dataset(os.path.join(base_dir, 'predicted_named_entities.json'))
        self.assertEqual(X_true, X_pred)
        f1, precision, recall = calculate_prediction_quality(y_true, y_pred, ('LOCATION', 'PERSON', 'ORG'))
        self.assertIsInstance(f1, float)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertAlmostEqual(f1, 0.842037, places=3)
        self.assertAlmostEqual(precision, 0.908352, places=3)
        self.assertAlmostEqual(recall, 0.784746, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
