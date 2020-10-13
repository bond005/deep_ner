import copy
import gc
import os
import pickle
import re
import sys
import tempfile
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from spacy_udpipe.language import UDPipeLanguage

try:
    from deep_ner.bert_ner import BERT_NER
    from deep_ner.utils import load_dataset_from_json, set_total_seed
    from deep_ner.quality import calculate_prediction_quality
    from deep_ner.udpipe_data import UNIVERSAL_DEPENDENCIES, UNIVERSAL_POS_TAGS
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from deep_ner.bert_ner import BERT_NER
    from deep_ner.utils import load_dataset_from_json, set_total_seed
    from deep_ner.quality import calculate_prediction_quality
    from deep_ner.udpipe_data import UNIVERSAL_DEPENDENCIES, UNIVERSAL_POS_TAGS


class TestBertNer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_total_seed(0)

    def tearDown(self):
        if hasattr(self, 'ner'):
            del self.ner
        if hasattr(self, 'another_ner'):
            del self.another_ner
        if hasattr(self, 'temp_file_name'):
            if os.path.isfile(self.temp_file_name):
                os.remove(self.temp_file_name)

    def test_creation(self):
        self.ner = BERT_NER(udpipe_lang='en')
        self.assertIsInstance(self.ner, BERT_NER)
        self.assertTrue(hasattr(self.ner, 'udpipe_lang'))
        self.assertTrue(hasattr(self.ner, 'use_shapes'))
        self.assertTrue(hasattr(self.ner, 'use_nlp_features'))
        self.assertTrue(hasattr(self.ner, 'batch_size'))
        self.assertTrue(hasattr(self.ner, 'lstm_units'))
        self.assertTrue(hasattr(self.ner, 'lr'))
        self.assertTrue(hasattr(self.ner, 'l2_reg'))
        self.assertTrue(hasattr(self.ner, 'clip_norm'))
        self.assertTrue(hasattr(self.ner, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(self.ner, 'finetune_bert'))
        self.assertTrue(hasattr(self.ner, 'max_epochs'))
        self.assertTrue(hasattr(self.ner, 'patience'))
        self.assertTrue(hasattr(self.ner, 'random_seed'))
        self.assertTrue(hasattr(self.ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.ner, 'verbose'))
        self.assertIsInstance(self.ner.batch_size, int)
        self.assertIsInstance(self.ner.lstm_units, int)
        self.assertIsInstance(self.ner.lr, float)
        self.assertIsInstance(self.ner.l2_reg, float)
        self.assertIsInstance(self.ner.clip_norm, float)
        self.assertIsInstance(self.ner.bert_hub_module_handle, str)
        self.assertIsInstance(self.ner.udpipe_lang, str)
        self.assertIsInstance(self.ner.finetune_bert, bool)
        self.assertIsInstance(self.ner.max_epochs, int)
        self.assertIsInstance(self.ner.patience, int)
        self.assertIsNone(self.ner.random_seed)
        self.assertIsInstance(self.ner.gpu_memory_frac, float)
        self.assertIsInstance(self.ner.max_seq_length, int)
        self.assertIsInstance(self.ner.validation_fraction, float)
        self.assertIsInstance(self.ner.verbose, bool)
        self.assertIsInstance(self.ner.use_shapes, bool)
        self.assertIsInstance(self.ner.use_nlp_features, bool)

    def test_check_params_positive(self):
        BERT_NER.check_params(
            bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1', finetune_bert=True,
            batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0, validation_fraction=0.0,
            max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=None,
            use_shapes=True, use_nlp_features=True, udpipe_lang='en'
        )
        self.assertTrue(True)

    def test_check_params_negative001(self):
        true_err_msg = re.escape('`bert_hub_module_handle` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=False, udpipe_lang='en'
            )

    def test_check_params_negative002(self):
        true_err_msg = re.escape('`bert_hub_module_handle` is wrong! Expected `{0}`, got `{1}`.'.format(
            type('abc'), type(123)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle=1, finetune_bert=True,
                batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=False, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative003(self):
        true_err_msg = re.escape('`batch_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative004(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size='32', max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative005(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=-3, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative006(self):
        true_err_msg = re.escape('`max_epochs` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative007(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs='10', patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42, lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative008(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=-3, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative009(self):
        true_err_msg = re.escape('`patience` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative010(self):
        true_err_msg = re.escape('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience='3', gpu_memory_frac=1.0, verbose=False,
                random_seed=42, lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative011(self):
        true_err_msg = re.escape('`patience` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=-3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative012(self):
        true_err_msg = re.escape('`max_seq_length` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, lr=1e-3, l2_reg=1e-4, clip_norm=5.0, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative013(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length='512', lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative014(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=-3, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative015(self):
        true_err_msg = re.escape('`validation_fraction` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative016(self):
        true_err_msg = re.escape('`validation_fraction` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction='0.1', max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42, lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative017(self):
        true_err_msg = '`validation_fraction` is wrong! Expected a positive floating-point value greater than or ' \
                       'equal to 0.0, but {0} is not positive.'.format(-0.1)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=-0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative018(self):
        true_err_msg = '`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, but ' \
                       '{0} is not less than 1.0.'.format(1.1)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=1.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative019(self):
        true_err_msg = re.escape('`gpu_memory_frac` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative020(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac='1.0', verbose=False,
                random_seed=42, lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative021(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(-1.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=-1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative022(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(1.3))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.3, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative023(self):
        true_err_msg = re.escape('`lr` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative024(self):
        true_err_msg = re.escape('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr='1e-3', l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative025(self):
        true_err_msg = re.escape('`lr` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=0.0, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative026(self):
        true_err_msg = re.escape('`lr` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative027(self):
        true_err_msg = re.escape('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr='1e-3', l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative028(self):
        true_err_msg = re.escape('`lr` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=0.0, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative029(self):
        true_err_msg = re.escape('`l2_reg` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, clip_norm=5.0, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative030(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg='1e-4', clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative031(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected a non-negative floating-point value, but {0} is '
                                 'negative.'.format(-2.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=-2.0, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative032(self):
        true_err_msg = re.escape('`finetune_bert` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, clip_norm=5.0,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative033(self):
        true_err_msg = re.escape('`finetune_bert` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert='True', batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative034(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, random_seed=42, lstm_units=128,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative035(self):
        true_err_msg = re.escape('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose='False',
                random_seed=42, lstm_units=128, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative036(self):
        true_err_msg = re.escape('`lstm_units` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative037(self):
        true_err_msg = re.escape('`lstm_units` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                lstm_units='128', finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4,
                clip_norm=5.0, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_params_negative038(self):
        true_err_msg = re.escape('`lstm_units` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, clip_norm=5.0,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42,
                lstm_units=-3, use_shapes=True, use_nlp_features=True, udpipe_lang='en'
            )

    def test_check_X_positive(self):
        X = ['abc', 'defgh', '4wdffg']
        BERT_NER.check_X(X, 'X_train')
        self.assertTrue(True)

    def test_check_X_negative01(self):
        X = {'abc', 'defgh', '4wdffg'}
        true_err_msg = re.escape('`X_train` is wrong, because it is not list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_X(X, 'X_train')

    def test_check_X_negative02(self):
        X = np.random.uniform(-1.0, 1.0, (10, 2))
        true_err_msg = re.escape('`X_train` is wrong, because it is not 1-D list!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_X(X, 'X_train')

    def test_check_X_negative03(self):
        X = ['abc', 23, '4wdffg']
        true_err_msg = re.escape('Item 1 of `X_train` is wrong, because it is not string-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_X(X, 'X_train')

    def text_check_Xy_positive(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_classes_list = ('LOC', 'ORG', 'PER')
        self.assertEqual(true_classes_list, BERT_NER.check_Xy(X, 'X_train', y, 'y_train'))

    def text_check_Xy_negative01(self):
        X = {
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        }
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('`X_train` is wrong, because it is not list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative02(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = {
            '1': {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            '2': {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        }
        true_err_msg = re.escape('`y_train` is wrong, because it is not a list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative03(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = np.random.uniform(-1.0, 1.0, (10, 2))
        true_err_msg = re.escape('`y_train` is wrong, because it is not 1-D list!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative04(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            },
            {
                'LOC': [(17, 24), (117, 130)]
            }
        ]
        true_err_msg = re.escape('Length of `X_train` does not correspond to length of `y_train`! 2 != 3')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative05(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            4
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because it is not a dictionary-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative06(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                1: [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because its key `1` is not a string-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative07(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'O': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because its key `O` incorrectly specifies a named '
                                 'entity!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative08(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                '123': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because its key `123` incorrectly specifies a named '
                                 'entity!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative09(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'loc': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because its key `loc` incorrectly specifies a named '
                                 'entity!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative10(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': {1, 2}
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because its value `{0}` is not a list-like '
                                 'object!'.format(y[0]['PER']))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative11(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), 63],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because named entity bounds `63` are not specified as '
                                 'list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative12(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77, 81)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because named entity bounds `{0}` are not specified as '
                                 '2-D list!'.format((63, 77, 81)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative13(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (219, 196)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because named entity bounds `{0}` are '
                                 'incorrect!'.format((219, 196)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative14(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 519)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because named entity bounds `{0}` are '
                                 'incorrect!'.format((196, 519)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative15(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(-1, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because named entity bounds `{0}` are '
                                 'incorrect!'.format((-1, 137)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def test_detect_token_labels_positive01(self):
        # source_text = 'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози.'
        tokenized_text = ['Ба', '##рак', 'Об', '##ама', 'принимает', 'в', 'Б', '##елом', 'доме', 'своего',
                          'французского', 'кол', '##ле', '##гу', 'Н', '##ико', '##ля', 'Са', '##рко', '##зи', '.']
        token_bounds = [(0, 2), (2, 5), (6, 8), (8, 11), (12, 21), (22, 23), (24, 25), (25, 29), (30, 34), (35, 41),
                        (42, 54), (55, 58), (58, 60), (60, 62), (63, 64), (64, 67), (67, 69), (70, 72), (72, 75),
                        (75, 77), (77, 78)]
        indices_of_named_entities = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 1}
        y_true = np.array(
            [0, 2, 1, 1, 1, 0, 0, 4, 3, 3, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = BERT_NER.detect_token_labels(tokenized_text, token_bounds, indices_of_named_entities, label_IDs, 32)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_detect_token_labels_positive02(self):
        # source_text = 'С 1876 г Павлов ассистирует профессору К. Н. Устимовичу в Медико-хирургической академии и ' \
        #               'параллельно изучает физиологию кровообращения.'
        tokenized_text = ['С', '1876', 'г', 'Павло', '##в', 'а', '##сси', '##сти', '##рует', 'профессор', '##у', 'К',
                          '.', 'Н', '.', 'У', '##сти', '##мов', '##ич', '##у', 'в', 'М', '##еди', '##ко', '-',
                          'х', '##ир', '##ург', '##ической', 'академии', 'и', 'пара', '##лл', '##ельно',
                          'из', '##уч', '##ает', 'ф', '##из', '##ио', '##логи', '##ю',
                          'к', '##рово', '##об', '##ращения', '.']
        token_bounds = [(0, 1), (2, 6), (7, 8), (9, 14), (14, 15), (16, 17), (17, 20), (20, 23), (23, 27), (28, 37),
                        (37, 38), (39, 40), (40, 41), (42, 43), (43, 44), (45, 46), (46, 49), (49, 52), (52, 54),
                        (54, 55), (56, 57), (58, 59), (59, 62), (62, 64), (64, 65), (65, 66), (66, 68), (68, 71),
                        (71, 78), (79, 87), (88, 89), (90, 94), (94, 96), (96, 101), (102, 104), (104, 106), (106, 109),
                        (110, 111), (111, 113), (113, 115), (115, 119), (119, 120), (121, 122), (122, 126), (126, 128),
                        (128, 135), (135, 136)]
        indices_of_named_entities = np.array(
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 3, 4: 2, 5: 4}
        y_true = np.array(
            [0, 0, 2, 1, 4, 3, 0, 0, 0, 0, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = BERT_NER.detect_token_labels(tokenized_text, token_bounds, indices_of_named_entities, label_IDs, 64)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_detect_token_labels_positive03(self):
        # source_text = 'Весной 1890 года Варшавский и Томский университеты избирают его профессором.'
        tokenized_text = ['В', '##есной', '1890', 'года', 'В', '##ар', '##ша', '##вский', 'и', 'Томск', '##ий',
                          'университет', '##ы', 'из', '##бира', '##ют', 'его', 'профессором', '.']
        token_bounds = [(0, 1), (1, 6), (7, 11), (12, 16), (17, 18), (18, 20), (20, 22), (22, 27), (28, 29), (30, 35),
                        (35, 37), (38, 49), (49, 50), (51, 52), (53, 57), (57, 59), (60, 63), (64, 75), (75, 76)]
        indices_of_named_entities = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 2}
        y_true = np.array(
            [0, 2, 1, 1, 1, 4, 3, 3, 3, 3, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = BERT_NER.detect_token_labels(tokenized_text, token_bounds, indices_of_named_entities, label_IDs, 32)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_calculate_indices_of_named_entities(self):
        source_text = 'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози.'
        classes_list = ('LOCATION', 'ORG', 'PERSON')
        named_entities = {'PERSON': [(0, 11), (63, 77)], 'LOCATION': [(24, 34)]}
        true_indices = np.array(
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 0],
            dtype=np.int32
        )
        true_labels_to_classes = {1: 1, 2: 3, 3: 3}
        indices, labels_to_classes = BERT_NER.calculate_indices_of_named_entities(source_text, classes_list,
                                                                                  named_entities)
        self.assertIsInstance(indices, np.ndarray)
        self.assertIsInstance(labels_to_classes, dict)
        self.assertEqual(true_indices.shape, indices.shape)
        self.assertEqual(true_indices.tolist(), indices.tolist())
        self.assertEqual(set(true_labels_to_classes.keys()), set(labels_to_classes.keys()))
        for label_ID in true_labels_to_classes:
            self.assertEqual(true_labels_to_classes[label_ID], labels_to_classes[label_ID])

    def test_fit_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=3, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, lstm_units=32, udpipe_lang='ru')
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, BERT_NER)
        self.assertTrue(hasattr(res, 'udpipe_lang'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lstm_units'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'clip_norm'))
        self.assertTrue(hasattr(res, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_bert'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'use_shapes'))
        self.assertTrue(hasattr(res, 'use_nlp_features'))
        self.assertIsInstance(res.udpipe_lang, str)
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lstm_units, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.clip_norm, float)
        self.assertIsInstance(res.bert_hub_module_handle, str)
        self.assertIsInstance(res.finetune_bert, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertIsInstance(res.use_shapes, bool)
        self.assertIsInstance(res.use_nlp_features, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'shapes_list_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertTrue(hasattr(res, 'universal_pos_tags_dict_'))
        self.assertTrue(hasattr(res, 'universal_dependencies_dict_'))
        self.assertTrue(hasattr(res, 'nlp_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        self.assertIsInstance(res.shapes_list_, tuple)
        self.assertGreater(len(res.shapes_list_), 3)
        self.assertEqual(res.shapes_list_[-3:], ('[CLS]', '[SEP]', '[UNK]'))
        self.assertIsInstance(res.universal_pos_tags_dict_, dict)
        self.assertIsInstance(res.universal_dependencies_dict_, dict)
        self.assertIsInstance(res.nlp_, UDPipeLanguage)
        self.assertEqual(len(res.universal_pos_tags_dict_), len(UNIVERSAL_POS_TAGS))
        self.assertEqual(len(res.universal_dependencies_dict_), len(UNIVERSAL_DEPENDENCIES))

    def test_fit_positive02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=True, max_epochs=3, batch_size=2, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=42, lstm_units=32, udpipe_lang='ru')
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, BERT_NER)
        self.assertTrue(hasattr(res, 'udpipe_lang'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lstm_units'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'clip_norm'))
        self.assertTrue(hasattr(res, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_bert'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'use_shapes'))
        self.assertTrue(hasattr(res, 'use_nlp_features'))
        self.assertIsInstance(res.udpipe_lang, str)
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lstm_units, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.clip_norm, float)
        self.assertIsInstance(res.bert_hub_module_handle, str)
        self.assertIsInstance(res.finetune_bert, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertIsInstance(res.use_shapes, bool)
        self.assertIsInstance(res.use_nlp_features, bool)
        self.assertEqual(res.random_seed, 42)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'shapes_list_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        self.assertIsInstance(res.shapes_list_, tuple)
        self.assertGreater(len(res.shapes_list_), 3)
        self.assertEqual(res.shapes_list_[-3:], ('[CLS]', '[SEP]', '[UNK]'))

    def test_fit_positive03(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=3, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, lstm_units=None, clip_norm=None,
                            udpipe_lang='ru', use_shapes=False, use_nlp_features=True)
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, BERT_NER)
        self.assertTrue(hasattr(res, 'udpipe_lang'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lstm_units'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'clip_norm'))
        self.assertTrue(hasattr(res, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_bert'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'use_shapes'))
        self.assertTrue(hasattr(res, 'use_nlp_features'))
        self.assertIsInstance(res.udpipe_lang, str)
        self.assertIsInstance(res.batch_size, int)
        self.assertIsNone(res.lstm_units)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsNone(res.clip_norm, None)
        self.assertIsInstance(res.bert_hub_module_handle, str)
        self.assertIsInstance(res.finetune_bert, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertIsInstance(res.use_shapes, bool)
        self.assertIsInstance(res.use_nlp_features, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'shapes_list_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        self.assertIsInstance(res.shapes_list_, tuple)
        self.assertGreater(len(res.shapes_list_), 3)
        self.assertEqual(res.shapes_list_[-3:], ('[CLS]', '[SEP]', '[UNK]'))

    def test_fit_predict(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=5, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=42, udpipe_lang='ru', use_shapes=True,
                            use_nlp_features=False)
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, BERT_NER)
        self.assertTrue(hasattr(res, 'udpipe_lang'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lstm_units'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'clip_norm'))
        self.assertTrue(hasattr(res, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_bert'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'use_shapes'))
        self.assertTrue(hasattr(res, 'use_nlp_features'))
        self.assertIsInstance(res.udpipe_lang, str)
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lstm_units, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.clip_norm, float)
        self.assertIsInstance(res.bert_hub_module_handle, str)
        self.assertIsInstance(res.finetune_bert, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertIsInstance(res.use_shapes, bool)
        self.assertIsInstance(res.use_nlp_features, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'shapes_list_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        self.assertIsInstance(res.shapes_list_, tuple)
        self.assertGreater(len(res.shapes_list_), 3)
        self.assertEqual(res.shapes_list_[-3:], ('[CLS]', '[SEP]', '[UNK]'))
        y_pred = res.predict(X_train)
        self.assertIsInstance(y_pred, list)
        self.assertEqual(len(X_train), len(y_pred))
        for sample_idx in range(len(y_pred)):
            self.assertIsInstance(y_pred[sample_idx], dict)
        f1, precision, recall, _ = calculate_prediction_quality(y_train, y_pred, res.classes_list_)
        self.assertGreater(f1, 0.0)
        self.assertGreater(precision, 0.0)
        self.assertGreater(recall, 0.0)

    def test_predict_negative(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=3, batch_size=4, random_seed=None, udpipe_lang='ru')
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        with self.assertRaises(NotFittedError):
            _ = self.ner.predict(X_train)

    def test_tokenize_all_01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=1, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, udpipe_lang='ru', use_nlp_features=True,
                            use_shapes=True)
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        self.ner.fit(X_train, y_train)
        res = self.ner.tokenize_all(X_train, y_train, shapes_vocabulary=self.ner.shapes_list_)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 4)
        X_train_tokenized, y_train_tokenized, shapes_list, bounds_of_tokens_for_training = res
        self.assertIsInstance(X_train_tokenized, list)
        self.assertIsInstance(y_train_tokenized, np.ndarray)
        self.assertIs(self.ner.shapes_list_, shapes_list)
        self.assertIsInstance(bounds_of_tokens_for_training, np.ndarray)
        self.assertEqual(len(X_train_tokenized), 4)
        self.assertEqual(y_train_tokenized.shape, (len(y_train), self.ner.max_seq_length))
        for data_idx in range(3):
            self.assertIsInstance(X_train_tokenized[data_idx], np.ndarray)
            self.assertEqual(X_train_tokenized[data_idx].shape, (len(X_train), self.ner.max_seq_length))
        data_idx = 3
        self.assertIsInstance(X_train_tokenized[data_idx], np.ndarray)
        self.assertEqual(len(X_train_tokenized[data_idx].shape), 3)
        self.assertEqual(X_train_tokenized[data_idx].shape[0], len(X_train))
        self.assertEqual(X_train_tokenized[data_idx].shape[1], self.ner.max_seq_length)
        self.assertGreater(X_train_tokenized[data_idx].shape[2],
                           4 + len(UNIVERSAL_POS_TAGS) + len(UNIVERSAL_DEPENDENCIES))
        for sample_idx in range(X_train_tokenized[data_idx].shape[0]):
            n = 0
            for token_idx in range(X_train_tokenized[data_idx].shape[1]):
                if X_train_tokenized[data_idx][sample_idx][token_idx].sum() < 1e-3:
                    break
                n += 1
            self.assertGreater(n, 0, msg='Sample {0}: additional features are not defined!'.format(sample_idx))
            for token_idx in range(n):
                self.assertAlmostEqual(X_train_tokenized[data_idx][sample_idx][token_idx][0:4].sum(), 1.0,
                                       msg='Sample {0}, token {1}: additional features are wrong!'.format(
                                           sample_idx, token_idx))
            for token_idx in range(1, n - 1):
                start_pos = 4
                end_pos = 4 + len(UNIVERSAL_POS_TAGS)
                self.assertAlmostEqual(X_train_tokenized[data_idx][sample_idx][token_idx][start_pos:end_pos].sum(), 1.0,
                                       msg='Sample {0}, token {1}: part of speech is not defined!'.format(
                                           sample_idx, token_idx))
                start_pos = 4 + len(UNIVERSAL_POS_TAGS)
                end_pos = 4 + len(UNIVERSAL_POS_TAGS) + len(UNIVERSAL_DEPENDENCIES)
                self.assertGreaterEqual(X_train_tokenized[data_idx][sample_idx][token_idx][start_pos:end_pos].sum(),
                                        1.0,
                                        msg='Sample {0}, token {1}: dependency tag is not defined!'.format(
                                            sample_idx, token_idx))

    def test_tokenize_all_02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=1, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, udpipe_lang='ru', use_shapes=False,
                            use_nlp_features=False)
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        self.ner.fit(X_train, y_train)
        res = self.ner.tokenize_all(X_train, y_train, shapes_vocabulary=self.ner.shapes_list_)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 4)
        X_train_tokenized, y_train_tokenized, shapes_list, bounds_of_tokens_for_training = res
        self.assertIsInstance(X_train_tokenized, list)
        self.assertIsInstance(y_train_tokenized, np.ndarray)
        self.assertIs(self.ner.shapes_list_, shapes_list)
        self.assertIsInstance(bounds_of_tokens_for_training, np.ndarray)
        self.assertEqual(len(X_train_tokenized), 3)
        self.assertEqual(y_train_tokenized.shape, (len(y_train), self.ner.max_seq_length))
        for data_idx in range(3):
            self.assertIsInstance(X_train_tokenized[data_idx], np.ndarray)
            self.assertEqual(X_train_tokenized[data_idx].shape, (len(X_train), self.ner.max_seq_length))

    def test_serialize_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=5, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=42, udpipe_lang='ru')
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, BERT_NER)
        self.assertTrue(hasattr(res, 'udpipe_lang'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lstm_units'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'clip_norm'))
        self.assertTrue(hasattr(res, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_bert'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'use_shapes'))
        self.assertTrue(hasattr(res, 'use_nlp_features'))
        self.assertIsInstance(res.udpipe_lang, str)
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lstm_units, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.clip_norm, float)
        self.assertIsInstance(res.bert_hub_module_handle, str)
        self.assertIsInstance(res.finetune_bert, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertIsInstance(res.use_shapes, bool)
        self.assertIsInstance(res.use_nlp_features, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'shapes_list_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        self.assertIsInstance(res.shapes_list_, tuple)
        self.assertGreater(len(res.shapes_list_), 3)
        self.assertEqual(res.shapes_list_[-3:], ('[CLS]', '[SEP]', '[UNK]'))
        y_pred1 = res.predict(X_train)
        self.assertIsInstance(y_pred1, list)
        self.assertEqual(len(X_train), len(y_pred1))
        for sample_idx in range(len(y_pred1)):
            self.assertIsInstance(y_pred1[sample_idx], dict)
        f1, precision, recall, _ = calculate_prediction_quality(y_train, y_pred1, res.classes_list_)
        self.assertGreater(f1, 0.0)
        self.assertGreater(precision, 0.0)
        self.assertGreater(recall, 0.0)
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as fp:
            self.temp_file_name = fp.name
        with open(self.temp_file_name, mode='wb') as fp:
            pickle.dump(res, fp)
        del res, self.ner
        gc.collect()
        with open(self.temp_file_name, mode='rb') as fp:
            self.ner = pickle.load(fp)
        y_pred2 = self.ner.predict(X_train)
        self.assertIsInstance(y_pred2, list)
        self.assertEqual(len(y_pred2), len(y_pred2))
        for sample_idx in range(len(y_pred2)):
            self.assertIsInstance(y_pred2[sample_idx], dict)
            self.assertEqual(set(y_pred1[sample_idx]), set(y_pred2[sample_idx]))
            for ne_type in y_pred1[sample_idx]:
                self.assertEqual(y_pred1[sample_idx][ne_type], y_pred2[sample_idx][ne_type])

    def test_serialize_positive02(self):
        self.ner = BERT_NER(random_seed=31, udpipe_lang='ru')
        old_udpipe_lang = self.ner.udpipe_lang
        old_batch_size = self.ner.batch_size
        old_lstm_units = self.ner.lstm_units
        old_lr = self.ner.lr
        old_l2_reg = self.ner.l2_reg
        old_clip_norm = self.ner.clip_norm
        old_bert_hub_module_handle = self.ner.bert_hub_module_handle
        old_finetune_bert = self.ner.finetune_bert
        old_max_epochs = self.ner.max_epochs
        old_patience = self.ner.patience
        old_random_seed = self.ner.random_seed
        old_gpu_memory_frac = self.ner.gpu_memory_frac
        old_max_seq_length = self.ner.max_seq_length
        old_validation_fraction = self.ner.validation_fraction
        old_verbose = self.ner.verbose
        old_use_shapes = self.ner.use_shapes
        old_use_nlp_features = self.ner.use_nlp_features
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as fp:
            self.temp_file_name = fp.name
        with open(self.temp_file_name, mode='wb') as fp:
            pickle.dump(self.ner, fp)
        del self.ner
        gc.collect()
        with open(self.temp_file_name, mode='rb') as fp:
            self.ner = pickle.load(fp)
        self.assertIsInstance(self.ner, BERT_NER)
        self.assertTrue(hasattr(self.ner, 'udpipe_lang'))
        self.assertTrue(hasattr(self.ner, 'batch_size'))
        self.assertTrue(hasattr(self.ner, 'lstm_units'))
        self.assertTrue(hasattr(self.ner, 'lr'))
        self.assertTrue(hasattr(self.ner, 'l2_reg'))
        self.assertTrue(hasattr(self.ner, 'clip_norm'))
        self.assertTrue(hasattr(self.ner, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(self.ner, 'finetune_bert'))
        self.assertTrue(hasattr(self.ner, 'max_epochs'))
        self.assertTrue(hasattr(self.ner, 'patience'))
        self.assertTrue(hasattr(self.ner, 'random_seed'))
        self.assertTrue(hasattr(self.ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.ner, 'verbose'))
        self.assertTrue(hasattr(self.ner, 'use_shapes'))
        self.assertTrue(hasattr(self.ner, 'use_nlp_features'))
        self.assertEqual(self.ner.udpipe_lang, old_udpipe_lang)
        self.assertEqual(self.ner.batch_size, old_batch_size)
        self.assertEqual(self.ner.lstm_units, old_lstm_units)
        self.assertAlmostEqual(self.ner.lr, old_lr)
        self.assertAlmostEqual(self.ner.l2_reg, old_l2_reg)
        self.assertAlmostEqual(self.ner.clip_norm, old_clip_norm)
        self.assertEqual(self.ner.bert_hub_module_handle, old_bert_hub_module_handle)
        self.assertEqual(self.ner.finetune_bert, old_finetune_bert)
        self.assertEqual(self.ner.max_epochs, old_max_epochs)
        self.assertEqual(self.ner.patience, old_patience)
        self.assertAlmostEqual(self.ner.gpu_memory_frac, old_gpu_memory_frac)
        self.assertEqual(self.ner.max_seq_length, old_max_seq_length)
        self.assertAlmostEqual(self.ner.validation_fraction, old_validation_fraction)
        self.assertEqual(self.ner.verbose, old_verbose)
        self.assertEqual(self.ner.use_shapes, old_use_shapes)
        self.assertEqual(self.ner.use_nlp_features, old_use_nlp_features)
        self.assertEqual(self.ner.random_seed, old_random_seed)

    def test_copy_positive01(self):
        self.ner = BERT_NER(random_seed=0, udpipe_lang='ru', use_shapes=False, use_nlp_features=True)
        self.another_ner = copy.copy(self.ner)
        self.assertIsInstance(self.another_ner, BERT_NER)
        self.assertIsNot(self.ner, self.another_ner)
        self.assertTrue(hasattr(self.another_ner, 'udpipe_lang'))
        self.assertTrue(hasattr(self.another_ner, 'batch_size'))
        self.assertTrue(hasattr(self.another_ner, 'lstm_units'))
        self.assertTrue(hasattr(self.another_ner, 'lr'))
        self.assertTrue(hasattr(self.another_ner, 'l2_reg'))
        self.assertTrue(hasattr(self.another_ner, 'clip_norm'))
        self.assertTrue(hasattr(self.another_ner, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(self.another_ner, 'finetune_bert'))
        self.assertTrue(hasattr(self.another_ner, 'max_epochs'))
        self.assertTrue(hasattr(self.another_ner, 'patience'))
        self.assertTrue(hasattr(self.another_ner, 'random_seed'))
        self.assertTrue(hasattr(self.another_ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.another_ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.another_ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.another_ner, 'verbose'))
        self.assertTrue(hasattr(self.another_ner, 'use_shapes'))
        self.assertTrue(hasattr(self.another_ner, 'use_nlp_features'))
        self.assertEqual(self.ner.udpipe_lang, self.another_ner.udpipe_lang)
        self.assertEqual(self.ner.batch_size, self.another_ner.batch_size)
        self.assertEqual(self.ner.lstm_units, self.another_ner.lstm_units)
        self.assertAlmostEqual(self.ner.lr, self.another_ner.lr)
        self.assertAlmostEqual(self.ner.l2_reg, self.another_ner.l2_reg)
        self.assertAlmostEqual(self.ner.clip_norm, self.another_ner.clip_norm)
        self.assertEqual(self.ner.bert_hub_module_handle, self.another_ner.bert_hub_module_handle)
        self.assertEqual(self.ner.finetune_bert, self.another_ner.finetune_bert)
        self.assertEqual(self.ner.max_epochs, self.another_ner.max_epochs)
        self.assertEqual(self.ner.patience, self.another_ner.patience)
        self.assertEqual(self.ner.random_seed, self.another_ner.random_seed)
        self.assertAlmostEqual(self.ner.gpu_memory_frac, self.another_ner.gpu_memory_frac)
        self.assertEqual(self.ner.max_seq_length, self.another_ner.max_seq_length)
        self.assertAlmostEqual(self.ner.validation_fraction, self.another_ner.validation_fraction)
        self.assertEqual(self.ner.verbose, self.another_ner.verbose)
        self.assertEqual(self.ner.use_shapes, self.another_ner.use_shapes)
        self.assertEqual(self.ner.use_nlp_features, self.another_ner.use_nlp_features)

    def test_copy_positive02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = BERT_NER(finetune_bert=False, max_epochs=3, batch_size=4, max_seq_length=128, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, udpipe_lang='ru')
        X_train, y_train = load_dataset_from_json(os.path.join(base_dir, 'true_named_entities.json'))
        self.ner.fit(X_train, y_train)
        self.another_ner = copy.copy(self.ner)
        self.assertIsInstance(self.another_ner, BERT_NER)
        self.assertIsNot(self.ner, self.another_ner)
        self.assertTrue(hasattr(self.another_ner, 'udpipe_lang'))
        self.assertTrue(hasattr(self.another_ner, 'batch_size'))
        self.assertTrue(hasattr(self.another_ner, 'lstm_units'))
        self.assertTrue(hasattr(self.another_ner, 'lr'))
        self.assertTrue(hasattr(self.another_ner, 'l2_reg'))
        self.assertTrue(hasattr(self.another_ner, 'clip_norm'))
        self.assertTrue(hasattr(self.another_ner, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(self.another_ner, 'finetune_bert'))
        self.assertTrue(hasattr(self.another_ner, 'max_epochs'))
        self.assertTrue(hasattr(self.another_ner, 'patience'))
        self.assertTrue(hasattr(self.another_ner, 'random_seed'))
        self.assertTrue(hasattr(self.another_ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.another_ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.another_ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.another_ner, 'verbose'))
        self.assertTrue(hasattr(self.another_ner, 'use_shapes'))
        self.assertTrue(hasattr(self.another_ner, 'use_nlp_features'))
        self.assertTrue(hasattr(self.another_ner, 'classes_list_'))
        self.assertTrue(hasattr(self.another_ner, 'shapes_list_'))
        self.assertTrue(hasattr(self.another_ner, 'tokenizer_'))
        self.assertTrue(hasattr(self.another_ner, 'sess_'))
        self.assertEqual(self.ner.udpipe_lang, self.another_ner.udpipe_lang)
        self.assertEqual(self.ner.batch_size, self.another_ner.batch_size)
        self.assertEqual(self.ner.lstm_units, self.another_ner.lstm_units)
        self.assertAlmostEqual(self.ner.lr, self.another_ner.lr)
        self.assertAlmostEqual(self.ner.l2_reg, self.another_ner.l2_reg)
        self.assertAlmostEqual(self.ner.clip_norm, self.another_ner.clip_norm)
        self.assertEqual(self.ner.bert_hub_module_handle, self.another_ner.bert_hub_module_handle)
        self.assertEqual(self.ner.finetune_bert, self.another_ner.finetune_bert)
        self.assertEqual(self.ner.max_epochs, self.another_ner.max_epochs)
        self.assertEqual(self.ner.patience, self.another_ner.patience)
        self.assertEqual(self.ner.random_seed, self.another_ner.random_seed)
        self.assertAlmostEqual(self.ner.gpu_memory_frac, self.another_ner.gpu_memory_frac)
        self.assertEqual(self.ner.max_seq_length, self.another_ner.max_seq_length)
        self.assertAlmostEqual(self.ner.validation_fraction, self.another_ner.validation_fraction)
        self.assertEqual(self.ner.verbose, self.another_ner.verbose)
        self.assertEqual(self.ner.use_shapes, self.another_ner.use_shapes)
        self.assertEqual(self.ner.use_nlp_features, self.another_ner.use_nlp_features)
        self.assertIs(self.ner.classes_list_, self.another_ner.classes_list_)
        self.assertIs(self.ner.shapes_list_, self.another_ner.shapes_list_)
        self.assertIs(self.ner.tokenizer_, self.another_ner.tokenizer_)
        self.assertIs(self.ner.sess_, self.another_ner.sess_)

    def test_calculate_bounds_of_named_entities(self):
        bounds_of_tokens = [(0, 2), (2, 5), (5, 8), (8, 10), (11, 16), (17, 20), (20, 22), (22, 26), (26, 27), (28, 31),
                            (31, 34), (34, 37), (38, 48), (49, 52), (52, 54), (55, 57), (58, 59), (59, 61), (61, 63),
                            (64, 70), (71, 83), (84, 87), (87, 90), (90, 93), (93, 95), (95, 98), (98, 99)]
        classes_list = ('LOCATION', 'ORG', 'PERSON')
        labels_of_tokens = [0, 0, 2, 1, 1, 2, 1, 0, 0, 0, 4, 3, 0, 6, 5, 5, 5, 0, 5, 5, 0, 2, 2, 3, 3, 6, 5]
        true_entities = {
            'LOCATION': [(5, 16), (17, 22), (84, 87), (87, 90)],
            'ORG': [(31, 37), (90, 95)],
            'PERSON': [(49, 59), (61, 70), (95, 99)]
        }
        calc_entities = BERT_NER.calculate_bounds_of_named_entities(bounds_of_tokens, classes_list, labels_of_tokens)
        self.assertIsInstance(calc_entities, dict)
        self.assertEqual(set(true_entities.keys()), set(calc_entities.keys()))
        for entity_type in true_entities:
            self.assertEqual(true_entities[entity_type], calc_entities[entity_type])

    def test_get_shape_of_string_positive01(self):
        src = '##чники'
        dst = 'a'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive02(self):
        src = 'уже'
        dst = 'a'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive03(self):
        src = 'К'
        dst = 'A'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive04(self):
        src = 'Однако'
        dst = 'Aa'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive05(self):
        src = '66–67'
        dst = 'D-D'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive06(self):
        src = '[UNK]'
        dst = '[UNK]'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive07(self):
        src = '…'
        dst = 'U'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_positive08(self):
        src = ','
        dst = 'P'
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_shape_of_string_negative(self):
        src = ''
        dst = ''
        self.assertEqual(dst, BERT_NER.get_shape_of_string(src))

    def test_get_subword_ID_positive01(self):
        src = '##чники'
        dst = 2
        self.assertEqual(dst, BERT_NER.get_subword_ID(src))

    def test_get_subword_ID_positive02(self):
        src = 'Однако'
        dst = 3
        self.assertEqual(dst, BERT_NER.get_subword_ID(src))

    def test_get_subword_ID_positive03(self):
        src = '[CLS]'
        dst = 0
        self.assertEqual(dst, BERT_NER.get_subword_ID(src))

    def test_get_subword_ID_positive04(self):
        src = '[SEP]'
        dst = 1
        self.assertEqual(dst, BERT_NER.get_subword_ID(src))


if __name__ == '__main__':
    unittest.main(verbosity=2)
