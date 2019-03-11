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

try:
    from elmo_ner.elmo_ner import ELMo_NER
    from elmo_ner.utils import load_dataset
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from elmo_ner.elmo_ner import ELMo_NER
    from elmo_ner.utils import load_dataset


class TestELMoNER(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ELMO_HUB_MODULE = 'http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'

    def tearDown(self):
        if hasattr(self, 'ner'):
            del self.ner
        if hasattr(self, 'another_ner'):
            del self.another_ner
        if hasattr(self, 'temp_file_name'):
            if os.path.isfile(self.temp_file_name):
                os.remove(self.temp_file_name)

    def test_creation(self):
        self.ner = ELMo_NER(elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        self.assertIsInstance(self.ner, ELMo_NER)
        self.assertTrue(hasattr(self.ner, 'batch_size'))
        self.assertTrue(hasattr(self.ner, 'lr'))
        self.assertTrue(hasattr(self.ner, 'l2_reg'))
        self.assertTrue(hasattr(self.ner, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(self.ner, 'finetune_elmo'))
        self.assertTrue(hasattr(self.ner, 'max_epochs'))
        self.assertTrue(hasattr(self.ner, 'patience'))
        self.assertTrue(hasattr(self.ner, 'random_seed'))
        self.assertTrue(hasattr(self.ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.ner, 'verbose'))
        self.assertIsInstance(self.ner.batch_size, int)
        self.assertIsInstance(self.ner.lr, float)
        self.assertIsInstance(self.ner.l2_reg, float)
        self.assertIsInstance(self.ner.finetune_elmo, bool)
        self.assertIsInstance(self.ner.max_epochs, int)
        self.assertIsInstance(self.ner.patience, int)
        self.assertIsNone(self.ner.random_seed)
        self.assertIsInstance(self.ner.gpu_memory_frac, float)
        self.assertIsInstance(self.ner.max_seq_length, int)
        self.assertIsInstance(self.ner.validation_fraction, float)
        self.assertIsInstance(self.ner.verbose, bool)

    def test_check_params_positive(self):
        ELMo_NER.check_params(
            elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512, lr=1e-3,
            l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
            random_seed=42
        )
        self.assertTrue(True)

    def test_check_params_negative001(self):
        true_err_msg = re.escape('`elmo_hub_module_handle` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                finetune_elmo=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative002(self):
        true_err_msg = re.escape('`elmo_hub_module_handle` is wrong! Expected `{0}`, got `{1}`.'.format(
            type('abc'), type(123)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=1, finetune_elmo=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative003(self):
        true_err_msg = re.escape('`batch_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, max_seq_length=512, lr=1e-3,
                l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42
            )

    def test_check_params_negative004(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size='32', max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative005(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=-3, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative006(self):
        true_err_msg = re.escape('`max_epochs` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42
            )

    def test_check_params_negative007(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs='10', patience=3,
                gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative008(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=-3, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative009(self):
        true_err_msg = re.escape('`patience` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, gpu_memory_frac=1.0, verbose=False,
                random_seed=42
            )

    def test_check_params_negative010(self):
        true_err_msg = re.escape('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience='3', gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative011(self):
        true_err_msg = re.escape('`patience` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=-3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative012(self):
        true_err_msg = re.escape('`max_seq_length` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE,
                finetune_elmo=True, batch_size=32, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10,
                patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative013(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length='512',
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative014(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=-3,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative015(self):
        true_err_msg = re.escape('`validation_fraction` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative016(self):
        true_err_msg = re.escape('`validation_fraction` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction='0.1', max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative017(self):
        true_err_msg = '`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, but ' \
                       '{0} is not positive.'.format(-0.1)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=-0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative018(self):
        true_err_msg = '`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, but ' \
                       '{0} is not less than 1.0.'.format(1.1)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=1.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative019(self):
        true_err_msg = re.escape('`gpu_memory_frac` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, verbose=False, random_seed=42
            )

    def test_check_params_negative020(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac='1.0',
                verbose=False, random_seed=42
            )

    def test_check_params_negative021(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(-1.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=-1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative022(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(1.3))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.3,
                verbose=False, random_seed=42
            )

    def test_check_params_negative023(self):
        true_err_msg = re.escape('`lr` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42
            )

    def test_check_params_negative024(self):
        true_err_msg = re.escape('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr='1e-3', l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative025(self):
        true_err_msg = re.escape('`lr` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=0.0, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative026(self):
        true_err_msg = re.escape('`lr` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42
            )

    def test_check_params_negative027(self):
        true_err_msg = re.escape('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr='1e-3', l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative028(self):
        true_err_msg = re.escape('`lr` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=0.0, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative029(self):
        true_err_msg = re.escape('`l2_reg` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False,
                random_seed=42
            )

    def test_check_params_negative030(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg='1e-4', validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative031(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected a non-negative floating-point value, but {0} is '
                                 'negative.'.format(-2.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=-2.0, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative032(self):
        true_err_msg = re.escape('`finetune_elmo` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4,
                validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative033(self):
        true_err_msg = re.escape('`finetune_elmo` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo='True', batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose=False, random_seed=42
            )

    def test_check_params_negative034(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                random_seed=42
            )

    def test_check_params_negative035(self):
        true_err_msg = re.escape('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_params(
                elmo_hub_module_handle=self.ELMO_HUB_MODULE, finetune_elmo=True, batch_size=32, max_seq_length=512,
                lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3, gpu_memory_frac=1.0,
                verbose='False', random_seed=42
            )

    def test_check_X_positive(self):
        X = ['abc', 'defgh', '4wdffg']
        ELMo_NER.check_X(X, 'X_train')
        self.assertTrue(True)

    def test_check_X_negative01(self):
        X = {'abc', 'defgh', '4wdffg'}
        true_err_msg = re.escape('`X_train` is wrong, because it is not list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_X(X, 'X_train')

    def test_check_X_negative02(self):
        X = np.random.uniform(-1.0, 1.0, (10, 2))
        true_err_msg = re.escape('`X_train` is wrong, because it is not 1-D list!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_X(X, 'X_train')

    def test_check_X_negative03(self):
        X = ['abc', 23, '4wdffg']
        true_err_msg = re.escape('Item 1 of `X_train` is wrong, because it is not string-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            ELMo_NER.check_X(X, 'X_train')

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
        self.assertEqual(true_classes_list, ELMo_NER.check_Xy(X, 'X_train', y, 'y_train'))

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

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
            ELMo_NER.check_Xy(X, 'X_train', y, 'y_train')

    def test_calculate_bounds_of_tokens_positive01(self):
        source_text = 'Совершенно новую технологию перекачки российской водки за рубеж начали использовать ' \
                      'контрабандисты.'
        tokenized_text = ['Совершенно', 'новую', 'технологию', 'перекачки', 'российской', 'водки', 'за', 'рубеж',
                          'начали', 'использовать', 'контрабандисты', '.']
        true_bounds = [(0, 10), (11, 16), (17, 27), (28, 37), (38, 48), (49, 54), (55, 57), (58, 63), (64, 70),
                       (71, 83), (84, 98), (98, 99)]
        self.assertEqual(true_bounds, ELMo_NER.calculate_bounds_of_tokens(source_text, tokenized_text))

    def test_calculate_bounds_of_tokens_positive02(self):
        source_text = 'Один из последних представителей клады, тираннозавр (Tyrannosaurus rex), живший 66–67 ' \
                      'миллионов лет назад, был одним из крупнейших когда-либо живших сухопутных хищников'
        tokenized_text = ['Один', 'из', 'последних', 'представителей', 'клады', ',', 'тираннозавр', '(',
                          'Tyrannosaurus', 'rex', ')', ',', 'живший', '66', '–', '67', 'миллионов', 'лет', 'назад', ',',
                          'был', 'одним', 'из', 'крупнейших', 'когда', '-', 'либо', 'живших', 'сухопутных', 'хищников']
        true_bounds = [(0, 4), (5, 7), (8, 17), (18, 32), (33, 38), (38, 39), (40, 51), (52, 53), (53, 66), (67, 70),
                       (70, 71), (71, 72), (73, 79), (80, 82), (82, 83), (83, 85), (86, 95), (96, 99), (100, 105),
                       (105, 106), (107, 110), (111, 116), (117, 119), (120, 130), (131, 136), (136, 137), (137, 141),
                       (142, 148), (149, 159), (160, 168)]
        self.assertEqual(true_bounds, ELMo_NER.calculate_bounds_of_tokens(source_text, tokenized_text))

    def test_detect_token_labels_positive01(self):
        source_text = 'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози.'
        tokenized_text = ['Барак', 'Обама', 'принимает', 'в', 'Белом', 'доме', 'своего',
                          'французского', 'коллегу', 'Николя', 'Саркози', '.']
        token_bounds = ELMo_NER.calculate_bounds_of_tokens(source_text, tokenized_text)
        indices_of_named_entities = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 1}
        y_true = np.array([2, 1, 0, 0, 4, 3, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], dtype=np.int32)
        y_pred = ELMo_NER.detect_token_labels(token_bounds, indices_of_named_entities, label_IDs, 16)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_detect_token_labels_positive02(self):
        source_text = 'С 1876 г Павлов ассистирует профессору К. Н. Устимовичу в Медико-хирургической академии и ' \
                      'параллельно изучает физиологию кровообращения.'
        tokenized_text = ['С', '1876', 'г', 'Павлов', 'ассистирует', 'профессору', 'К', '.', 'Н', '.', 'Устимовичу',
                          'в', 'Медико', '-', 'хирургической', 'академии', 'и', 'параллельно', 'изучает', 'физиологию',
                          'кровообращения', '.']
        token_bounds = ELMo_NER.calculate_bounds_of_tokens(source_text, tokenized_text)
        indices_of_named_entities = np.array(
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 3, 4: 2, 5: 4}
        y_true = np.array(
            [0, 2, 1, 4, 0, 6, 4, 3, 3, 3, 3, 0, 8, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = ELMo_NER.detect_token_labels(token_bounds, indices_of_named_entities, label_IDs, 32)
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
        indices, labels_to_classes = ELMo_NER.calculate_indices_of_named_entities(source_text, classes_list,
                                                                                  named_entities)
        self.assertIsInstance(indices, np.ndarray)
        self.assertIsInstance(labels_to_classes, dict)
        self.assertEqual(true_indices.shape, indices.shape)
        self.assertEqual(true_indices.tolist(), indices.tolist())
        self.assertEqual(set(true_labels_to_classes.keys()), set(labels_to_classes.keys()))
        for label_ID in true_labels_to_classes:
            self.assertEqual(true_labels_to_classes[label_ID], labels_to_classes[label_ID])

    def test_tokenize_by_character_groups(self):
        source_text = 'Один из последних представителей клады, тираннозавр (Tyrannosaurus rex), живший 66–67 ' \
                      'миллионов лет назад, был одним из крупнейших когда-либо живших сухопутных хищников'
        true_tokens = ['Один', 'из', 'последних', 'представителей', 'клады', ',', 'тираннозавр', '(', 'Tyrannosaurus',
                       'rex', ')', ',', 'живший', '66', '–', '67', 'миллионов', 'лет', 'назад', ',', 'был', 'одним',
                       'из', 'крупнейших', 'когда', '-', 'либо', 'живших', 'сухопутных', 'хищников']
        self.assertEqual(true_tokens, ELMo_NER.tokenize_by_character_groups(source_text))

    def test_calc_similarity_between_entities_positive01(self):
        gold_entity = (3, 9)
        predicted_entity = (3, 9)
        true_similarity = 1.0
        true_tp = 6
        true_fp = 0
        true_fn = 0
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
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
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
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
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
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
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
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
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
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
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
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
        similarity, tp, fp, fn = ELMo_NER.calc_similarity_between_entities(gold_entity, predicted_entity)
        self.assertAlmostEqual(true_similarity, similarity, places=4)
        self.assertEqual(true_tp, tp)
        self.assertEqual(true_fp, fp)
        self.assertEqual(true_fn, fn)

    def test_load_dataset_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        file_name = os.path.join(base_dir, 'dataset_with_paragraphs.json')
        X_true = [
            'Александр Вертинский. «Я не знаю, зачем и кому это нужно…»',
            '21 марта 1889 года родился главный русский шансонье XX века, печальный Пьеро, вписавший свою судьбу в '
            'историю отечественной культуры',
            'Жизнь с самого начала оставляла для Александра Вертинского слишком много вопросов без ответов. Слишком '
            'много «пустого» пространства. И он научился заполнять его вымыслом. Создал собственный театр с безумным '
            'множеством персонажей, каждый из которых — от сироток-калек и безымянных кокаинеточек до гениальных '
            'скрипачей и кинодив — был им самим.',
            'Трехкомнатная квартира на последнем этаже дома на углу Тверской и Козицкого переулка в Москве и сегодня '
            'выглядит так, словно ее хозяин вот-вот вернется. В просторном кабинете все те же большие книжные шкафы, '
            'все тот же гигантский письменный стол с наполеоновским вензелем и бюстом Вольтера.',
            'Сейчас в кабинете все чаще бывает лишь вдова Вертинского. Вновь и вновь перечитывает его письма, '
            'рукописи. Он смотрит на нее с фотографий, развешанных на стенах, расставленных на столе, и словно '
            'возвращает в те пятнадцать лет неизбывного счастья, когда по квартире витает запах табака и лаванды, дом '
            'полон гостей и шумные застолья длятся допоздна. И все это — будто здесь и сейчас. Нет, время не '
            'остановилось, оно сомкнуло объятия, чтобы вновь и вновь перечитывать эту странную, загадочную судьбу.',
            'Считается, что свой голос Георгий Иванов обрёл в эмиграции и благодаря эмиграции. Мол, утрата родины '
            'стала для него тем «простым человеческим горем», которого так не хватало по форме безупречным его стихам, '
            'чтобы они наполнились содержанием. На самом деле это не совсем так, потому что точка сборки Георгия '
            'Иванова была смещена ещё в Петербурге.',
            'Георгий Иванов. На грани музыки и сна',
            'Первое детское воспоминание Вертинского — о смерти матери. Трехлетний Саша сидит на горшке и выковыривает '
            'глаза у плюшевого медвежонка. Горничная Лизка отрывает мальчика от увлекательного занятия: «Вставай, твоя '
            'мама умерла!» Мать лежит в серебристом гробу на столе, тело ее скрывают цветы; у изголовья стоят '
            'серебряные подсвечники и маленькая табуретка. В руке Саша сжимает шоколадку, он бросается к матери, чтобы '
            'угостить. Но мать не раскрывает рта…',
            'Через два года от чахотки умер отец. Однажды ранней весной его нашли без чувств на могиле супруги. '
            'Оправиться от болезни он уже не смог. Когда кровь хлынула горлом, рядом с ним была только десятилетняя '
            'дочь Надя, не знавшая, как помочь. Обессиленный отец упал на подушку и захлебнулся кровью.',
            'Старшая сестра матери забрала Надю к себе в Ковно. Саша остался жить в Киеве с другой сестрой матери, '
            'которая уверила мальчика в том, что его сестра умерла. То же самое было сказано Наде о брате. Спустя годы '
            'Александр случайно обнаружит упоминание о Н. Н. Вертинской в журнале «Театр и искусство», напишет ей, и '
            'выяснится, что это его сестра. Во время Первой мировой Вертинскому сообщат, что Надя покончила с собой. '
            'Только после смерти Вертинского его вдова выяснит, что Надежда Николаевна живет в Ленинграде.',
            'Смерть причудливо и неотвратимо вписалась в его жизнь. Смерть была тем миром, где кончались тщета '
            'мальчика Мая и тревоги Безноженьки и наступал долгожданный покой.',
            'Александр Вертинский появился на свет «незаконнорожденным». Родственники отца и матери не одобряли союз '
            'Николая Вертинского с Евгенией Скалацкой (Сколацкой) даже тогда, когда родились Надя и Саша. Евгения '
            'Степановна происходила из дворянского рода, а Николай Петрович был присяжным поверенным. Первая жена отца '
            'по настоянию родственников Николая Вертинского не давала ему развода. Так что пришлось усыновить '
            'собственных детей.',
            'Жизнь с самого начала оставляла для Александра Вертинского слишком много вопросов без ответов. Слишком '
            'много «пустого» пространства. И он научился заполнять его вымыслом. Создал собственный театр с безумным '
            'множеством персонажей, каждый из которых — от сироток-калек и безымянных кокаинеточек до гениальных '
            'скрипачей и кинодив — был им самим.',
            'Театр стал маниакальной страстью Вертинского еще с гимназических лет. Он любыми способами проникал на '
            'спектакли, оперы, концерты, выступал в любительских постановках в контрактовом зале на киевском Подоле и '
            'подвизался статистом в Соловцовском театре — разумеется, бесплатно. А чтобы не умереть с голоду, брался '
            'за любую работу — пописывал рецензии на выступления гастролеров, служил корректором в типографии, '
            'нанимался помощником бухгалтера в гостиницу, продавал открытки, грузил арбузы на барках и даже '
            'подворовывал у двоюродной сестры безделушки, чтобы сбыть их на толкучке.',
            'С армией Колчака бежала из Владивостока семья цыган Димитриевичей, на пароходах генерала Врангеля '
            'спасались Александр Вертинский и Надежда Плевицкая, уходили куда угодно, лишь бы подальше от Советов, '
            'многие звёзды и звёздочки... Да, в первой эмиграции оказалось немало творческих личностей, работавших в '
            'интересующем нас жанре русской песни, но даже самые яркие их имена блекнут рядом со сверкающей снежной '
            'шапкой Монблана в лице Фёдора Ивановича Шаляпина.',
            'Живой бог русской музыки',
            'В 1911–1912 годах журналы «Киевская неделя» и «Лукоморье» опубликовали первые рассказы Вертинского: '
            '«Красные бабочки» и «Моя невеста» — декадентские, но с бунинской интонацией. «Красные бабочки» — о '
            'мальчике-сироте, случайно погубившем красных бабочек, вышитых на черном платье. Мальчик наказан суровой '
            'теткой, но бабочки являются ему во сне, чтобы отомстить за погибших сестер. «Моя невеста» — о сумасшедшей '
            'бездомной, читающей стихи на эстраде опустевшего осеннего парка. Эта «светлая малютка-невеста» при '
            'ближайшем рассмотрении оказывается «маленьким уродливым существом» с «длинным, острым, серо-зеленого '
            'цвета лицом», «черно-синими припухшими губами», «без бровей, без ресниц, с глубоко вдавленными в череп '
            'глазами».',
            'Свободное от литературных посиделок и работы время Вертинский коротал с киевской богемной молодежью в '
            'подвальном кабачке, закусывая дешевое вино дешевым сыром. В приобретенном на толкучке подержанном фраке, '
            'всегда с живым цветком в петлице, всегда презрительный и надменный, он сыпал заранее продуманными '
            'афоризмами и производил на окружающих впечатление большого оригинала. Но прекрасно понимал, что вечно так '
            'продолжаться не может.',
            'Скопив 25 рублей и подыскав компаньона с театральным гардеробчиком (без собственных костюмов в театрах '
            'тогда статистов не брали), Вертинский подался в Москву.',
            'Здесь он играл небольшие роли в любительских студиях, поступил в театр миниатюр Марьи Арцыбушевой, где '
            'служил за котлеты и борщ, соглашался на любые роли в кино, показывался во МХАТе — но из-за своего '
            'грассирующего «р» был отвергнут Станиславским.',
            'А внутри бурлило и клокотало, требовало выхода и не находило его. Слишком много вокруг было никому '
            'неизвестных талантов и знаменитых бездарностей. Столицы захлестнула эпидемия увлечения кокаином. Его '
            'покупали сначала в аптеках, затем с рук, носили в пудреницах и портсигарах, щедро одалживали и '
            'одалживались. Однажды выглянув из выходившего на крышу окна мансарды, которую Вертинский снимал, он '
            'обнаружил, что весь скат усеян пустыми коричневыми бутылочками из-под кокаина.',
            'Вертинский отправился к психиатру, профессору Баженову, и, подойдя к трамвайной остановке, увидел, как '
            'Пушкин сошел со своего пьедестала, оставив на нем четкий след. Александр Сергеевич сел вместе с '
            'Вертинским в трамвай и достал большой старинный медный пятак — для оплаты.',
            'Справиться с пристрастием к кокаину Вертинскому помогла война. Под именем Брат Пьеро он записался в '
            'санитарный поезд, курсировавший от Москвы к фронту и обратно. Почти два года Вертинский перевязывал '
            'раненых, читал им письма от родных, пел и даже, по его уверению, оперировал.',
            'В 1915 году Вертинский вернулся в театр миниатюр Арцыбушевой с собственным номером — «Ариетки Пьеро». На '
            'фоне черного занавеса в лунном луче прожектора на сцене появлялся высокий молодой человек. На его густо '
            'покрытом белилами лице резко выделялись ярко-красный рот, обведенные тушью большие глаза и печально '
            'вздернутые нарисованные брови. После вступления рояля этот странный юноша взмахивал руками и тихо '
            'начинал:',
            'Я люблю Вас, моя сегоглазочка, Золотая ошибка моя! Вы — вечегняя жуткая сказочка, Вы — цветок из кагтины '
            'Гойя.',
            'После бесконечных ямщиков и соловьев, аллей и ночей, дышащих сладострастьем, с одной стороны, а с другой '
            'с другой — на фоне бравад футуристов, претенциозных поэз Игоря Северянина и одесской шансоньетки Изы '
            'Кремер с ее занзибарами-кларами, — печальный Пьеро Вертинского стал сенсацией. Ему удалось невозможное: '
            'вписать богемную экзотику — всех этих маленьких креольчиков, смуглых принцев с Антильских островов, '
            'китайчат Ли, лиловых негров — в живописный ландшафт одинокой и беззащитной души; превратить ироничную '
            'игру культурными символами в откровение глубокой печали.',
            'Так певец без выдающихся вокальных данных, композитор, не знавший нотной грамоты, актер с дефектом дикции '
            'стал всероссийским кумиром. Издательство «Прогрессивные новости» Б. Андржеевского огромными тиражами '
            'выпускало «Песенки Вертинского», которые впечатлительные курсистки развозили по всей стране.',
            'Начались гастроли и бенефисы, от восторженной и возмущенной публики нередко приходилось спасаться через '
            'черный ход. Посыпались приглашения в кино. Популярность Вертинского была столь велика, что в феврале 1917 '
            'года Александра Керенского называли «печальным Пьеро российской революции».',
            'Как и подавляющее большинство представителей русской интеллигенции, Вертинский связывал с Февральской '
            'революцией опьяняющие надежды на обновление и очищение. Октябрьский переворот заставил протрезветь. Под '
            'впечатлением гибели московских юнкеров, убитых большевиками, Вертинский написал знаменитых «Юнкеров»:',
            'Я не знаю, зачем и кому это нужно, Кто послал их на смерть недрожавшей рукой, Только так беспощадно, так '
            'зло и ненужно Опустили их в вечный покой.',
            'Песня стала настоящим белогвардейским гимном — с нею шли в бой и умирали русские офицеры и юнкера. '
            'Существует легенда, что Вертинского вызывали в ЧК для дачи объяснений по поводу контрреволюционной песни. '
            'Артист возмутился: «Но вы же не можете запретить мне их жалеть!» И в ответ услышал: «Дышать запретим, '
            'если потребуется».',
            'Как и многие эпизоды из жизни Вертинского, допрос в ЧК не имеет документальных подтверждений. Тем не '
            'менее факт остается фактом: вслед за отступающей белой армией, как и многие российские артисты, '
            'Вертинский подался на юг, где все еще верили в счастливую развязку и мучились тяжелым предчувствием, что '
            'ее никогда не будет.',
            'В 1920 году на пароходе «Великий князь Александр Михайлович», увозящем барона Врангеля, Вертинский '
            'покинул Россию, отправившись в добровольное изгнание на 23 года.',
            'Его одиссея началась с Константинополя, где он пел разноязыким эмигрантам цыганские романсы и раздобыл '
            'греческий паспорт на имя Александра Вертидиса. Закружилась круговерть авантюр, лиц, городов, стран. '
            'Румыния, Польша, Германия, Австрия, Венгрия, Палестина, Египет, Ливия, Франция, США… Выступления в '
            'ресторанах и кабаках — между горячим и десертом; в мюзик-холлах и фешенебельных отелях — для королей '
            'Густава Шведского, Альфонса Испанского, принца Уэльского, для Вандербильтов и Ротшильдов.',
            'В Бессарабии его арестовали по обвинению в просоветской пропаганде песней «В степи молдаванской» — в '
            'особенности строками «О, как сладко, как больно сквозь слезы / Хоть взглянуть на родную страну…» '
            'Естественно, в деятельности Вертинского усмотрели происки НКВД. С тех пор слава чекистского агента '
            'бросает тень на его репутацию по сей день — как будто агент НКВД не может быть великим артистом…',
            'Все двадцать с лишним лет, где бы Вертинский ни выступал, он пел только на русском (исключение делал лишь '
            'для любимой Франции, где исполнял несколько своих песенок по-французски). Его основной аудиторией, '
            'конечно же, была русская эмиграция, для которой печальный Пьеро являлся не просто символом утраченной '
            'России, но, по выражению Шаляпина, «сказителем земли русской».',
            'Уже с начала 1920-х Вертинский просил разрешения вернуться — через советское консульство, через Анатолия '
            'Луначарского, возглавившего советскую делегацию в Берлине, — но неизменно получал отказ.',
            'В конце 1935 года он приехал в Китай — в Шанхае и Харбине была довольно обширная русская община. В Шанхае '
            'артист дал двадцать аншлаговых концертов (даже Шаляпину здесь сумели организовать только два '
            'выступления), однако бесконечно петь для одной и той же аудитории невозможно, и Вертинский намеревался '
            'через какое-то время вернуться в Европу. Но в 1937 году его вдруг пригласили в СССР — без всяких просьб '
            'со стороны артиста. Вертинский остался в Китае, ожидая, когда организуют возвращение. Он ждал пять лет.',
            'Что побудило Сталина позвать Вертинского? Рассказывали, что генералиссимус любил слушать ариетки Брата '
            'Пьеро в часы отдыха — особенно песню «В синем и далеком океане». Легенда приписывает также Сталину '
            'известную фразу «Дадим артисту Вертинскому спокойно дожить на Родине», произнесенную после того, как '
            '«отец всех народов» лично вычеркнул артиста из ждановского постановления, громившего Дмитрия Шостаковича '
            'и Сергея Прокофьева. Нравился Сталину Вертинский или нет, несомненно одно — возвращение «соловья '
            'белоэмиграции», мировой знаменитости было идеологически выгодно советскому режиму, тем более в 1943 году, '
            'когда открылся союзный фронт и в стране бродили оттепельные настроения.',
            'Вертинский же всегда и всем говорил о том, что возвращается, чтобы «рассказать о страданиях эмиграции» и '
            '«помирить Родину с ней». «Шанхайская Тэффи» Наталия Ильина не преминула по этому поводу съязвить в '
            'автобиографическом романе «Возвращение». Ее Джордж Эрмин (Георгий Еремин), подозрительно похожий на '
            'Вертинского, прочитав Конституцию СССР, перекрестился и изрек: «Я подумал, что же это — Китеж, '
            'воскресающий без нас!»',
            'Ранним утром 4 ноября 1943 года на пароходе «Дайрен-Мару» Вертинский покинул Шанхай. С ним были его '
            'двадцатилетняя жена Лидия и ее мать, на руках он держал трехмесячную дочь Марианну. Необходимость '
            'содержать семью была не самой последней причиной переезда в СССР. Шла война, зверствовала инфляция, '
            'иностранные конторы в Китае закрывались, русские эмигранты спасались от японской оккупации. Выступать '
            'становилось все труднее. Вертинский пускался в рискованные финансовые авантюры, не имевшие успеха. Его '
            'самой удачной коммерческой операцией была закупка пяти бутылей водки накануне рождения ребенка. Продав '
            'их после повышения цен, Вертинский оплатил счета за услуги роддома.',
            'Первым советским городом на их пути стала Чита. Стоял жуткий мороз, семью Вертинского поселили в '
            'гостинице, где практически не топили, а по стенам ползали клопы. А в местной филармонии артиста уже '
            'поджидала телеграмма из Москвы с распоряжением дать в Чите несколько концертов. Родина встречала блудного '
            'сына.',
            'О его возвращении ходили анекдоты. В одном из них рассказывалось, как Вертинский, приехав в СССР, выходит '
            'из вагона с двумя чемоданами, ставит их, целует землю и смотрит вокруг: «Не узнаю тебя, Россия!» '
            'Обернувшись, обнаруживает, что чемоданов нет. «Узнаю тебя, Россия!» — восклицает артист. В другом '
            'повествовалось о приеме, устроенном в честь Вертинского «пролетарским графом» Алексеем Николаевичем '
            'Толстым. Гости долго томятся, ожидая, когда их пригласят к столу. Кто-то из присутствующих, оглядев '
            'собравшееся общество — граф Толстой, граф Игнатьев, митрополит Николай Крутицкий, Александр Вертинский, —'
            ' спрашивает: «Кого ждем?» Остроумец-куплетист Смирнов-Сокольский отвечает: «Государя!»',
            'Первой советской киноролью Вертинского стал кардинал Бирнч в фильме Михаила Калатозова «Заговор '
            'обреченных». Актер сыграл изысканного, сладкоречивого патриция со следами былого донжуанства. Так и '
            'должен выглядеть настоящий враг советского режима — образованный, воспитанный, обвораживающий своим '
            'лоском. Только такие и могут строить заговоры и вынашивать планы государственного переворота. Сталинская '
            'премия за роль кардинала свидетельствовала о высочайшем одобрении этой трактовки.',
            'Такого же двуликого Януса Вертинский исполнил в помпезном фильме Сергея Юткевича «Великий воин '
            'Скандербег». Возможно, он играл бы маскирующихся иродов и дальше, если бы Исидор Анненский не предложил '
            'ему роль князя в экранизации чеховской «Анны на шее». Одним своим появлением на экране Вертинский, этот '
            'обломок царской России, воскрешал шик дворянских собраний и балов при дворе.',
            'Положение «советского артиста» Вертинского было довольно странным. С одной стороны, явное благоволение '
            'властей: его с семьей поселили в «Метрополе», затем выделили квартиру, наградили высшей государственной '
            'премией. Правда, семья в течение трех лет обитала в «Метрополе» не от хорошей жизни. Съехать было просто '
            'некуда, потому что выделенная квартира находилась на первом этаже двухэтажного дома на Хорошевском шоссе. '
            'Артист опасался поселяться в ней и с помощью сложных маневров обменял ее на квартиру на улице Горького, '
            'которая была в таком жутком состоянии, что нуждалась в капитальном ремонте. Опасения Вертинского, как '
            'выяснилось позже, были не напрасны — квартира на Хорошевском шоссе подверглась налету знаменитой «Черной '
            'кошки».',
            'С другой стороны, из ста с лишним песен к исполнению было разрешено не более тридцати (авторство текстов '
            'Георгия Иванова и Николая Гумилева Вертинскому пришлось приписать себе), единственная прижизненная '
            'пластинка вышла в 1944 году, о концертах — ни строчки в прессе. «Я существую на правах публичного дома, —'
            ' горько шутил Вертинский, — все ходят, но в обществе говорить об этом не принято».',
            'Из эмиграции Вертинский вернулся практически с пустыми карманами, вскоре родилась вторая дочь, Настя. '
            'Гастрольбюро обеспечило артисту по 20–25 концертов в месяц по всей стране от Средней Азии до Дальнего '
            'Востока — в нетопленных, неприспособленных для выступлений залах с расстроенными роялями и пьяной '
            'публикой. Но концертная жизнь в европейских кабаках приучила его работать в любых условиях.',
            'Платили Вертинскому по самому низкому тарифу, поскольку у него не было никаких званий. За концерт артист '
            'получал около 800 рублей, при этом его выступления всегда проходили при аншлагах и собирали десятки тысяч '
            'рублей. Приходилось соглашаться на все, давать левые концерты, выкручиваться, объясняться… Вместе с '
            'аккомпаниатором Михаилом Брохесом он вдоль и поперек исколесил всю страну по нескольку раз, дав около '
            'трех тысяч концертов. Написал два десятка стихов, работал над мемуарами, которые не успел закончить. 14 '
            'лет на Родине превратили бодрого, моложавого мужчину в глубокого старика.',
            'Он не хотел умереть дома, не желал, чтобы родные видели «кухню смерти». 21 мая 1957 года Вертинский '
            'готовился к концерту в Ленинграде, был сдержан и немногословен. Он находился в своем 208-м номере '
            '«Астории», когда начался сердечный приступ. Лекарства под рукой не оказалось. Как выяснилось позже — оно '
            'бы уже не помогло. При вскрытии сосуды рассыпались, как хрупкое стекло',
            'Назначен куратор строительства российской Кремниевой долины',
            'Дмитрий Медведев доверил пост руководителя иннограда миллиардеру Виктору Вексельбергу.',
            'Всё меньше остаётся нерешённых вопросов, касающихся возведения в России уникального Центра по разработке '
            'и коммерциализации новых технологий. Власти уже не только выбрали площадку для строительства '
            'отечественной Кремниевой долины в подмосковном Сколково, а также частично одобрили концепцию наукограда, '
            'но и определили куратора большой инновационной стройки. «Были проведены определённые консультации по '
            'поводу того, кто конкретно мог бы осуществлять такого рода работу. Мною принято решение, что российскую '
            'часть этой координирующей структуры, которую мы создадим, возглавит Виктор Феликсович Вексельберг», — '
            'цитирует «Взгляд» Дмитрия Медведева.',
            'Исходя из заявления президента, понятно, что у проекта будут не только российские инвесторы, но и '
            'иностранные партнёры, в числе которых, по словам главы государства, будут и представители иностранных '
            'научных кругов. Именно на базе взаимодействия науки и бизнеса должен появиться и работать инноград. «Всё '
            'это затеяли не ради того, чтобы построить определённое количество коттеджей или же создать там нормальные '
            'производственные условия, лаборатории. Это всё важно, но это всё инфраструктура. Самое главное, чтобы '
            'там появились люди. Для того чтобы люди появились, должна быть внятная система управления. Эта система '
            'управления зависит от нас. Я думаю, что с учётом масштабности этого проекта, а с другой стороны, того, '
            'что в реализации этого проекта должны быть заинтересованы не только государственные структуры, но, '
            'прежде всего, российский бизнес, я считаю, что координацией российский бизнес и мог бы заняться», — '
            'заявил Дмитрий Медведев.',
            'Это выступление президента вполне объясняет выбор руководителя проекта. Виктор Вексельберг — бизнесмен с '
            '30-летним стажем, капитал которого оценивается в 6,4 млрд долларов. Вексельберг является главой правления '
            'ОАО «Тюменская нефтяная компания» (ТНК) и президентом ЗАО «Ренова». Именно он является владельцем '
            'значительной части российского титана и алюминиевого бизнеса.',
            'О том, почему площадкой для строительства Кремниевой долины выбрано Подмосковье, читайте в статье '
            'Частного корреспондента «Сколково назначили Кремниевой долиной»'
        ]
        y_true = [
            {
                "PERSON": [(0, 20)]
            },
            {
                "PERSON": [(71, 76)]
            },
            {
                "PERSON": [(36, 58)]
            },
            {
                "LOCATION": [(55, 63), (66, 84), (87, 93)],
                "PERSON": [(281, 289)]
            },
            {
                "PERSON": [(45, 56)]
            },
            {
                "LOCATION": [(334, 344)],
                "PERSON": [(26, 40), (299, 314)]
            },
            {
                "PERSON": [(0, 14)]
            },
            {
                "PERSON": [(28, 39), (70, 74), (146, 151), (362, 366)]
            },
            {
                "PERSON": [(207, 211)]
            },
            {
                "PERSON": [(30, 34), (51, 55), (182, 186), (208, 217), (250, 266), (367, 378), (392, 396), (436, 447),
                           (471, 489)],
                "LOCATION": [(44, 49), (71, 76), (498, 508)],
                "ORG": [(269, 295)]
            },
            {
                "PERSON": [(107, 110), (121, 132)]
            },
            {
                "PERSON": [(0, 20), (104, 123), (126, 144), (146, 155), (184, 188), (191, 195), (197, 215), (251, 267),
                           (338, 357)]
            },
            {
                "PERSON": [(36, 58)]
            },
            {
                "PERSON": [(33, 44)],
                "LOCATION": [(168, 185), (189, 197), (198, 204)],
                "ORG": [(230, 249)]
            },
            {
                "PERSON": [(9, 16), (89, 97), (108, 128), (131, 148), (430, 455)],
                "LOCATION": [(27, 39), (191, 198), (414, 422)],
                "ORG": [(2, 16)]
            },
            dict(),
            {
                "PERSON": [(87, 98)],
                "ORG": [(18, 56)]
            },
            {
                "LOCATION": [(72, 80)],
                "PERSON": [(51, 61)]
            },
            {
                "LOCATION": [(151, 157)],
                "PERSON": [(130, 140)]
            },
            {
                "PERSON": [(80, 97), (233, 246)],
                "ORG": [(65, 97), (177, 182)]
            },
            {
                "PERSON": [(373, 383)]
            },
            {
                "PERSON": [(0, 10), (46, 54), (103, 109), (166, 185), (199, 209)]
            },
            {
                "LOCATION": [(135, 141)],
                "PERSON": [(36, 47), (79, 84), (177, 187)]
            },
            {
                "PERSON": [(12, 22), (49, 60), (94, 99)],
                "ORG": [(34, 60)]
            },
            {
                "PERSON": [(105, 109)]
            },
            {
                "LOCATION": [(389, 408)],
                "PERSON": [(162, 178), (202, 212), (251, 256), (257, 268)]
            },
            {
                "PERSON": [(171, 187), (226, 237)],
                "ORG": [(134, 169)]
            },
            {
                "PERSON": [(160, 171), (215, 236), (257, 262)]
            },
            {
                "PERSON": [(68, 78), (267, 277)]
            },
            dict(),
            {
                "PERSON": [(123, 134)],
                "ORG": [(146, 148)]
            },
            {
                "PERSON": [(30, 41), (197, 207)],
                "ORG": [(52, 54)]
            },
            {
                "LOCATION": [(107, 113)],
                "PERSON": [(39, 59), (78, 86), (88, 98)]
            },
            {
                "LOCATION": [(23, 38), (203, 210), (212, 218), (220, 228), (230, 237), (239, 246), (248, 257),
                             (259, 265), (267, 272), (274, 281), (283, 286)],
                "PERSON": [(128, 148), (403, 420), (422, 441), (450, 459)]
            },
            {
                "PERSON": [(226, 237)],
                "LOCATION": [(2, 12)],
                "ORG": [(256, 260), (357, 361)]
            },
            {
                "LOCATION": [(118, 125), (307, 313)],
                "PERSON": [(34, 44), (263, 268), (332, 340)]
            },
            {
                "PERSON": [(20, 30), (96, 117)],
                "LOCATION": [(155, 162)],
                "ORG": [(67, 88), (133, 152)]
            },
            {
                "LOCATION": [(31, 36), (41, 47), (50, 57), (99, 105), (335, 341), (381, 385), (447, 452)],
                "PERSON": [(153, 161), (279, 289), (426, 436)]
            },
            {
                "PERSON": [(13, 20), (29, 40), (103, 108), (194, 201), (233, 244), (388, 407), (410, 427), (438, 445),
                           (446, 456)]
            },
            {
                "LOCATION": [(338, 342), (392, 397)],
                "PERSON": [(0, 10), (142, 147), (149, 163), (248, 260), (262, 276), (304, 315)]
            },
            {
                "LOCATION": [(45, 51), (52, 56), (77, 83), (258, 262), (320, 325)],
                "PERSON": [(58, 68), (120, 125), (174, 182), (425, 435), (630, 640)]
            },
            {
                "LOCATION": [(42, 46), (221, 227), (251, 255)],
                "PERSON": [(74, 85)]
            },
            {
                "LOCATION": [(92, 96), (194, 200), (262, 268)],
                "PERSON": [(70, 80), (345, 356), (379, 408), (529, 536), (543, 551), (564, 581), (583, 603), (652, 670)]
            },
            {
                "PERSON": [(27, 38), (53, 58), (68, 86)]
            },
            {
                "LOCATION": [(319, 325)],
                "PERSON": [(20, 25), (26, 36), (65, 80), (95, 105), (169, 185), (239, 243), (286, 296)]
            },
            {
                "PERSON": [(31, 42), (607, 618)],
                "LOCATION": [(137, 146), (260, 269), (399, 416), (506, 520), (673, 690)],
                "ORG": [(722, 734)]
            },
            {
                "PERSON": [(105, 120), (123, 139), (140, 151), (323, 333)]
            },
            {
                "PERSON": [(13, 23), (95, 100)],
                "LOCATION": [(179, 191), (195, 211)],
                "ORG": [(102, 114)]
            },
            {
                "PERSON": [(8, 19), (327, 344)]
            },
            {
                "LOCATION": [(123, 133), (199, 206)],
                "PERSON": [(89, 99)]
            },
            {
                "LOCATION": [(31, 59)]
            },
            {
                "LOCATION": [(43, 52)],
                "PERSON": [(0, 16), (65, 85)]
            },
            {
                "ORG": [(84, 140), (620, 626)],
                "LOCATION": [(65, 71), (212, 229), (232, 253), (291, 301)],
                "PERSON": [(576, 605), (628, 645)]
            },
            {
                "LOCATION": [(153, 164), (290, 298)],
                "PERSON": [(925, 941)]
            },
            {
                "ORG": [(201, 243), (246, 249), (265, 276)],
                "PERSON": [(72, 90), (173, 184)]
            },
            {
                "LOCATION": [(42, 59), (68, 79), (123, 131), (142, 160)],
                "ORG": [(98, 121)]
            }
        ]
        X_loaded, y_loaded = load_dataset(file_name)
        self.assertIsInstance(X_loaded, list)
        self.assertIsInstance(y_loaded, list)
        self.assertEqual(len(X_true), len(X_loaded))
        self.assertEqual(len(y_true), len(y_loaded))
        for sample_idx in range(len(X_true)):
            self.assertEqual(X_true[sample_idx], X_loaded[sample_idx])
            self.assertIsInstance(y_loaded[sample_idx], dict)
            self.assertEqual(set(y_true[sample_idx]), set(y_loaded[sample_idx]))
            for ne_type in y_true[sample_idx]:
                self.assertIsInstance(y_loaded[sample_idx][ne_type], list)
                self.assertEqual(len(y_true[sample_idx][ne_type]), len(y_loaded[sample_idx][ne_type]))
                for entity_idx in range(len(y_true[sample_idx][ne_type])):
                    self.assertEqual(y_true[sample_idx][ne_type][entity_idx], y_loaded[sample_idx][ne_type][entity_idx])

    def test_load_dataset_positive02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        file_name = os.path.join(base_dir, 'dataset_without_paragraphs.json')
        X_true = [
            'Александр Вертинский. «Я не знаю, зачем и кому это нужно…»',
            '21 марта 1889 года родился главный русский шансонье XX века, печальный Пьеро, вписавший свою судьбу в '
            'историю отечественной культуры',
            'Жизнь с самого начала оставляла для Александра Вертинского слишком много вопросов без ответов. Слишком '
            'много «пустого» пространства. И он научился заполнять его вымыслом. Создал собственный театр с безумным '
            'множеством персонажей, каждый из которых — от сироток-калек и безымянных кокаинеточек до гениальных '
            'скрипачей и кинодив — был им самим.',
            'Трехкомнатная квартира на последнем этаже дома на углу Тверской и Козицкого переулка в Москве и сегодня '
            'выглядит так, словно ее хозяин вот-вот вернется. В просторном кабинете все те же большие книжные шкафы, '
            'все тот же гигантский письменный стол с наполеоновским вензелем и бюстом Вольтера.',
            'Сейчас в кабинете все чаще бывает лишь вдова Вертинского. Вновь и вновь перечитывает его письма, '
            'рукописи. Он смотрит на нее с фотографий, развешанных на стенах, расставленных на столе, и словно '
            'возвращает в те пятнадцать лет неизбывного счастья, когда по квартире витает запах табака и лаванды, дом '
            'полон гостей и шумные застолья длятся допоздна. И все это — будто здесь и сейчас. Нет, время не '
            'остановилось, оно сомкнуло объятия, чтобы вновь и вновь перечитывать эту странную, загадочную судьбу.',
            'Считается, что свой голос Георгий Иванов обрёл в эмиграции и благодаря эмиграции. Мол, утрата родины '
            'стала для него тем «простым человеческим горем», которого так не хватало по форме безупречным его стихам, '
            'чтобы они наполнились содержанием. На самом деле это не совсем так, потому что точка сборки Георгия '
            'Иванова была смещена ещё в Петербурге.',
            'Георгий Иванов. На грани музыки и сна',
            'Первое детское воспоминание Вертинского — о смерти матери. Трехлетний Саша сидит на горшке и выковыривает '
            'глаза у плюшевого медвежонка. Горничная Лизка отрывает мальчика от увлекательного занятия: «Вставай, твоя '
            'мама умерла!» Мать лежит в серебристом гробу на столе, тело ее скрывают цветы; у изголовья стоят '
            'серебряные подсвечники и маленькая табуретка. В руке Саша сжимает шоколадку, он бросается к матери, чтобы '
            'угостить. Но мать не раскрывает рта…',
            'Через два года от чахотки умер отец. Однажды ранней весной его нашли без чувств на могиле супруги. '
            'Оправиться от болезни он уже не смог. Когда кровь хлынула горлом, рядом с ним была только десятилетняя '
            'дочь Надя, не знавшая, как помочь. Обессиленный отец упал на подушку и захлебнулся кровью.',
            'Старшая сестра матери забрала Надю к себе в Ковно. Саша остался жить в Киеве с другой сестрой матери, '
            'которая уверила мальчика в том, что его сестра умерла. То же самое было сказано Наде о брате. Спустя годы '
            'Александр случайно обнаружит упоминание о Н. Н. Вертинской в журнале «Театр и искусство», напишет ей, и '
            'выяснится, что это его сестра. Во время Первой мировой Вертинскому сообщат, что Надя покончила с собой. '
            'Только после смерти Вертинского его вдова выяснит, что Надежда Николаевна живет в Ленинграде.',
            'Смерть причудливо и неотвратимо вписалась в его жизнь. Смерть была тем миром, где кончались тщета '
            'мальчика Мая и тревоги Безноженьки и наступал долгожданный покой.',
            'Александр Вертинский появился на свет «незаконнорожденным». Родственники отца и матери не одобряли союз '
            'Николая Вертинского с Евгенией Скалацкой (Сколацкой) даже тогда, когда родились Надя и Саша. Евгения '
            'Степановна происходила из дворянского рода, а Николай Петрович был присяжным поверенным. Первая жена отца '
            'по настоянию родственников Николая Вертинского не давала ему развода. Так что пришлось усыновить '
            'собственных детей.',
            'Жизнь с самого начала оставляла для Александра Вертинского слишком много вопросов без ответов. Слишком '
            'много «пустого» пространства. И он научился заполнять его вымыслом. Создал собственный театр с безумным '
            'множеством персонажей, каждый из которых — от сироток-калек и безымянных кокаинеточек до гениальных '
            'скрипачей и кинодив — был им самим.',
            'Театр стал маниакальной страстью Вертинского еще с гимназических лет. Он любыми способами проникал на '
            'спектакли, оперы, концерты, выступал в любительских постановках в контрактовом зале на киевском Подоле и '
            'подвизался статистом в Соловцовском театре — разумеется, бесплатно. А чтобы не умереть с голоду, брался '
            'за любую работу — пописывал рецензии на выступления гастролеров, служил корректором в типографии, '
            'нанимался помощником бухгалтера в гостиницу, продавал открытки, грузил арбузы на барках и даже '
            'подворовывал у двоюродной сестры безделушки, чтобы сбыть их на толкучке.',
            'С армией Колчака бежала из Владивостока семья цыган Димитриевичей, на пароходах генерала Врангеля '
            'спасались Александр Вертинский и Надежда Плевицкая, уходили куда угодно, лишь бы подальше от Советов, '
            'многие звёзды и звёздочки... Да, в первой эмиграции оказалось немало творческих личностей, работавших в '
            'интересующем нас жанре русской песни, но даже самые яркие их имена блекнут рядом со сверкающей снежной '
            'шапкой Монблана в лице Фёдора Ивановича Шаляпина.',
            'Живой бог русской музыки',
            'В 1911–1912 годах журналы «Киевская неделя» и «Лукоморье» опубликовали первые рассказы Вертинского: '
            '«Красные бабочки» и «Моя невеста» — декадентские, но с бунинской интонацией. «Красные бабочки» — о '
            'мальчике-сироте, случайно погубившем красных бабочек, вышитых на черном платье. Мальчик наказан суровой '
            'теткой, но бабочки являются ему во сне, чтобы отомстить за погибших сестер. «Моя невеста» — о сумасшедшей '
            'бездомной, читающей стихи на эстраде опустевшего осеннего парка. Эта «светлая малютка-невеста» при '
            'ближайшем рассмотрении оказывается «маленьким уродливым существом» с «длинным, острым, серо-зеленого '
            'цвета лицом», «черно-синими припухшими губами», «без бровей, без ресниц, с глубоко вдавленными в череп '
            'глазами».',
            'Свободное от литературных посиделок и работы время Вертинский коротал с киевской богемной молодежью в '
            'подвальном кабачке, закусывая дешевое вино дешевым сыром. В приобретенном на толкучке подержанном фраке, '
            'всегда с живым цветком в петлице, всегда презрительный и надменный, он сыпал заранее продуманными '
            'афоризмами и производил на окружающих впечатление большого оригинала. Но прекрасно понимал, что вечно так '
            'продолжаться не может.',
            'Скопив 25 рублей и подыскав компаньона с театральным гардеробчиком (без собственных костюмов в театрах '
            'тогда статистов не брали), Вертинский подался в Москву.',
            'Здесь он играл небольшие роли в любительских студиях, поступил в театр миниатюр Марьи Арцыбушевой, где '
            'служил за котлеты и борщ, соглашался на любые роли в кино, показывался во МХАТе — но из-за своего '
            'грассирующего «р» был отвергнут Станиславским.',
            'А внутри бурлило и клокотало, требовало выхода и не находило его. Слишком много вокруг было никому '
            'неизвестных талантов и знаменитых бездарностей. Столицы захлестнула эпидемия увлечения кокаином. Его '
            'покупали сначала в аптеках, затем с рук, носили в пудреницах и портсигарах, щедро одалживали и '
            'одалживались. Однажды выглянув из выходившего на крышу окна мансарды, которую Вертинский снимал, он '
            'обнаружил, что весь скат усеян пустыми коричневыми бутылочками из-под кокаина.',
            'Вертинский отправился к психиатру, профессору Баженову, и, подойдя к трамвайной остановке, увидел, как '
            'Пушкин сошел со своего пьедестала, оставив на нем четкий след. Александр Сергеевич сел вместе с '
            'Вертинским в трамвай и достал большой старинный медный пятак — для оплаты.',
            'Справиться с пристрастием к кокаину Вертинскому помогла война. Под именем Брат Пьеро он записался в '
            'санитарный поезд, курсировавший от Москвы к фронту и обратно. Почти два года Вертинский перевязывал '
            'раненых, читал им письма от родных, пел и даже, по его уверению, оперировал.',
            'В 1915 году Вертинский вернулся в театр миниатюр Арцыбушевой с собственным номером — «Ариетки Пьеро». На '
            'фоне черного занавеса в лунном луче прожектора на сцене появлялся высокий молодой человек. На его густо '
            'покрытом белилами лице резко выделялись ярко-красный рот, обведенные тушью большие глаза и печально '
            'вздернутые нарисованные брови. После вступления рояля этот странный юноша взмахивал руками и тихо '
            'начинал:',
            'Я люблю Вас, моя сегоглазочка, Золотая ошибка моя! Вы — вечегняя жуткая сказочка, Вы — цветок из кагтины '
            'Гойя.',
            'После бесконечных ямщиков и соловьев, аллей и ночей, дышащих сладострастьем, с одной стороны, а с другой '
            'с другой — на фоне бравад футуристов, претенциозных поэз Игоря Северянина и одесской шансоньетки Изы '
            'Кремер с ее занзибарами-кларами, — печальный Пьеро Вертинского стал сенсацией. Ему удалось невозможное: '
            'вписать богемную экзотику — всех этих маленьких креольчиков, смуглых принцев с Антильских островов, '
            'китайчат Ли, лиловых негров — в живописный ландшафт одинокой и беззащитной души; превратить ироничную '
            'игру культурными символами в откровение глубокой печали.',
            'Так певец без выдающихся вокальных данных, композитор, не знавший нотной грамоты, актер с дефектом дикции '
            'стал всероссийским кумиром. Издательство «Прогрессивные новости» Б. Андржеевского огромными тиражами '
            'выпускало «Песенки Вертинского», которые впечатлительные курсистки развозили по всей стране.',
            'Начались гастроли и бенефисы, от восторженной и возмущенной публики нередко приходилось спасаться через '
            'черный ход. Посыпались приглашения в кино. Популярность Вертинского была столь велика, что в феврале 1917 '
            'года Александра Керенского называли «печальным Пьеро российской революции».',
            'Как и подавляющее большинство представителей русской интеллигенции, Вертинский связывал с Февральской '
            'революцией опьяняющие надежды на обновление и очищение. Октябрьский переворот заставил протрезветь. Под '
            'впечатлением гибели московских юнкеров, убитых большевиками, Вертинский написал знаменитых «Юнкеров»:',
            'Я не знаю, зачем и кому это нужно, Кто послал их на смерть недрожавшей рукой, Только так беспощадно, так '
            'зло и ненужно Опустили их в вечный покой.',
            'Песня стала настоящим белогвардейским гимном — с нею шли в бой и умирали русские офицеры и юнкера. '
            'Существует легенда, что Вертинского вызывали в ЧК для дачи объяснений по поводу контрреволюционной песни. '
            'Артист возмутился: «Но вы же не можете запретить мне их жалеть!» И в ответ услышал: «Дышать запретим, '
            'если потребуется».',
            'Как и многие эпизоды из жизни Вертинского, допрос в ЧК не имеет документальных подтверждений. Тем не '
            'менее факт остается фактом: вслед за отступающей белой армией, как и многие российские артисты, '
            'Вертинский подался на юг, где все еще верили в счастливую развязку и мучились тяжелым предчувствием, что '
            'ее никогда не будет.',
            'В 1920 году на пароходе «Великий князь Александр Михайлович», увозящем барона Врангеля, Вертинский '
            'покинул Россию, отправившись в добровольное изгнание на 23 года.',
            'Его одиссея началась с Константинополя, где он пел разноязыким эмигрантам цыганские романсы и раздобыл '
            'греческий паспорт на имя Александра Вертидиса. Закружилась круговерть авантюр, лиц, городов, стран. '
            'Румыния, Польша, Германия, Австрия, Венгрия, Палестина, Египет, Ливия, Франция, США… Выступления в '
            'ресторанах и кабаках — между горячим и десертом; в мюзик-холлах и фешенебельных отелях — для королей '
            'Густава Шведского, Альфонса Испанского, принца Уэльского, для Вандербильтов и Ротшильдов.',
            'В Бессарабии его арестовали по обвинению в просоветской пропаганде песней «В степи молдаванской» — в '
            'особенности строками «О, как сладко, как больно сквозь слезы / Хоть взглянуть на родную страну…» '
            'Естественно, в деятельности Вертинского усмотрели происки НКВД. С тех пор слава чекистского агента '
            'бросает тень на его репутацию по сей день — как будто агент НКВД не может быть великим артистом…',
            'Все двадцать с лишним лет, где бы Вертинский ни выступал, он пел только на русском (исключение делал лишь '
            'для любимой Франции, где исполнял несколько своих песенок по-французски). Его основной аудиторией, '
            'конечно же, была русская эмиграция, для которой печальный Пьеро являлся не просто символом утраченной '
            'России, но, по выражению Шаляпина, «сказителем земли русской».',
            'Уже с начала 1920-х Вертинский просил разрешения вернуться — через советское консульство, через Анатолия '
            'Луначарского, возглавившего советскую делегацию в Берлине, — но неизменно получал отказ.',
            'В конце 1935 года он приехал в Китай — в Шанхае и Харбине была довольно обширная русская община. В Шанхае '
            'артист дал двадцать аншлаговых концертов (даже Шаляпину здесь сумели организовать только два '
            'выступления), однако бесконечно петь для одной и той же аудитории невозможно, и Вертинский намеревался '
            'через какое-то время вернуться в Европу. Но в 1937 году его вдруг пригласили в СССР — без всяких просьб '
            'со стороны артиста. Вертинский остался в Китае, ожидая, когда организуют возвращение. Он ждал пять лет.',
            'Что побудило Сталина позвать Вертинского? Рассказывали, что генералиссимус любил слушать ариетки Брата '
            'Пьеро в часы отдыха — особенно песню «В синем и далеком океане». Легенда приписывает также Сталину '
            'известную фразу «Дадим артисту Вертинскому спокойно дожить на Родине», произнесенную после того, как '
            '«отец всех народов» лично вычеркнул артиста из ждановского постановления, громившего Дмитрия Шостаковича '
            'и Сергея Прокофьева. Нравился Сталину Вертинский или нет, несомненно одно — возвращение «соловья '
            'белоэмиграции», мировой знаменитости было идеологически выгодно советскому режиму, тем более в 1943 году, '
            'когда открылся союзный фронт и в стране бродили оттепельные настроения.',
            'Вертинский же всегда и всем говорил о том, что возвращается, чтобы «рассказать о страданиях эмиграции» и '
            '«помирить Родину с ней». «Шанхайская Тэффи» Наталия Ильина не преминула по этому поводу съязвить в '
            'автобиографическом романе «Возвращение». Ее Джордж Эрмин (Георгий Еремин), подозрительно похожий на '
            'Вертинского, прочитав Конституцию СССР, перекрестился и изрек: «Я подумал, что же это — Китеж, '
            'воскресающий без нас!»',
            'Ранним утром 4 ноября 1943 года на пароходе «Дайрен-Мару» Вертинский покинул Шанхай. С ним были его '
            'двадцатилетняя жена Лидия и ее мать, на руках он держал трехмесячную дочь Марианну. Необходимость '
            'содержать семью была не самой последней причиной переезда в СССР. Шла война, зверствовала инфляция, '
            'иностранные конторы в Китае закрывались, русские эмигранты спасались от японской оккупации. Выступать '
            'становилось все труднее. Вертинский пускался в рискованные финансовые авантюры, не имевшие успеха. Его '
            'самой удачной коммерческой операцией была закупка пяти бутылей водки накануне рождения ребенка. Продав '
            'их после повышения цен, Вертинский оплатил счета за услуги роддома.',
            'Первым советским городом на их пути стала Чита. Стоял жуткий мороз, семью Вертинского поселили в '
            'гостинице, где практически не топили, а по стенам ползали клопы. А в местной филармонии артиста уже '
            'поджидала телеграмма из Москвы с распоряжением дать в Чите несколько концертов. Родина встречала блудного '
            'сына.',
            'О его возвращении ходили анекдоты. В одном из них рассказывалось, как Вертинский, приехав в СССР, выходит '
            'из вагона с двумя чемоданами, ставит их, целует землю и смотрит вокруг: «Не узнаю тебя, Россия!» '
            'Обернувшись, обнаруживает, что чемоданов нет. «Узнаю тебя, Россия!» — восклицает артист. В другом '
            'повествовалось о приеме, устроенном в честь Вертинского «пролетарским графом» Алексеем Николаевичем '
            'Толстым. Гости долго томятся, ожидая, когда их пригласят к столу. Кто-то из присутствующих, оглядев '
            'собравшееся общество — граф Толстой, граф Игнатьев, митрополит Николай Крутицкий, Александр Вертинский, —'
            ' спрашивает: «Кого ждем?» Остроумец-куплетист Смирнов-Сокольский отвечает: «Государя!»',
            'Первой советской киноролью Вертинского стал кардинал Бирнч в фильме Михаила Калатозова «Заговор '
            'обреченных». Актер сыграл изысканного, сладкоречивого патриция со следами былого донжуанства. Так и '
            'должен выглядеть настоящий враг советского режима — образованный, воспитанный, обвораживающий своим '
            'лоском. Только такие и могут строить заговоры и вынашивать планы государственного переворота. Сталинская '
            'премия за роль кардинала свидетельствовала о высочайшем одобрении этой трактовки.',
            'Такого же двуликого Януса Вертинский исполнил в помпезном фильме Сергея Юткевича «Великий воин '
            'Скандербег». Возможно, он играл бы маскирующихся иродов и дальше, если бы Исидор Анненский не предложил '
            'ему роль князя в экранизации чеховской «Анны на шее». Одним своим появлением на экране Вертинский, этот '
            'обломок царской России, воскрешал шик дворянских собраний и балов при дворе.',
            'Положение «советского артиста» Вертинского было довольно странным. С одной стороны, явное благоволение '
            'властей: его с семьей поселили в «Метрополе», затем выделили квартиру, наградили высшей государственной '
            'премией. Правда, семья в течение трех лет обитала в «Метрополе» не от хорошей жизни. Съехать было просто '
            'некуда, потому что выделенная квартира находилась на первом этаже двухэтажного дома на Хорошевском шоссе. '
            'Артист опасался поселяться в ней и с помощью сложных маневров обменял ее на квартиру на улице Горького, '
            'которая была в таком жутком состоянии, что нуждалась в капитальном ремонте. Опасения Вертинского, как '
            'выяснилось позже, были не напрасны — квартира на Хорошевском шоссе подверглась налету знаменитой «Черной '
            'кошки».',
            'С другой стороны, из ста с лишним песен к исполнению было разрешено не более тридцати (авторство текстов '
            'Георгия Иванова и Николая Гумилева Вертинскому пришлось приписать себе), единственная прижизненная '
            'пластинка вышла в 1944 году, о концертах — ни строчки в прессе. «Я существую на правах публичного дома, —'
            ' горько шутил Вертинский, — все ходят, но в обществе говорить об этом не принято».',
            'Из эмиграции Вертинский вернулся практически с пустыми карманами, вскоре родилась вторая дочь, Настя. '
            'Гастрольбюро обеспечило артисту по 20–25 концертов в месяц по всей стране от Средней Азии до Дальнего '
            'Востока — в нетопленных, неприспособленных для выступлений залах с расстроенными роялями и пьяной '
            'публикой. Но концертная жизнь в европейских кабаках приучила его работать в любых условиях.',
            'Платили Вертинскому по самому низкому тарифу, поскольку у него не было никаких званий. За концерт артист '
            'получал около 800 рублей, при этом его выступления всегда проходили при аншлагах и собирали десятки тысяч '
            'рублей. Приходилось соглашаться на все, давать левые концерты, выкручиваться, объясняться… Вместе с '
            'аккомпаниатором Михаилом Брохесом он вдоль и поперек исколесил всю страну по нескольку раз, дав около '
            'трех тысяч концертов. Написал два десятка стихов, работал над мемуарами, которые не успел закончить. 14 '
            'лет на Родине превратили бодрого, моложавого мужчину в глубокого старика.',
            'Он не хотел умереть дома, не желал, чтобы родные видели «кухню смерти». 21 мая 1957 года Вертинский '
            'готовился к концерту в Ленинграде, был сдержан и немногословен. Он находился в своем 208-м номере '
            '«Астории», когда начался сердечный приступ. Лекарства под рукой не оказалось. Как выяснилось позже — оно '
            'бы уже не помогло. При вскрытии сосуды рассыпались, как хрупкое стекло',
            'Назначен куратор строительства российской Кремниевой долины',
            'Дмитрий Медведев доверил пост руководителя иннограда миллиардеру Виктору Вексельбергу.',
            'Всё меньше остаётся нерешённых вопросов, касающихся возведения в России уникального Центра по разработке '
            'и коммерциализации новых технологий. Власти уже не только выбрали площадку для строительства '
            'отечественной Кремниевой долины в подмосковном Сколково, а также частично одобрили концепцию наукограда, '
            'но и определили куратора большой инновационной стройки. «Были проведены определённые консультации по '
            'поводу того, кто конкретно мог бы осуществлять такого рода работу. Мною принято решение, что российскую '
            'часть этой координирующей структуры, которую мы создадим, возглавит Виктор Феликсович Вексельберг», — '
            'цитирует «Взгляд» Дмитрия Медведева.',
            'Исходя из заявления президента, понятно, что у проекта будут не только российские инвесторы, но и '
            'иностранные партнёры, в числе которых, по словам главы государства, будут и представители иностранных '
            'научных кругов. Именно на базе взаимодействия науки и бизнеса должен появиться и работать инноград. «Всё '
            'это затеяли не ради того, чтобы построить определённое количество коттеджей или же создать там нормальные '
            'производственные условия, лаборатории. Это всё важно, но это всё инфраструктура. Самое главное, чтобы '
            'там появились люди. Для того чтобы люди появились, должна быть внятная система управления. Эта система '
            'управления зависит от нас. Я думаю, что с учётом масштабности этого проекта, а с другой стороны, того, '
            'что в реализации этого проекта должны быть заинтересованы не только государственные структуры, но, '
            'прежде всего, российский бизнес, я считаю, что координацией российский бизнес и мог бы заняться», — '
            'заявил Дмитрий Медведев.',
            'Это выступление президента вполне объясняет выбор руководителя проекта. Виктор Вексельберг — бизнесмен с '
            '30-летним стажем, капитал которого оценивается в 6,4 млрд долларов. Вексельберг является главой правления '
            'ОАО «Тюменская нефтяная компания» (ТНК) и президентом ЗАО «Ренова». Именно он является владельцем '
            'значительной части российского титана и алюминиевого бизнеса.',
            'О том, почему площадкой для строительства Кремниевой долины выбрано Подмосковье, читайте в статье '
            'Частного корреспондента «Сколково назначили Кремниевой долиной»'
        ]
        y_true = [
            {
                "PERSON": [(0, 20)]
            },
            {
                "PERSON": [(71, 76)]
            },
            {
                "PERSON": [(36, 58)]
            },
            {
                "LOCATION": [(55, 63), (66, 84), (87, 93)],
                "PERSON": [(281, 289)]
            },
            {
                "PERSON": [(45, 56)]
            },
            {
                "LOCATION": [(334, 344)],
                "PERSON": [(26, 40), (299, 314)]
            },
            {
                "PERSON": [(0, 14)]
            },
            {
                "PERSON": [(28, 39), (70, 74), (146, 151), (362, 366)]
            },
            {
                "PERSON": [(207, 211)]
            },
            {
                "PERSON": [(30, 34), (51, 55), (182, 186), (208, 217), (250, 266), (367, 378), (392, 396), (436, 447),
                           (471, 489)],
                "LOCATION": [(44, 49), (71, 76), (498, 508)],
                "ORG": [(269, 295)]
            },
            {
                "PERSON": [(107, 110), (121, 132)]
            },
            {
                "PERSON": [(0, 20), (104, 123), (126, 144), (146, 155), (184, 188), (191, 195), (197, 215), (251, 267),
                           (338, 357)]
            },
            {
                "PERSON": [(36, 58)]
            },
            {
                "PERSON": [(33, 44)],
                "LOCATION": [(168, 185), (189, 197), (198, 204)],
                "ORG": [(230, 249)]
            },
            {
                "PERSON": [(9, 16), (89, 97), (108, 128), (131, 148), (430, 455)],
                "LOCATION": [(27, 39), (191, 198), (414, 422)],
                "ORG": [(2, 16)]
            },
            dict(),
            {
                "PERSON": [(87, 98)],
                "ORG": [(18, 56)]
            },
            {
                "LOCATION": [(72, 80)],
                "PERSON": [(51, 61)]
            },
            {
                "LOCATION": [(151, 157)],
                "PERSON": [(130, 140)]
            },
            {
                "PERSON": [(80, 97), (233, 246)],
                "ORG": [(65, 97), (177, 182)]
            },
            {
                "PERSON": [(373, 383)]
            },
            {
                "PERSON": [(0, 10), (46, 54), (103, 109), (166, 185), (199, 209)]
            },
            {
                "LOCATION": [(135, 141)],
                "PERSON": [(36, 47), (79, 84), (177, 187)]
            },
            {
                "PERSON": [(12, 22), (49, 60), (94, 99)],
                "ORG": [(34, 60)]
            },
            {
                "PERSON": [(105, 109)]
            },
            {
                "LOCATION": [(389, 408)],
                "PERSON": [(162, 178), (202, 212), (251, 256), (257, 268)]
            },
            {
                "PERSON": [(171, 187), (226, 237)],
                "ORG": [(134, 169)]
            },
            {
                "PERSON": [(160, 171), (215, 236), (257, 262)]
            },
            {
                "PERSON": [(68, 78), (267, 277)]
            },
            dict(),
            {
                "PERSON": [(123, 134)],
                "ORG": [(146, 148)]
            },
            {
                "PERSON": [(30, 41), (197, 207)],
                "ORG": [(52, 54)]
            },
            {
                "LOCATION": [(107, 113)],
                "PERSON": [(39, 59), (78, 86), (88, 98)]
            },
            {
                "LOCATION": [(23, 38), (203, 210), (212, 218), (220, 228), (230, 237), (239, 246), (248, 257),
                             (259, 265), (267, 272), (274, 281), (283, 286)],
                "PERSON": [(128, 148), (403, 420), (422, 441), (450, 459)]
            },
            {
                "PERSON": [(226, 237)],
                "LOCATION": [(2, 12)],
                "ORG": [(256, 260), (357, 361)]
            },
            {
                "LOCATION": [(118, 125), (307, 313)],
                "PERSON": [(34, 44), (263, 268), (332, 340)]
            },
            {
                "PERSON": [(20, 30), (96, 117)],
                "LOCATION": [(155, 162)],
                "ORG": [(67, 88), (133, 152)]
            },
            {
                "LOCATION": [(31, 36), (41, 47), (50, 57), (99, 105), (335, 341), (381, 385), (447, 452)],
                "PERSON": [(153, 161), (279, 289), (426, 436)]
            },
            {
                "PERSON": [(13, 20), (29, 40), (103, 108), (194, 201), (233, 244), (388, 407), (410, 427), (438, 445),
                           (446, 456)]
            },
            {
                "LOCATION": [(338, 342), (392, 397)],
                "PERSON": [(0, 10), (142, 147), (149, 163), (248, 260), (262, 276), (304, 315)]
            },
            {
                "LOCATION": [(45, 51), (52, 56), (77, 83), (258, 262), (320, 325)],
                "PERSON": [(58, 68), (120, 125), (174, 182), (425, 435), (630, 640)]
            },
            {
                "LOCATION": [(42, 46), (221, 227), (251, 255)],
                "PERSON": [(74, 85)]
            },
            {
                "LOCATION": [(92, 96), (194, 200), (262, 268)],
                "PERSON": [(70, 80), (345, 356), (379, 408), (529, 536), (543, 551), (564, 581), (583, 603), (652, 670)]
            },
            {
                "PERSON": [(27, 38), (53, 58), (68, 86)]
            },
            {
                "LOCATION": [(319, 325)],
                "PERSON": [(20, 25), (26, 36), (65, 80), (95, 105), (169, 185), (239, 243), (286, 296)]
            },
            {
                "PERSON": [(31, 42), (607, 618)],
                "LOCATION": [(137, 146), (260, 269), (399, 416), (506, 520), (673, 690)],
                "ORG": [(722, 734)]
            },
            {
                "PERSON": [(105, 120), (123, 139), (140, 151), (323, 333)]
            },
            {
                "PERSON": [(13, 23), (95, 100)],
                "LOCATION": [(179, 191), (195, 211)],
                "ORG": [(102, 114)]
            },
            {
                "PERSON": [(8, 19), (327, 344)]
            },
            {
                "LOCATION": [(123, 133), (199, 206)],
                "PERSON": [(89, 99)]
            },
            {
                "LOCATION": [(31, 59)]
            },
            {
                "LOCATION": [(43, 52)],
                "PERSON": [(0, 16), (65, 85)]
            },
            {
                "ORG": [(84, 140), (620, 626)],
                "LOCATION": [(65, 71), (212, 229), (232, 253), (291, 301)],
                "PERSON": [(576, 605), (628, 645)]
            },
            {
                "LOCATION": [(153, 164), (290, 298)],
                "PERSON": [(925, 941)]
            },
            {
                "ORG": [(201, 243), (246, 249), (265, 276)],
                "PERSON": [(72, 90), (173, 184)]
            },
            {
                "LOCATION": [(42, 59), (68, 79), (123, 131), (142, 160)],
                "ORG": [(98, 121)]
            }
        ]
        X_loaded, y_loaded = load_dataset(file_name)
        self.assertIsInstance(X_loaded, list)
        self.assertIsInstance(y_loaded, list)
        self.assertEqual(len(X_true), len(X_loaded))
        self.assertEqual(len(y_true), len(y_loaded))
        for sample_idx in range(len(X_true)):
            self.assertEqual(X_true[sample_idx], X_loaded[sample_idx])
            self.assertIsInstance(y_loaded[sample_idx], dict)
            self.assertEqual(set(y_true[sample_idx]), set(y_loaded[sample_idx]))
            for ne_type in y_true[sample_idx]:
                self.assertIsInstance(y_loaded[sample_idx][ne_type], list)
                self.assertEqual(len(y_true[sample_idx][ne_type]), len(y_loaded[sample_idx][ne_type]),
                                 msg='Sample {0}'.format(sample_idx))
                for entity_idx in range(len(y_true[sample_idx][ne_type])):
                    self.assertEqual(y_true[sample_idx][ne_type][entity_idx], y_loaded[sample_idx][ne_type][entity_idx])

    def test_calculate_prediction_quality(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        X_true, y_true = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        X_pred, y_pred = load_dataset(os.path.join(base_dir, 'predicted_named_entities.json'))
        self.assertEqual(X_true, X_pred)
        f1, precision, recall = ELMo_NER.calculate_prediction_quality(y_true, y_pred, ('LOCATION', 'PERSON', 'ORG'))
        self.assertIsInstance(f1, float)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertAlmostEqual(f1, 0.842037, places=3)
        self.assertAlmostEqual(precision, 0.908352, places=3)
        self.assertAlmostEqual(recall, 0.784746, places=3)

    def test_fit_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=False, max_epochs=3, batch_size=4, max_seq_length=64, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, ELMo_NER)
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_elmo'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.elmo_hub_module_handle, str)
        self.assertIsInstance(res.finetune_elmo, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'transition_params_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'input_tokens_'))
        self.assertTrue(hasattr(res, 'sequence_lengths_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))

    def test_fit_positive02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=True, max_epochs=3, batch_size=2, max_seq_length=64, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=42, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, ELMo_NER)
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_elmo'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.elmo_hub_module_handle, str)
        self.assertIsInstance(res.finetune_elmo, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertEqual(res.random_seed, 42)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'transition_params_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'input_tokens_'))
        self.assertTrue(hasattr(res, 'sequence_lengths_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))

    def test_fit_positive03(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=False, max_epochs=3, batch_size=4, max_seq_length=64, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, ELMo_NER)
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_elmo'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.elmo_hub_module_handle, str)
        self.assertIsInstance(res.finetune_elmo, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'transition_params_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'input_tokens_'))
        self.assertTrue(hasattr(res, 'sequence_lengths_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))

    def test_fit_predict(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=False, max_epochs=3, batch_size=4, max_seq_length=64, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, ELMo_NER)
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_elmo'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.elmo_hub_module_handle, str)
        self.assertIsInstance(res.finetune_elmo, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'transition_params_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'input_tokens_'))
        self.assertTrue(hasattr(res, 'sequence_lengths_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        y_pred = res.predict(X_train)
        self.assertIsInstance(y_pred, list)
        self.assertEqual(len(X_train), len(y_pred))
        for sample_idx in range(len(y_pred)):
            self.assertIsInstance(y_pred[sample_idx], dict)
        f1, precision, recall = res.calculate_prediction_quality(y_train, y_pred, res.classes_list_)
        self.assertGreater(f1, 0.0)
        self.assertGreater(precision, 0.0)
        self.assertGreater(recall, 0.0)

    def test_predict_negative(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=False, max_epochs=3, batch_size=4, random_seed=None,
                            elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        with self.assertRaises(NotFittedError):
            _ = self.ner.predict(X_train)

    def test_serialize_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=False, max_epochs=3, batch_size=4, max_seq_length=64, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        res = self.ner.fit(X_train, y_train)
        self.assertIsInstance(res, ELMo_NER)
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'lr'))
        self.assertTrue(hasattr(res, 'l2_reg'))
        self.assertTrue(hasattr(res, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(res, 'finetune_elmo'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'patience'))
        self.assertTrue(hasattr(res, 'random_seed'))
        self.assertTrue(hasattr(res, 'gpu_memory_frac'))
        self.assertTrue(hasattr(res, 'max_seq_length'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertIsInstance(res.batch_size, int)
        self.assertIsInstance(res.lr, float)
        self.assertIsInstance(res.l2_reg, float)
        self.assertIsInstance(res.elmo_hub_module_handle, str)
        self.assertIsInstance(res.finetune_elmo, bool)
        self.assertIsInstance(res.max_epochs, int)
        self.assertIsInstance(res.patience, int)
        self.assertIsInstance(res.random_seed, int)
        self.assertIsInstance(res.gpu_memory_frac, float)
        self.assertIsInstance(res.max_seq_length, int)
        self.assertIsInstance(res.validation_fraction, float)
        self.assertIsInstance(res.verbose, bool)
        self.assertTrue(hasattr(res, 'classes_list_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'transition_params_'))
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'input_tokens_'))
        self.assertTrue(hasattr(res, 'sequence_lengths_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertEqual(res.classes_list_, ('LOCATION', 'ORG', 'PERSON'))
        y_pred1 = res.predict(X_train)
        self.assertIsInstance(y_pred1, list)
        self.assertEqual(len(X_train), len(y_pred1))
        for sample_idx in range(len(y_pred1)):
            self.assertIsInstance(y_pred1[sample_idx], dict)
        f1, precision, recall = res.calculate_prediction_quality(y_train, y_pred1, res.classes_list_)
        self.assertGreater(f1, 0.0)
        self.assertGreater(precision, 0.0)
        self.assertGreater(recall, 0.0)
        self.temp_file_name = tempfile.NamedTemporaryFile(mode='w').name
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
        self.ner = ELMo_NER(random_seed=31, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        old_batch_size = self.ner.batch_size
        old_lr = self.ner.lr
        old_l2_reg = self.ner.l2_reg
        old_elmo_hub_module_handle = self.ner.elmo_hub_module_handle
        old_finetune_elmo = self.ner.finetune_elmo
        old_max_epochs = self.ner.max_epochs
        old_patience = self.ner.patience
        old_random_seed = self.ner.random_seed
        old_gpu_memory_frac = self.ner.gpu_memory_frac
        old_max_seq_length = self.ner.max_seq_length
        old_validation_fraction = self.ner.validation_fraction
        old_verbose = self.ner.verbose
        self.temp_file_name = tempfile.NamedTemporaryFile().name
        with open(self.temp_file_name, mode='wb') as fp:
            pickle.dump(self.ner, fp)
        del self.ner
        gc.collect()
        with open(self.temp_file_name, mode='rb') as fp:
            self.ner = pickle.load(fp)
        self.assertIsInstance(self.ner, ELMo_NER)
        self.assertTrue(hasattr(self.ner, 'batch_size'))
        self.assertTrue(hasattr(self.ner, 'lr'))
        self.assertTrue(hasattr(self.ner, 'l2_reg'))
        self.assertTrue(hasattr(self.ner, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(self.ner, 'finetune_elmo'))
        self.assertTrue(hasattr(self.ner, 'max_epochs'))
        self.assertTrue(hasattr(self.ner, 'patience'))
        self.assertTrue(hasattr(self.ner, 'random_seed'))
        self.assertTrue(hasattr(self.ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.ner, 'verbose'))
        self.assertEqual(self.ner.batch_size, old_batch_size)
        self.assertAlmostEqual(self.ner.lr, old_lr)
        self.assertAlmostEqual(self.ner.l2_reg, old_l2_reg)
        self.assertEqual(self.ner.elmo_hub_module_handle, old_elmo_hub_module_handle)
        self.assertEqual(self.ner.finetune_elmo, old_finetune_elmo)
        self.assertEqual(self.ner.max_epochs, old_max_epochs)
        self.assertEqual(self.ner.patience, old_patience)
        self.assertAlmostEqual(self.ner.gpu_memory_frac, old_gpu_memory_frac)
        self.assertEqual(self.ner.max_seq_length, old_max_seq_length)
        self.assertAlmostEqual(self.ner.validation_fraction, old_validation_fraction)
        self.assertEqual(self.ner.verbose, old_verbose)
        self.assertEqual(self.ner.random_seed, old_random_seed)

    def test_copy_positive01(self):
        self.ner = ELMo_NER(random_seed=0, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        self.another_ner = copy.copy(self.ner)
        self.assertIsInstance(self.another_ner, ELMo_NER)
        self.assertIsNot(self.ner, self.another_ner)
        self.assertTrue(hasattr(self.another_ner, 'batch_size'))
        self.assertTrue(hasattr(self.another_ner, 'lr'))
        self.assertTrue(hasattr(self.another_ner, 'l2_reg'))
        self.assertTrue(hasattr(self.another_ner, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(self.another_ner, 'finetune_elmo'))
        self.assertTrue(hasattr(self.another_ner, 'max_epochs'))
        self.assertTrue(hasattr(self.another_ner, 'patience'))
        self.assertTrue(hasattr(self.another_ner, 'random_seed'))
        self.assertTrue(hasattr(self.another_ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.another_ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.another_ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.another_ner, 'verbose'))
        self.assertEqual(self.ner.batch_size, self.another_ner.batch_size)
        self.assertAlmostEqual(self.ner.lr, self.another_ner.lr)
        self.assertAlmostEqual(self.ner.l2_reg, self.another_ner.l2_reg)
        self.assertEqual(self.ner.elmo_hub_module_handle, self.another_ner.elmo_hub_module_handle)
        self.assertEqual(self.ner.finetune_elmo, self.another_ner.finetune_elmo)
        self.assertEqual(self.ner.max_epochs, self.another_ner.max_epochs)
        self.assertEqual(self.ner.patience, self.another_ner.patience)
        self.assertEqual(self.ner.random_seed, self.another_ner.random_seed)
        self.assertAlmostEqual(self.ner.gpu_memory_frac, self.another_ner.gpu_memory_frac)
        self.assertEqual(self.ner.max_seq_length, self.another_ner.max_seq_length)
        self.assertAlmostEqual(self.ner.validation_fraction, self.another_ner.validation_fraction)
        self.assertEqual(self.ner.verbose, self.another_ner.verbose)

    def test_copy_positive02(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.ner = ELMo_NER(finetune_elmo=False, max_epochs=3, batch_size=4, max_seq_length=64, gpu_memory_frac=0.9,
                            validation_fraction=0.3, random_seed=None, elmo_hub_module_handle=self.ELMO_HUB_MODULE)
        X_train, y_train = load_dataset(os.path.join(base_dir, 'true_named_entities.json'))
        self.ner.fit(X_train, y_train)
        self.another_ner = copy.copy(self.ner)
        self.assertIsInstance(self.another_ner, ELMo_NER)
        self.assertIsNot(self.ner, self.another_ner)
        self.assertTrue(hasattr(self.another_ner, 'batch_size'))
        self.assertTrue(hasattr(self.another_ner, 'lr'))
        self.assertTrue(hasattr(self.another_ner, 'l2_reg'))
        self.assertTrue(hasattr(self.another_ner, 'elmo_hub_module_handle'))
        self.assertTrue(hasattr(self.another_ner, 'finetune_elmo'))
        self.assertTrue(hasattr(self.another_ner, 'max_epochs'))
        self.assertTrue(hasattr(self.another_ner, 'patience'))
        self.assertTrue(hasattr(self.another_ner, 'random_seed'))
        self.assertTrue(hasattr(self.another_ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.another_ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.another_ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.another_ner, 'verbose'))
        self.assertTrue(hasattr(self.another_ner, 'classes_list_'))
        self.assertTrue(hasattr(self.another_ner, 'logits_'))
        self.assertTrue(hasattr(self.another_ner, 'transition_params_'))
        self.assertTrue(hasattr(self.another_ner, 'tokenizer_'))
        self.assertTrue(hasattr(self.another_ner, 'input_tokens_'))
        self.assertTrue(hasattr(self.another_ner, 'sequence_lengths_'))
        self.assertTrue(hasattr(self.another_ner, 'y_ph_'))
        self.assertTrue(hasattr(self.another_ner, 'sess_'))
        self.assertEqual(self.ner.batch_size, self.another_ner.batch_size)
        self.assertAlmostEqual(self.ner.lr, self.another_ner.lr)
        self.assertAlmostEqual(self.ner.l2_reg, self.another_ner.l2_reg)
        self.assertEqual(self.ner.elmo_hub_module_handle, self.another_ner.elmo_hub_module_handle)
        self.assertEqual(self.ner.finetune_elmo, self.another_ner.finetune_elmo)
        self.assertEqual(self.ner.max_epochs, self.another_ner.max_epochs)
        self.assertEqual(self.ner.patience, self.another_ner.patience)
        self.assertEqual(self.ner.random_seed, self.another_ner.random_seed)
        self.assertAlmostEqual(self.ner.gpu_memory_frac, self.another_ner.gpu_memory_frac)
        self.assertEqual(self.ner.max_seq_length, self.another_ner.max_seq_length)
        self.assertAlmostEqual(self.ner.validation_fraction, self.another_ner.validation_fraction)
        self.assertEqual(self.ner.verbose, self.another_ner.verbose)
        self.assertIs(self.ner.classes_list_, self.another_ner.classes_list_)
        self.assertIs(self.ner.logits_, self.another_ner.logits_)
        self.assertIs(self.ner.transition_params_, self.another_ner.transition_params_)
        self.assertIs(self.ner.tokenizer_, self.another_ner.tokenizer_)
        self.assertIs(self.ner.input_tokens_, self.another_ner.input_tokens_)
        self.assertIs(self.ner.sequence_lengths_, self.another_ner.sequence_lengths_)
        self.assertIs(self.ner.y_ph_, self.another_ner.y_ph_)
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
        calc_entities = ELMo_NER.calculate_bounds_of_named_entities(bounds_of_tokens, classes_list, labels_of_tokens)
        self.assertIsInstance(calc_entities, dict)
        self.assertEqual(set(true_entities.keys()), set(calc_entities.keys()))
        for entity_type in true_entities:
            self.assertEqual(true_entities[entity_type], calc_entities[entity_type])


if __name__ == '__main__':
    unittest.main(verbosity=2)
