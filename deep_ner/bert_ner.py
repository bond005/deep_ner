import copy
import logging
import os
import random
import tempfile
import time
from typing import Dict, Union, List, Tuple

from nltk.tokenize.nist import NISTTokenizer
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import tensorflow_hub as tfhub
from bert.tokenization import FullTokenizer
from bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

from .quality import calculate_prediction_quality
from .dataset_splitting import split_dataset
from .dataset import NER_dataset
from .data_loader import DataLoader


tfhub_path = '/mnt/data/jupyter/TFHUB_CACHE_DIR'
os.environ["TFHUB_CACHE_DIR"] = tfhub_path

bert_ner_logger = logging.getLogger(__name__)

handlers = []

bert_ner_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
for handler in handlers:
    handler.setFormatter(formatter)
    bert_ner_logger.addHandler(handler)


class BERT_NER(BaseEstimator, ClassifierMixin):
    PATH_TO_BERT = '/mnt/data/jupyter/zp_deep_ner/pretrained/rubert_cased_L-12_H-768_A-12_v1'

    def __init__(self, finetune_bert: bool=False,
                 bert_hub_module_handle: Union[str, None]='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                 batch_size: int=32, max_seq_length: int=512, lr: float=1e-3, lstm_units: Union[int, None]=256,
                 l2_reg: float=1e-4, clip_norm: Union[float, None]=5.0, validation_fraction: float=0.1,
                 max_epochs: int=10, patience: int=3, gpu_memory_frac: float=1.0, verbose: bool=False,
                 random_seed: Union[int, None]=None):
        self.batch_size = batch_size
        self.lr = lr
        self.l2_reg = l2_reg
        self.clip_norm = clip_norm
        self.bert_hub_module_handle = bert_hub_module_handle
        self.finetune_bert = finetune_bert
        self.lstm_units = lstm_units
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed
        self.gpu_memory_frac = gpu_memory_frac
        self.max_seq_length = max_seq_length
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.nltk_tokenizer_ = NISTTokenizer()

    def __del__(self):
        if hasattr(self, 'classes_list_'):
            del self.classes_list_
        if hasattr(self, 'shapes_list_'):
            del self.shapes_list_
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()

    def fit(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array],
            validation_data: Union[None, Tuple[Union[list, tuple, np.array], Union[list, tuple, np.array]]]=None):
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, finetune_bert=self.finetune_bert,
            lstm_units=self.lstm_units, batch_size=self.batch_size, max_seq_length=self.max_seq_length, lr=self.lr,
            l2_reg=self.l2_reg, validation_fraction=self.validation_fraction, max_epochs=self.max_epochs,
            patience=self.patience, gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose,
            clip_norm=self.clip_norm, random_seed=self.random_seed
        )

        if hasattr(self, 'shapes_list_'):
            del self.shapes_list_
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()
        # Set random seed
        if self.random_seed is None:
            self.random_seed = int(round(time.time()))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        # Create Dataset
        self.classes_list_ = NER_dataset.make_classes_list(y)
        if validation_data is None:
            X_train_, y_train_, X_val_, y_val_ = NER_dataset.split_dataset(X, y, self.validation_fraction)
        else:
            if (not isinstance(validation_data, tuple)) and (not isinstance(validation_data, list)):
                raise ValueError('')
            if len(validation_data) != 2:
                raise ValueError('')
            NER_dataset.check_Xy(validation_data[0], 'X_val', validation_data[1], 'y_val')
            classes_list_for_validation = NER_dataset.make_classes_list(validation_data[1])
            if not (set(classes_list_for_validation) <= set(self.classes_list_)):
                raise ValueError('')
            X_train_ = X
            y_train_ = y
            X_val_ = validation_data[0]
            y_val_ = validation_data[1]

        self.train_dataset = NER_dataset(texts=X_train_,
                                         annotations=y_train_,
                                         max_seq_length=self.max_seq_length,
                                         bert_hub_module_handle=self.bert_hub_module_handle)
        self.shapes_list_ = self.train_dataset.shapes_list_
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        valid_dataset = NER_dataset(texts=X_val_,
                                    annotations=y_val_,
                                    max_seq_length=self.max_seq_length,
                                    shapes_list=self.shapes_list_,
                                    bert_hub_module_handle=self.bert_hub_module_handle)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False)

        if self.verbose:
            lengths_of_texts = []
            sum_of_lengths = 0
            for sample_idx in range(len(y_train_)):
                lengths_of_texts.append(sum(self.train_dataset.X_tokenized[1][sample_idx]))
                sum_of_lengths += lengths_of_texts[-1]
            mean_length = sum_of_lengths / float(len(lengths_of_texts))
            lengths_of_texts.sort()
            bert_ner_logger.info('Maximal length of text (in BPE): {0}'.format(max(lengths_of_texts)))
            bert_ner_logger.info('Mean length of text (in BPE): {0}'.format(mean_length))
            bert_ner_logger.info('Median length of text (in BPE): {0}'.format(
                lengths_of_texts[len(lengths_of_texts) // 2]))

        # Build model
        train_op, accuracy = self.build_model()
        init = tf.global_variables_initializer()
        init.run(session=self.sess_)
        tmp_model_name = self.get_temp_model_name()

        n_epochs_without_improving = 0
        bounds_of_tokens_for_validation = valid_dataset.bounds_of_tokens_for_training
        try:
            best_acc = None
            for epoch in range(self.max_epochs):
                # Train
                feed_dict_for_batch = None
                for X_batch, y_batch in train_loader:
                    feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                    self.sess_.run(train_op, feed_dict=feed_dict_for_batch)
                acc_train = accuracy.eval(feed_dict=feed_dict_for_batch, session=self.sess_)
                # Validation
                acc_test = 0.0
                y_pred = []
                for X_batch, y_batch in valid_loader:
                    feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                    acc_test_, logits, trans_params, mask = self.sess_.run(
                        [accuracy, self.logits_, self.transition_params_, self.input_mask_],
                        feed_dict=feed_dict_for_batch
                    )
                    acc_test += self.batch_size * acc_test_
                    sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
                    for logit, sequence_length in zip(logits, sequence_lengths):
                        logit = logit[:int(sequence_length)]
                        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                        y_pred += [viterbi_seq]
                acc_test /= float(len(valid_loader))
                if self.verbose:
                    bert_ner_logger.info('Epoch {0}'.format(epoch))
                    bert_ner_logger.info('  Train acc.: {0: 10.8f}'.format(acc_train))
                    bert_ner_logger.info('  Val. acc.:  {0: 10.8f}'.format(acc_test))
                    pred_entities_val = []
                    for sample_idx, labels_in_text in enumerate(y_pred[0:len(X_val_)]):
                        n_tokens = len(labels_in_text)
                        new_entities = self.calculate_bounds_of_named_entities(
                            bounds_of_tokens_for_validation[sample_idx],
                            self.classes_list_,
                            labels_in_text[1:(n_tokens - 1)]
                        )
                        pred_entities_val.append(new_entities)
                    f1_test, precision_test, recall_test, quality_by_entities = calculate_prediction_quality(
                        y_val_, pred_entities_val, self.classes_list_)
                    if best_acc is None:
                        best_acc = f1_test
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    elif f1_test > best_acc:
                        best_acc = f1_test
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    else:
                        n_epochs_without_improving += 1
                    if self.verbose:
                        bert_ner_logger.info('  Val. quality for all entities:')
                        bert_ner_logger.info('      F1={0:>6.4f}, P={1:>6.4f}, R={2:>6.4f}'.format(
                            f1_test, precision_test, recall_test))
                        max_text_width = 0
                        for ne_type in sorted(list(quality_by_entities.keys())):
                            text_width = len(ne_type)
                            if text_width > max_text_width:
                                max_text_width = text_width
                        for ne_type in sorted(list(quality_by_entities.keys())):
                            bert_ner_logger.info('    Val. quality for {0:>{1}}:'.format(ne_type, max_text_width))
                            bert_ner_logger.info('      F1={0:>6.4f}, P={1:>6.4f}, R={2:>6.4f})'.format(
                                quality_by_entities[ne_type][0], quality_by_entities[ne_type][1],
                                quality_by_entities[ne_type][2]))
                    del y_pred, pred_entities_val
                else:
                    if best_acc is None:
                        best_acc = acc_train
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    elif acc_train > best_acc:
                        best_acc = acc_train
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    else:
                        n_epochs_without_improving += 1
                    if self.verbose:
                        bert_ner_logger.info('{0:>5}   {1:>10.8f}'.format(epoch, acc_train))
                if n_epochs_without_improving >= self.patience:
                    if self.verbose:
                        bert_ner_logger.info('Epoch %05d: early stopping' % (epoch + 1))
                    break
            if best_acc is not None:
                self.finalize_model()
                _, accuracy = self.build_model()
                self.load_model(tmp_model_name)
                if self.verbose:
                    acc_test = 0.0
                    y_pred = []

                    for X_batch, y_batch in valid_loader:
                        feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                        acc_test_, logits, trans_params, mask = self.sess_.run(
                            [accuracy, self.logits_, self.transition_params_, self.input_mask_],
                            feed_dict=feed_dict_for_batch
                        )
                        acc_test += self.batch_size * acc_test_
                        sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
                        for logit, sequence_length in zip(logits, sequence_lengths):
                            logit = logit[:int(sequence_length)]
                            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                            y_pred += [viterbi_seq]
                    acc_test /= float(len(valid_loader))
                    pred_entities_val = []
                    for sample_idx, labels_in_text in enumerate(y_pred[0:len(X_val_)]):
                        n_tokens = len(labels_in_text)
                        new_entities = self.calculate_bounds_of_named_entities(
                            bounds_of_tokens_for_validation[sample_idx],
                            self.classes_list_,
                            labels_in_text[1:(n_tokens - 1)]
                        )
                        pred_entities_val.append(new_entities)
                    f1_test, _, _, _ = calculate_prediction_quality(y_val_, pred_entities_val,
                                                                    self.classes_list_)
                    bert_ner_logger.info('Best val. F1 is {0:>8.6f}'.format(f1_test))
                    bert_ner_logger.info('Best val. acc. is {0:>10.8f}'.format(acc_test))
        finally:
            for cur_name in self.find_all_model_files(tmp_model_name):
                os.remove(cur_name)
        return self

    def predict(self, X: Union[list, tuple, np.array]) -> List[Dict[str, List[Tuple[int, int]]]]:
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, finetune_bert=self.finetune_bert,
            lstm_units=self.lstm_units, batch_size=self.batch_size, max_seq_length=self.max_seq_length, lr=self.lr,
            l2_reg=self.l2_reg, validation_fraction=self.validation_fraction, max_epochs=self.max_epochs,
            patience=self.patience, gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose,
            random_seed=self.random_seed, clip_norm=self.clip_norm
        )

        self.is_fitted()
        # Create Dataset
        NER_dataset.check_X(X, 'X')
        test_dataset = NER_dataset(texts=X, annotations=None,
                                   max_seq_length=self.max_seq_length,
                                   shapes_list=self.shapes_list_,
                                   bert_hub_module_handle=self.bert_hub_module_handle,
                                   mode='test')
        test_loader = DataLoader(dataset=test_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False)

        y_pred = []
        for X_batch in test_loader:
            feed_dict = self.fill_feed_dict(X_batch)
            logits, trans_params, mask = self.sess_.run([self.logits_, self.transition_params_, self.input_mask_],
                                                        feed_dict=feed_dict)
            sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:int(sequence_length)]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                y_pred += [viterbi_seq]

        bounds_of_tokens = test_dataset.bounds_of_tokens_for_training
        recognized_entities_in_texts = list()
        n_samples = len(X)
        for sample_idx, labels_in_text in enumerate(y_pred[0: n_samples]):
            n_tokens = len(labels_in_text)
            new_entities = self.calculate_bounds_of_named_entities(bounds_of_tokens[sample_idx], self.classes_list_,
                                                                   labels_in_text[1:(n_tokens - 1)])
            recognized_entities_in_texts.append(new_entities)
        return recognized_entities_in_texts

    def is_fitted(self):
        check_is_fitted(self, ['classes_list_', 'shapes_list_', 'logits_', 'transition_params_',
                               'input_ids_', 'input_mask_', 'segment_ids_', 'additional_features_', 'y_ph_', 'sess_'])

    def score(self, X, y, sample_weight=None) -> float:
        y_pred = self.predict(X)
        return calculate_prediction_quality(y, y_pred, self.classes_list_)[0]

    def fit_predict(self, X: Union[list, tuple, np.array],  y: Union[list, tuple, np.array], **kwargs):
        return self.fit(X, y).predict(X)

    def fill_feed_dict(self, X: List[np.array], y: np.array=None) -> dict:
        assert len(X) == 4
        assert len(X[0]) == self.batch_size
        feed_dict = {
            ph: x for ph, x in zip([self.input_ids_, self.input_mask_, self.segment_ids_, self.additional_features_], X)
        }
        if y is not None:
            feed_dict[self.y_ph_] = y
        return feed_dict

    def get_params(self, deep=True) -> dict:
        return {'bert_hub_module_handle': self.bert_hub_module_handle, 'finetune_bert': self.finetune_bert,
                'lstm_units': self.lstm_units, 'batch_size': self.batch_size, 'max_seq_length': self.max_seq_length,
                'lr': self.lr, 'l2_reg': self.l2_reg, 'clip_norm': self.clip_norm, 'max_epochs': self.max_epochs,
                'patience': self.patience, 'validation_fraction': self.validation_fraction,
                'gpu_memory_frac': self.gpu_memory_frac, 'verbose': self.verbose, 'random_seed': self.random_seed}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            bert_hub_module_handle=self.bert_hub_module_handle, finetune_bert=self.finetune_bert,
            lstm_units=self.lstm_units, batch_size=self.batch_size, max_seq_length=self.max_seq_length, lr=self.lr,
            l2_reg=self.l2_reg, clip_norm=self.clip_norm, validation_fraction=self.validation_fraction,
            max_epochs=self.max_epochs, patience=self.patience, gpu_memory_frac=self.gpu_memory_frac,
            verbose=self.verbose, random_seed=self.random_seed
        )
        result.nltk_tokenizer_ = NISTTokenizer()
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_list_ = self.classes_list_
            result.shapes_list_ = self.shapes_list_
            result.logits_ = self.logits_
            result.transition_params_ = self.transition_params_
            result.train_dataset = self.train_dataset
            result.input_ids_ = self.input_ids_
            result.input_mask_ = self.input_mask_
            result.segment_ids_ = self.segment_ids_
            result.additional_features_ = self.additional_features_
            result.y_ph_ = self.y_ph_
            result.sess_ = self.sess_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            bert_hub_module_handle=self.bert_hub_module_handle,  finetune_bert=self.finetune_bert,
            lstm_units=self.lstm_units, batch_size=self.batch_size, max_seq_length=self.max_seq_length, lr=self.lr,
            l2_reg=self.l2_reg, clip_norm=self.clip_norm, validation_fraction=self.validation_fraction,
            max_epochs=self.max_epochs, patience=self.patience, gpu_memory_frac=self.gpu_memory_frac,
            verbose=self.verbose, random_seed=self.random_seed
        )
        result.nltk_tokenizer_ = NISTTokenizer()
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_list_ = self.classes_list_
            result.shapes_list_ = self.shapes_list_
            result.logits_ = self.logits_
            result.transition_params_ = self.transition_params_
            result.tokenizer_ = self.tokenizer_
            result.input_ids_ = self.input_ids_
            result.input_mask_ = self.input_mask_
            result.segment_ids_ = self.segment_ids_
            result.additional_features_ = self.additional_features_
            result.y_ph_ = self.y_ph_
            result.sess_ = self.sess_
        return result

    def __getstate__(self):
        return self.dump_all()

    def __setstate__(self, state: dict):
        self.load_all(state)

    def dump_all(self):
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        params = self.get_params(True)
        if is_fitted:
            params['classes_list_'] = copy.copy(self.classes_list_)
            params['shapes_list_'] = copy.copy(self.shapes_list_)
            params['train_dataset'] = copy.deepcopy(self.train_dataset)
            model_file_name = self.get_temp_model_name()
            try:
                params['model_name_'] = os.path.basename(model_file_name)
                self.save_model(model_file_name)
                for cur_name in self.find_all_model_files(model_file_name):
                    with open(cur_name, 'rb') as fp:
                        model_data = fp.read()
                    params['model.' + os.path.basename(cur_name)] = model_data
                    del model_data
            finally:
                for cur_name in self.find_all_model_files(model_file_name):
                    os.remove(cur_name)
        return params

    def load_all(self, new_params: dict):
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected `{0}`, got `{1}`.'.format(type({0: 1}), type(new_params)))
        self.check_params(**new_params)
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()
        is_fitted = ('classes_list_' in new_params) and ('shapes_list_' in new_params) and \
                    ('tokenizer_' in new_params) and  ('model_name_' in new_params)
        model_files = list(
            filter(
                lambda it3: len(it3) > 0,
                map(
                    lambda it2: it2[len('model.'):].strip(),
                    filter(
                        lambda it1: it1.startswith('model.') and (len(it1) > len('model.')),
                        new_params.keys()
                    )
                )
            )
        )
        if is_fitted and (len(model_files) == 0):
            is_fitted = False
        if is_fitted:
            tmp_dir_name = tempfile.gettempdir()
            tmp_file_names = [os.path.join(tmp_dir_name, cur) for cur in model_files]
            for cur in tmp_file_names:
                if os.path.isfile(cur):
                    raise ValueError('File `{0}` exists, and so it cannot be used for data transmission!'.format(cur))
            self.set_params(**new_params)
            self.nltk_tokenizer_ = NISTTokenizer()
            self.classes_list_ = copy.copy(new_params['classes_list_'])
            self.shapes_list_ = copy.copy(new_params['shapes_list_'])
            self.tokenizer_ = copy.deepcopy(new_params['tokenizer_'])
            if self.random_seed is None:
                self.random_seed = int(round(time.time()))
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            try:
                for idx in range(len(model_files)):
                    with open(tmp_file_names[idx], 'wb') as fp:
                        fp.write(new_params['model.' + model_files[idx]])
                self.build_model()
                self.load_model(os.path.join(tmp_dir_name, new_params['model_name_']))
            finally:
                for cur in tmp_file_names:
                    if os.path.isfile(cur):
                        os.remove(cur)
        else:
            self.set_params(**new_params)
            self.nltk_tokenizer_ = NISTTokenizer()
        return self

    def build_model(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
        self.sess_ = tf.Session(config=config)
        self.input_ids_ = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32,
                                         name='input_ids')
        self.input_mask_ = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32,
                                          name='input_mask')
        self.segment_ids_ = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32,
                                           name='segment_ids')
        self.y_ph_ = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32, name='y_ph')
        bert_inputs = dict(
            input_ids=self.input_ids_,
            input_mask=self.input_mask_,
            segment_ids=self.segment_ids_
        )
        if self.bert_hub_module_handle is not None:
            bert_module = tfhub.Module(self.bert_hub_module_handle, trainable=True)
            bert_outputs = bert_module(bert_inputs, signature='tokens', as_dict=True)
            sequence_output = bert_outputs['sequence_output']
            if self.verbose:
                bert_ner_logger.info('The BERT model has been loaded from the TF-Hub.')
        else:
            if self.PATH_TO_BERT is None:
                raise ValueError('Path to the BERT model is not defined!')
            path_to_bert = os.path.normpath(self.PATH_TO_BERT)
            if not self.check_path_to_bert(path_to_bert):
                raise ValueError('`path_to_bert` is wrong! There are no BERT files into the directory `{0}`.'.format(
                    self.PATH_TO_BERT))
            bert_config = BertConfig.from_json_file(os.path.join(path_to_bert, 'bert_config.json'))
            bert_model = BertModel(config=bert_config, is_training=self.finetune_bert, input_ids=self.input_ids_,
                                   input_mask=self.input_mask_, token_type_ids=self.segment_ids_,
                                   use_one_hot_embeddings=False)
            sequence_output = bert_model.sequence_output
            tvars = tf.trainable_variables()
            init_checkpoint = os.path.join(self.PATH_TO_BERT, 'bert_model.ckpt')
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if self.verbose:
                bert_ner_logger.info('The BERT model has been loaded from a local drive.')
        self.additional_features_ = tf.placeholder(
            shape=(self.batch_size, self.max_seq_length, len(self.shapes_list_) + 4), dtype=tf.float32,
            name='additional_features'
        )
        if self.verbose:
            bert_ner_logger.info('Number of shapes is {0}.'.format(len(self.shapes_list_)))
        n_tags = len(self.classes_list_) * 2 + 1
        he_init = tf.contrib.layers.variance_scaling_initializer(seed=self.random_seed)
        glorot_init = tf.keras.initializers.glorot_uniform(seed=self.random_seed)
        sequence_lengths = tf.reduce_sum(self.input_mask_, axis=1)
        if self.lstm_units is None:
            if self.finetune_bert:
                self.logits_ = tf.layers.dense(tf.concat([sequence_output, self.additional_features_], axis=-1),
                                               n_tags, activation=None, kernel_regularizer=tf.nn.l2_loss,
                                               kernel_initializer=he_init, name='outputs_of_NER')
            else:
                sequence_output_stop = tf.stop_gradient(sequence_output)
                self.logits_ = tf.layers.dense(tf.concat([sequence_output_stop, self.additional_features_], axis=-1),
                                               n_tags, activation=None, kernel_regularizer=tf.nn.l2_loss,
                                               kernel_initializer=he_init, name='outputs_of_NER')
        else:
            if self.finetune_bert:
                with tf.name_scope('bilstm_layer'):
                    rnn_cell = tf.keras.layers.LSTMCell(units=self.lstm_units, activation=tf.nn.tanh, dropout=0.3,
                                                        recurrent_dropout=0.15, kernel_initializer=glorot_init)
                    rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(rnn_cell, return_sequences=True))
                    rnn_output = rnn_layer(tf.concat([sequence_output, self.additional_features_], axis=-1))
            else:
                sequence_output_stop = tf.stop_gradient(sequence_output)
                with tf.name_scope('bilstm_layer'):
                    rnn_cell = tf.keras.layers.LSTMCell(units=self.lstm_units, activation=tf.nn.tanh, dropout=0.3,
                                                        recurrent_dropout=0.15, kernel_initializer=glorot_init)
                    rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(rnn_cell, return_sequences=True))
                    rnn_output = rnn_layer(tf.concat([sequence_output_stop, self.additional_features_], axis=-1))
            self.logits_ = tf.layers.dense(rnn_output, n_tags, activation=None,
                                           kernel_regularizer=tf.nn.l2_loss,
                                           kernel_initializer=he_init, name='outputs_of_NER')
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits_, self.y_ph_,
                                                                              sequence_lengths)
        loss_tensor = -log_likelihood
        base_loss = tf.reduce_mean(loss_tensor)
        regularization_loss = self.l2_reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        final_loss = base_loss + regularization_loss
        self.transition_params_ = transition_params
        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.9, decay=0.9, epsilon=1e-10)
            if (self.lstm_units is None) or (self.clip_norm is None):
                train_op = optimizer.minimize(final_loss)
            else:
                grads_and_vars = optimizer.compute_gradients(final_loss)
                capped_gvs = [
                    (grad, var) if grad is None else (
                        tf.clip_by_norm(grad, self.clip_norm, name='grad_clipping_{0}'.format(idx + 1)),
                        var
                    )
                    for idx, (grad, var) in enumerate(grads_and_vars)
                ]
                train_op = optimizer.apply_gradients(capped_gvs)
        with tf.name_scope('eval'):
            seq_scores = tf.contrib.crf.crf_sequence_score(self.logits_, self.y_ph_, sequence_lengths,
                                                           self.transition_params_)
            seq_norm = tf.contrib.crf.crf_log_norm(self.logits_, sequence_lengths, self.transition_params_)
            accuracy = tf.reduce_mean(tf.cast(seq_scores, tf.float32) / tf.cast(seq_norm, tf.float32))
        return train_op, accuracy

    def finalize_model(self):
        if hasattr(self, 'input_ids_'):
            del self.input_ids_
        if hasattr(self, 'input_mask_'):
            del self.input_mask_
        if hasattr(self, 'segment_ids_'):
            del self.segment_ids_
        if hasattr(self, 'additional_features_'):
            del self.additional_features_
        if hasattr(self, 'y_ph_'):
            del self.y_ph_
        if hasattr(self, 'logits_'):
            del self.logits_
        if hasattr(self, 'transition_params_'):
            del self.transition_params_
        if hasattr(self, 'sess_'):
            for k in list(self.sess_.graph.get_all_collection_keys()):
                self.sess_.graph.clear_collection(k)
            self.sess_.close()
            del self.sess_
        tf.reset_default_graph()

    def save_model(self, file_name: str):
        saver = tf.train.Saver(allow_empty=True)
        saver.save(self.sess_, file_name)

    def load_model(self, file_name: str):
        saver = tf.train.Saver(allow_empty=True)
        saver.restore(self.sess_, file_name)

    @staticmethod
    def get_temp_model_name() -> str:
        return tempfile.NamedTemporaryFile(mode='w', suffix='bert_crf.ckpt').name

    @staticmethod
    def find_all_model_files(model_name: str) -> List[str]:
        model_files = []
        if os.path.isfile(model_name):
            model_files.append(model_name)
        dir_name = os.path.dirname(model_name)
        base_name = os.path.basename(model_name)
        for cur in filter(lambda it: it.lower().find(base_name.lower()) >= 0, os.listdir(dir_name)):
            model_files.append(os.path.join(dir_name, cur))
        return sorted(model_files)

    @staticmethod
    def check_path_to_bert(dir_name: str) -> bool:
        if not os.path.isdir(dir_name):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'vocab.txt')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.data-00000-of-00001')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.index')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.meta')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_config.json')):
            return False
        return True

    @staticmethod
    def check_params(**kwargs):
        if 'batch_size' not in kwargs:
            raise ValueError('`batch_size` is not specified!')
        if (not isinstance(kwargs['batch_size'], int)) and (not isinstance(kwargs['batch_size'], np.int32)) and \
                (not isinstance(kwargs['batch_size'], np.uint32)):
            raise ValueError('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['batch_size'])))
        if kwargs['batch_size'] < 1:
            raise ValueError('`batch_size` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['batch_size']))
        if 'lstm_units' not in kwargs:
            raise ValueError('`lstm_units` is not specified!')
        if kwargs['lstm_units'] is not None:
            if (not isinstance(kwargs['lstm_units'], int)) and (not isinstance(kwargs['lstm_units'], np.int32)) and \
                    (not isinstance(kwargs['lstm_units'], np.uint32)):
                raise ValueError('`lstm_units` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3), type(kwargs['lstm_units'])))
            if kwargs['lstm_units'] < 1:
                raise ValueError('`lstm_units` is wrong! Expected a positive integer value, '
                                 'but {0} is not positive.'.format(kwargs['lstm_units']))
        if 'lr' not in kwargs:
            raise ValueError('`lr` is not specified!')
        if (not isinstance(kwargs['lr'], float)) and (not isinstance(kwargs['lr'], np.float32)) and \
                (not isinstance(kwargs['lr'], np.float64)):
            raise ValueError('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(kwargs['lr'])))
        if kwargs['lr'] <= 0.0:
            raise ValueError('`lr` is wrong! Expected a positive floating-point value, '
                             'but {0} is not positive.'.format(kwargs['lr']))
        if 'l2_reg' not in kwargs:
            raise ValueError('`l2_reg` is not specified!')
        if (not isinstance(kwargs['l2_reg'], float)) and (not isinstance(kwargs['l2_reg'], np.float32)) and \
                (not isinstance(kwargs['l2_reg'], np.float64)):
            raise ValueError('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(kwargs['l2_reg'])))
        if kwargs['l2_reg'] < 0.0:
            raise ValueError('`l2_reg` is wrong! Expected a non-negative floating-point value, '
                             'but {0} is negative.'.format(kwargs['l2_reg']))
        if 'clip_norm' not in kwargs:
            raise ValueError('`clip_norm` is not specified!')
        if kwargs['clip_norm'] is not None:
            if (not isinstance(kwargs['clip_norm'], float)) and (not isinstance(kwargs['clip_norm'], np.float32)) and \
                    (not isinstance(kwargs['clip_norm'], np.float64)):
                raise ValueError('`clip_norm` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3.5), type(kwargs['clip_norm'])))
            if kwargs['clip_norm'] <= 0.0:
                raise ValueError('`clip_norm` is wrong! Expected a positive floating-point value, '
                                 'but {0} is not positive.'.format(kwargs['clip_norm']))
        if 'bert_hub_module_handle' not in kwargs:
            raise ValueError('`bert_hub_module_handle` is not specified!')
        if kwargs['bert_hub_module_handle'] is not None:
            if not isinstance(kwargs['bert_hub_module_handle'], str):
                raise ValueError('`bert_hub_module_handle` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type('abc'), type(kwargs['bert_hub_module_handle'])))
            if len(kwargs['bert_hub_module_handle']) < 1:
                raise ValueError('`bert_hub_module_handle` is wrong! Expected a nonepty string.')
        if 'finetune_bert' not in kwargs:
            raise ValueError('`finetune_bert` is not specified!')
        if (not isinstance(kwargs['finetune_bert'], int)) and (not isinstance(kwargs['finetune_bert'], np.int32)) and \
                (not isinstance(kwargs['finetune_bert'], np.uint32)) and \
                (not isinstance(kwargs['finetune_bert'], bool)) and (not isinstance(kwargs['finetune_bert'], np.bool)):
            raise ValueError('`finetune_bert` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['finetune_bert'])))
        if 'max_epochs' not in kwargs:
            raise ValueError('`max_epochs` is not specified!')
        if (not isinstance(kwargs['max_epochs'], int)) and (not isinstance(kwargs['max_epochs'], np.int32)) and \
                (not isinstance(kwargs['max_epochs'], np.uint32)):
            raise ValueError('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['max_epochs'])))
        if kwargs['max_epochs'] < 1:
            raise ValueError('`max_epochs` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['max_epochs']))
        if 'patience' not in kwargs:
            raise ValueError('`patience` is not specified!')
        if (not isinstance(kwargs['patience'], int)) and (not isinstance(kwargs['patience'], np.int32)) and \
                (not isinstance(kwargs['patience'], np.uint32)):
            raise ValueError('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['patience'])))
        if kwargs['patience'] < 1:
            raise ValueError('`patience` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['patience']))
        if 'random_seed' not in kwargs:
            raise ValueError('`random_seed` is not specified!')
        if kwargs['random_seed'] is not None:
            if (not isinstance(kwargs['random_seed'], int)) and (not isinstance(kwargs['random_seed'], np.int32)) and \
                    (not isinstance(kwargs['random_seed'], np.uint32)):
                raise ValueError('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3), type(kwargs['random_seed'])))
        if 'gpu_memory_frac' not in kwargs:
            raise ValueError('`gpu_memory_frac` is not specified!')
        if (not isinstance(kwargs['gpu_memory_frac'], float)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float32)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float64)):
            raise ValueError('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['gpu_memory_frac'])))
        if (kwargs['gpu_memory_frac'] <= 0.0) or (kwargs['gpu_memory_frac'] > 1.0):
            raise ValueError('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                             'but {0} is not proper.'.format(kwargs['gpu_memory_frac']))
        if 'max_seq_length' not in kwargs:
            raise ValueError('`max_seq_length` is not specified!')
        if (not isinstance(kwargs['max_seq_length'], int)) and \
                (not isinstance(kwargs['max_seq_length'], np.int32)) and \
                (not isinstance(kwargs['max_seq_length'], np.uint32)):
            raise ValueError('`max_seq_length` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['max_seq_length'])))
        if kwargs['max_seq_length'] < 1:
            raise ValueError('`max_seq_length` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['max_seq_length']))
        if 'validation_fraction' not in kwargs:
            raise ValueError('`validation_fraction` is not specified!')
        if (not isinstance(kwargs['validation_fraction'], float)) and \
                (not isinstance(kwargs['validation_fraction'], np.float32)) and \
                (not isinstance(kwargs['validation_fraction'], np.float64)):
            raise ValueError('`validation_fraction` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['validation_fraction'])))
        if kwargs['validation_fraction'] <= 0.0:
            raise ValueError('`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, '
                             'but {0} is not positive.'.format(kwargs['validation_fraction']))
        if kwargs['validation_fraction'] >= 1.0:
            raise ValueError('`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, '
                             'but {0} is not less than 1.0.'.format(kwargs['validation_fraction']))
        if 'verbose' not in kwargs:
            raise ValueError('`verbose` is not specified!')
        if (not isinstance(kwargs['verbose'], int)) and (not isinstance(kwargs['verbose'], np.int32)) and \
                (not isinstance(kwargs['verbose'], np.uint32)) and \
                (not isinstance(kwargs['verbose'], bool)) and (not isinstance(kwargs['verbose'], np.bool)):
            raise ValueError('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['verbose'])))

    @staticmethod
    def calculate_bounds_of_named_entities(bounds_of_tokens: List[Tuple[int, int]], classes_list: tuple,
                                           token_labels: List[int]) -> Dict[str, List[Tuple[int, int]]]:
        named_entities_for_text = dict()
        ne_start = -1
        ne_type = ''
        n_tokens = len(bounds_of_tokens)
        for token_idx in range(n_tokens):
            class_id = token_labels[token_idx]
            if (class_id > 0) and ((class_id - 1) // 2 < len(classes_list)):
                if ne_start < 0:
                    ne_start = token_idx
                    ne_type = classes_list[(class_id - 1) // 2]
                else:
                    if class_id % 2 == 0:
                        if ne_type in named_entities_for_text:
                            named_entities_for_text[ne_type].append(
                                (bounds_of_tokens[ne_start][0], bounds_of_tokens[token_idx - 1][1])
                            )
                        else:
                            named_entities_for_text[ne_type] = [
                                (bounds_of_tokens[ne_start][0], bounds_of_tokens[token_idx - 1][1])
                            ]
                        ne_start = token_idx
                        ne_type = classes_list[(class_id - 1) // 2]
                    else:
                        if classes_list[(class_id - 1) // 2] != ne_type:
                            if ne_type in named_entities_for_text:
                                named_entities_for_text[ne_type].append(
                                    (bounds_of_tokens[ne_start][0], bounds_of_tokens[token_idx - 1][1])
                                )
                            else:
                                named_entities_for_text[ne_type] = [
                                    (bounds_of_tokens[ne_start][0], bounds_of_tokens[token_idx - 1][1])
                                ]
                            ne_start = token_idx
                            ne_type = classes_list[(class_id - 1) // 2]
            else:
                if ne_start >= 0:
                    if ne_type in named_entities_for_text:
                        named_entities_for_text[ne_type].append(
                            (bounds_of_tokens[ne_start][0], bounds_of_tokens[token_idx - 1][1])
                        )
                    else:
                        named_entities_for_text[ne_type] = [
                            (bounds_of_tokens[ne_start][0], bounds_of_tokens[token_idx - 1][1])
                        ]
                    ne_start = -1
                    ne_type = ''
        if ne_start >= 0:
            if ne_type in named_entities_for_text:
                named_entities_for_text[ne_type].append(
                    (bounds_of_tokens[ne_start][0], bounds_of_tokens[-1][1])
                )
            else:
                named_entities_for_text[ne_type] = [
                    (bounds_of_tokens[ne_start][0], bounds_of_tokens[-1][1])
                ]
        return named_entities_for_text

