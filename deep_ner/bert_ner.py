import copy
import logging
import os
import random
import tempfile
import time
from typing import Dict, Union, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as tfhub
from bert.tokenization import FullTokenizer
from bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

from .quality import calculate_prediction_quality
from .dataset_splitting import split_dataset
from .udpipe_data import UNIVERSAL_DEPENDENCIES, UNIVERSAL_POS_TAGS, create_udpipe_pipeline, prepare_dependency_tag
from .utils import normalize_text


tf.logging.set_verbosity(tf.logging.ERROR)


bert_ner_logger = logging.getLogger(__name__)


class BERT_NER(BaseEstimator, ClassifierMixin):
    PATH_TO_BERT = None

    def __init__(self, udpipe_lang: str, use_nlp_features: bool = False,
                 use_shapes: bool = False, finetune_bert: bool = False,
                 bert_hub_module_handle: str = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                 batch_size: int=32, max_seq_length: int = 512, lr: float = 1e-4, lstm_units: Union[int, None] = 256,
                 l2_reg: float = 1e-5, clip_norm: Union[float, None] = 5.0, validation_fraction: float = 0.1,
                 max_epochs: int = 10, patience: int = 3, gpu_memory_frac: float = 1.0, verbose: bool = False,
                 random_seed: Union[int, None] = None):
        super().__init__()
        self.udpipe_lang = udpipe_lang
        self.use_nlp_features = use_nlp_features
        self.use_shapes = use_shapes
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

    def __del__(self):
        if hasattr(self, 'classes_list_'):
            del self.classes_list_
        if hasattr(self, 'shapes_list_'):
            del self.shapes_list_
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        if hasattr(self, 'nlp_'):
            del self.nlp_
        if hasattr(self, 'universal_pos_tags_dict_'):
            del self.universal_pos_tags_dict_
        if hasattr(self, 'universal_dependencies_dict_'):
            del self.universal_dependencies_dict_
        self.finalize_model()

    def fit(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array],
            validation_data: Union[None, Tuple[Union[list, tuple, np.array], Union[list, tuple, np.array]]]=None):
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, finetune_bert=self.finetune_bert,
            lstm_units=self.lstm_units, batch_size=self.batch_size, max_seq_length=self.max_seq_length, lr=self.lr,
            l2_reg=self.l2_reg, validation_fraction=self.validation_fraction, max_epochs=self.max_epochs,
            patience=self.patience, gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose,
            clip_norm=self.clip_norm, random_seed=self.random_seed, udpipe_lang=self.udpipe_lang,
            use_shapes=self.use_shapes, use_nlp_features=self.use_nlp_features
        )
        self.classes_list_ = self.check_Xy(X, 'X', y, 'y')
        if hasattr(self, 'shapes_list_'):
            del self.shapes_list_
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()
        self.update_random_seed()
        if validation_data is None:
            if self.validation_fraction > 0.0:
                train_index, test_index = split_dataset(y, self.validation_fraction, logger=bert_ner_logger)
                X_train_ = [X[idx] for idx in train_index]
                y_train_ = [y[idx] for idx in train_index]
                X_val_ = [X[idx] for idx in test_index]
                y_val_ = [y[idx] for idx in test_index]
                del train_index, test_index
            else:
                X_train_ = X
                y_train_ = y
                X_val_ = None
                y_val_ = None
        else:
            if (not isinstance(validation_data, tuple)) and (not isinstance(validation_data, list)):
                raise ValueError('')
            if len(validation_data) != 2:
                raise ValueError('')
            classes_list_for_validation = self.check_Xy(validation_data[0], 'X_val', validation_data[1], 'y_val')
            if not (set(classes_list_for_validation) <= set(self.classes_list_)):
                raise ValueError('')
            X_train_ = X
            y_train_ = y
            X_val_ = validation_data[0]
            y_val_ = validation_data[1]
        self.tokenizer_ = self.initialize_bert_tokenizer()
        X_train_tokenized, y_train_tokenized, self.shapes_list_, bounds_of_tokens_for_training = self.tokenize_all(
            X_train_, y_train_)
        if self.verbose:
            lengths_of_texts = []
            sum_of_lengths = 0
            for sample_idx in range(len(y_train_)):
                lengths_of_texts.append(sum(X_train_tokenized[1][sample_idx]))
                sum_of_lengths += lengths_of_texts[-1]
            mean_length = sum_of_lengths / float(len(lengths_of_texts))
            lengths_of_texts.sort()
            bert_ner_logger.info('Maximal length of text (in BPE): {0}'.format(max(lengths_of_texts)))
            bert_ner_logger.info('Mean length of text (in BPE): {0}'.format(mean_length))
            bert_ner_logger.info('Median length of text (in BPE): {0}'.format(
                lengths_of_texts[len(lengths_of_texts) // 2]))
        X_train_tokenized, y_train_tokenized, _ = self.extend_Xy(
            X_train_tokenized, bounds_of_tokens_for_training, y_train_tokenized, shuffle=True)
        del bounds_of_tokens_for_training
        if (X_val_ is not None) and (y_val_ is not None):
            X_val_tokenized, y_val_tokenized, _, bounds_of_tokens_for_validation = self.tokenize_all(
                X_val_, y_val_, shapes_vocabulary=self.shapes_list_)
            X_val_tokenized, y_val_tokenized, bounds_of_tokens_for_validation = self.extend_Xy(
                X_val_tokenized, bounds_of_tokens_for_validation, y_val_tokenized, shuffle=False)
        else:
            X_val_tokenized = None
            y_val_tokenized = None
            bounds_of_tokens_for_validation = None
        train_op, log_likelihood, logits_, transition_params_, input_mask_ = self.build_model()
        n_batches = int(np.ceil(X_train_tokenized[0].shape[0] / float(self.batch_size)))
        bounds_of_batches_for_training = []
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = min(batch_start + self.batch_size, X_train_tokenized[0].shape[0])
            bounds_of_batches_for_training.append((batch_start,  batch_end))
        if X_val_tokenized is None:
            bounds_of_batches_for_validation = None
        else:
            n_batches = int(np.ceil(X_val_tokenized[0].shape[0] / float(self.batch_size)))
            bounds_of_batches_for_validation = []
            for iteration in range(n_batches):
                batch_start = iteration * self.batch_size
                batch_end = min(batch_start + self.batch_size, X_val_tokenized[0].shape[0])
                bounds_of_batches_for_validation.append((batch_start, batch_end))
        init = tf.global_variables_initializer()
        init.run(session=self.sess_)
        tmp_model_name = self.get_temp_model_name()
        if self.verbose:
            if X_val_tokenized is None:
                bert_ner_logger.info('Epoch   Log-likelihood')
        n_epochs_without_improving = 0
        try:
            best_acc = None
            for epoch in range(self.max_epochs):
                random.shuffle(bounds_of_batches_for_training)
                feed_dict_for_batch = None
                for cur_batch in bounds_of_batches_for_training:
                    X_batch = [X_train_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                               for channel_idx in range(len(X_train_tokenized))]
                    y_batch = y_train_tokenized[cur_batch[0]:cur_batch[1]]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                    self.sess_.run(train_op, feed_dict=feed_dict_for_batch)
                acc_train = log_likelihood.eval(feed_dict=feed_dict_for_batch, session=self.sess_)
                if bounds_of_batches_for_validation is not None:
                    acc_test = 0.0
                    y_pred = []
                    for cur_batch in bounds_of_batches_for_validation:
                        X_batch = [X_val_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                                   for channel_idx in range(len(X_val_tokenized))]
                        y_batch = y_val_tokenized[cur_batch[0]:cur_batch[1]]
                        feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                        acc_test_, logits, trans_params, mask = self.sess_.run(
                            [log_likelihood, logits_, transition_params_, input_mask_],
                            feed_dict=feed_dict_for_batch
                        )
                        acc_test += self.batch_size * acc_test_
                        sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
                        for logit, sequence_length in zip(logits, sequence_lengths):
                            logit = logit[:int(sequence_length)]
                            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                            y_pred += [viterbi_seq]
                    acc_test /= float(X_val_tokenized[0].shape[0])
                    if self.verbose:
                        bert_ner_logger.info('Epoch {0}'.format(epoch))
                        bert_ner_logger.info('  Train log-likelihood: {0: 10.8f}'.format(acc_train))
                        bert_ner_logger.info('  Val. log-likelihood:  {0: 10.8f}'.format(acc_test))
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
                        bert_ner_logger.info('{0:>5}   {1:>14.8f}'.format(epoch, acc_train))
                if n_epochs_without_improving >= self.patience:
                    if self.verbose:
                        bert_ner_logger.info('Epoch %05d: early stopping' % (epoch + 1))
                    break
            if best_acc is not None:
                self.finalize_model()
                self.load_model(tmp_model_name)
                if self.verbose and (bounds_of_batches_for_validation is not None):
                    acc_test = 0.0
                    y_pred = []
                    for cur_batch in bounds_of_batches_for_validation:
                        X_batch = [X_val_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                                   for channel_idx in range(len(X_val_tokenized))]
                        y_batch = y_val_tokenized[cur_batch[0]:cur_batch[1]]
                        feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                        acc_test_, logits, trans_params, mask = self.sess_.run(
                            ['eval/Mean:0', 'outputs_of_NER/BiasAdd:0', 'transitions:0', 'input_mask:0'],
                            feed_dict=feed_dict_for_batch
                        )
                        acc_test += self.batch_size * acc_test_
                        sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
                        for logit, sequence_length in zip(logits, sequence_lengths):
                            logit = logit[:int(sequence_length)]
                            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                            y_pred += [viterbi_seq]
                    acc_test /= float(X_val_tokenized[0].shape[0])
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
                    bert_ner_logger.info('Best val. log-likelihood is {0:>10.8f}'.format(acc_test))
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
            random_seed=self.random_seed, clip_norm=self.clip_norm, udpipe_lang=self.udpipe_lang,
            use_nlp_features=self.use_nlp_features, use_shapes=self.use_shapes
        )
        self.check_X(X, 'X')
        self.is_fitted()
        X_tokenized, _, _, bounds_of_tokens = self.tokenize_all(X, shapes_vocabulary=self.shapes_list_)
        n_samples = X_tokenized[0].shape[0]
        X_tokenized, bounds_of_tokens = self.extend_Xy(X_tokenized, bounds_of_tokens)
        n_batches = X_tokenized[0].shape[0] // self.batch_size
        bounds_of_batches = []
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = batch_start + self.batch_size
            bounds_of_batches.append((batch_start, batch_end))
        y_pred = []
        for cur_batch in bounds_of_batches:
            feed_dict = self.fill_feed_dict(
                [
                    X_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                    for channel_idx in range(len(X_tokenized))
                ]
            )
            logits, trans_params, mask = self.sess_.run(
                ['outputs_of_NER/BiasAdd:0', 'transitions:0', 'input_mask:0'],
                feed_dict=feed_dict
            )
            sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:int(sequence_length)]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                y_pred += [viterbi_seq]
        del bounds_of_batches
        recognized_entities_in_texts = []
        for sample_idx, labels_in_text in enumerate(y_pred[0:n_samples]):
            n_tokens = len(labels_in_text)
            new_entities = self.calculate_bounds_of_named_entities(bounds_of_tokens[sample_idx], self.classes_list_,
                                                                   labels_in_text[1:(n_tokens - 1)])
            recognized_entities_in_texts.append(new_entities)
        return recognized_entities_in_texts

    def is_fitted(self):
        check_is_fitted(self, ['classes_list_', 'shapes_list_', 'tokenizer_', 'sess_'])

    def score(self, X, y, sample_weight=None) -> float:
        y_pred = self.predict(X)
        return calculate_prediction_quality(y, y_pred, self.classes_list_)[0]

    def fit_predict(self, X: Union[list, tuple, np.array],  y: Union[list, tuple, np.array], **kwargs):
        return self.fit(X, y).predict(X)

    def fill_feed_dict(self, X: List[np.array], y: np.array=None) -> dict:
        if self.use_nlp_features or self.use_shapes:
            assert len(X) == 4
        else:
            assert len(X) == 3
        assert len(X[0]) == self.batch_size
        if self.use_nlp_features or self.use_shapes:
            feed_dict = {
                ph: x for ph, x in zip(['input_ids:0', 'input_mask:0', 'segment_ids:0', 'additional_features:0'], X)
            }
        else:
            feed_dict = {
                ph: x for ph, x in zip(['input_ids:0', 'input_mask:0', 'segment_ids:0'], X)
            }
        if y is not None:
            feed_dict['y_ph:0'] = y
        return feed_dict

    def extend_Xy(self, X: List[np.array], bounds_of_tokens: np.ndarray, y: np.array=None, shuffle: bool=False) -> \
            Union[Tuple[List[np.ndarray], np.ndarray], Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:
        n_samples = X[0].shape[0]
        n_extend = n_samples % self.batch_size
        if n_extend == 0:
            if y is None:
                return X, bounds_of_tokens
            return X, y, bounds_of_tokens
        n_extend = self.batch_size - n_extend
        X_ext = [
            np.concatenate(
                (
                    X[idx],
                    np.full(
                        shape=((n_extend, self.max_seq_length) if len(X[idx].shape) == 2 else
                               (n_extend, self.max_seq_length, X[idx].shape[2])),
                        fill_value=X[idx][-1],
                        dtype=X[idx].dtype
                    )
                )
            )
            for idx in range(len(X))
        ]
        bounds_of_tokens_ext = bounds_of_tokens.tolist()
        bounds_of_tokens_for_last_text = bounds_of_tokens[len(bounds_of_tokens) - 1]
        bounds_of_tokens_ext += [bounds_of_tokens_for_last_text for _ in range(n_extend)]
        bounds_of_tokens_ext = np.array(bounds_of_tokens_ext, dtype=object)
        del bounds_of_tokens_for_last_text
        if y is None:
            if shuffle:
                indices = np.arange(0, n_samples + n_extend, 1, dtype=np.int32)
                np.random.shuffle(indices)
                return [X_ext[idx][indices] for idx in range(len(X_ext))], bounds_of_tokens_ext[indices]
            return X_ext, bounds_of_tokens_ext
        y_ext = np.concatenate(
            (
                y,
                np.full(shape=(n_extend, self.max_seq_length), fill_value=y[-1], dtype=y.dtype)
            )
        )
        if shuffle:
            indices = np.arange(0, n_samples + n_extend, 1, dtype=np.int32)
            return [X_ext[idx][indices] for idx in range(len(X_ext))], y_ext[indices], bounds_of_tokens_ext[indices]
        return X_ext, y_ext, bounds_of_tokens_ext

    def tokenize_all(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array] = None,
                     shapes_vocabulary: Union[tuple, None] = None) -> Tuple[List[np.ndarray], Union[np.ndarray, None],
                                                                            tuple, np.ndarray]:
        if shapes_vocabulary is not None:
            if len(shapes_vocabulary) < 4:
                raise ValueError('Shapes vocabulary is wrong!')
            if shapes_vocabulary[-1] != '[UNK]':
                raise ValueError('Shapes vocabulary is wrong!')
            if shapes_vocabulary[-2] != '[SEP]':
                raise ValueError('Shapes vocabulary is wrong!')
            if shapes_vocabulary[-3] != '[CLS]':
                raise ValueError('Shapes vocabulary is wrong!')
        if not hasattr(self, 'universal_pos_tags_dict_'):
            self.universal_pos_tags_dict_ = dict(zip(UNIVERSAL_POS_TAGS, range(len(UNIVERSAL_POS_TAGS))))
        if not hasattr(self, 'universal_dependencies_dict_'):
            self.universal_dependencies_dict_ = dict(zip(UNIVERSAL_DEPENDENCIES, range(len(UNIVERSAL_DEPENDENCIES))))
        if self.use_nlp_features or self.use_shapes:
            n_additonal_features = 4
            if self.use_nlp_features:
                n_additonal_features += len(self.universal_dependencies_dict_) + len(self.universal_pos_tags_dict_)
            X_tokenized = [
                np.zeros((len(X), self.max_seq_length), dtype=np.int32),
                np.zeros((len(X), self.max_seq_length), dtype=np.int32),
                np.zeros((len(X), self.max_seq_length), dtype=np.int32),
                np.zeros((len(X), self.max_seq_length, n_additonal_features), dtype=np.float32)
            ]
        else:
            X_tokenized = [
                np.zeros((len(X), self.max_seq_length), dtype=np.int32),
                np.zeros((len(X), self.max_seq_length), dtype=np.int32),
                np.zeros((len(X), self.max_seq_length), dtype=np.int32)
            ]
        all_tokenized_texts = []
        bounds_of_tokens = []
        n_samples = len(X)
        shapes = []
        shapes_dict = dict()
        for sample_idx in range(n_samples):
            source_text = X[sample_idx]
            tokenized_text = []
            bounds_of_tokens_for_text = []
            start_pos = 0
            if not hasattr(self, 'nlp_'):
                self.nlp_ = create_udpipe_pipeline(self.udpipe_lang)
            normalized_text = normalize_text(source_text)
            spacy_doc = self.nlp_(normalized_text)
            pos_tags = []
            dependencies = []
            for spacy_token in spacy_doc:
                cur_word = spacy_token.text
                parts_of_cur_word = list(filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), cur_word.split())
                ))
                if len(parts_of_cur_word) == 0:
                    raise ValueError('Text `{0}` cannot be tokenized!'.format(X[sample_idx]))
                for cur_word_part in parts_of_cur_word:
                    found_idx_1 = normalized_text[start_pos:].find(cur_word_part)
                    if found_idx_1 < 0:
                        raise ValueError(
                            'Text `{0}` cannot be tokenized!'.format(X[sample_idx]))
                    cur_word_ = cur_word_part.lower() if self.tokenizer_.basic_tokenizer.do_lower_case else cur_word_part
                    subwords = self.tokenizer_.tokenize(cur_word_)
                    if '[UNK]' in subwords:
                        tokenized_text.append('[UNK]')
                        bounds_of_tokens_for_text.append((start_pos + found_idx_1,
                                                          start_pos + found_idx_1 + len(cur_word_part)))
                        pos_tags.append(spacy_token.pos_)
                        dependencies.append(spacy_token.dep_)
                    else:
                        start_pos_2 = 0
                        for cur_subword in subwords:
                            if cur_subword.startswith('##'):
                                subword_len = len(cur_subword) - 2
                                found_idx_2 = cur_word_[start_pos_2:].find(cur_subword[2:])
                            else:
                                subword_len = len(cur_subword)
                                found_idx_2 = cur_word_[start_pos_2:].find(cur_subword)
                            if found_idx_2 < 0:
                                raise ValueError(
                                    'Text `{0}` cannot be tokenized!'.format(X[sample_idx]))
                            tokenized_text.append(cur_subword)
                            bounds_of_tokens_for_text.append(
                                (
                                    start_pos + found_idx_1 + start_pos_2 + found_idx_2,
                                    start_pos + found_idx_1 + start_pos_2 + found_idx_2 + subword_len
                                )
                            )
                            start_pos_2 += (found_idx_2 + subword_len)
                            pos_tags.append(spacy_token.pos_)
                            dependencies.append(spacy_token.dep_)
                    start_pos += (found_idx_1 + len(cur_word_part))
            del spacy_doc
            if len(tokenized_text) > (self.max_seq_length - 2):
                tokenized_text = tokenized_text[:(self.max_seq_length - 2)]
                bounds_of_tokens_for_text = bounds_of_tokens_for_text[:(self.max_seq_length - 2)]
            shapes_of_text = [self.get_shape_of_string(normalized_text[cur[0]:cur[1]]) for cur in bounds_of_tokens_for_text]
            if shapes_vocabulary is None:
                for cur_shape in shapes_of_text:
                    if cur_shape != '[UNK]':
                        shapes_dict[cur_shape] = shapes_dict.get(cur_shape, 0) + 1
            shapes.append(shapes_of_text)
            all_tokenized_texts.append(copy.copy(tokenized_text))
            bounds_of_tokens.append(copy.deepcopy(bounds_of_tokens_for_text))
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            pos_tags = [''] + pos_tags + ['']
            dependencies = [''] + dependencies + ['']
            token_IDs = self.tokenizer_.convert_tokens_to_ids(tokenized_text)
            for token_idx in range(len(token_IDs)):
                X_tokenized[0][sample_idx][token_idx] = token_IDs[token_idx]
                X_tokenized[1][sample_idx][token_idx] = 1
                if self.use_nlp_features or self.use_shapes:
                    X_tokenized[3][sample_idx][token_idx][self.get_subword_ID(tokenized_text[token_idx])] = 1.0
                    if self.use_nlp_features:
                        pos_tag_id = self.universal_pos_tags_dict_.get(pos_tags[token_idx], -1)
                        if pos_tag_id >= 0:
                            X_tokenized[3][sample_idx][token_idx][pos_tag_id + 4] = 1.0
                        elif len(pos_tags[token_idx]) > 0:
                            raise ValueError('Part-of-speech tag `{0}` is unknown!'.format(pos_tags[token_idx]))
                        ok = False
                        for dependency_tag in prepare_dependency_tag(dependencies[token_idx]):
                            dependency_id = self.universal_dependencies_dict_.get(dependency_tag, -1)
                            if dependency_id >= 0:
                                X_tokenized[3][sample_idx][token_idx][dependency_id + 4 + len(UNIVERSAL_POS_TAGS)] = 1.0
                                ok = True
                        if (not ok) and (len(dependencies[token_idx]) > 0):
                            raise ValueError('Dependency tag `{0}` is unknown!'.format(dependencies[token_idx]))
            del bounds_of_tokens_for_text, tokenized_text, token_IDs, pos_tags, dependencies
        if y is None:
            y_tokenized = None
        else:
            y_tokenized = np.empty((len(y), self.max_seq_length), dtype=np.int32)
            for sample_idx in range(n_samples):
                source_text = X[sample_idx]
                normalized_text = normalize_text(source_text)
                tokenized_text = all_tokenized_texts[sample_idx]
                bounds_of_tokens_for_text = bounds_of_tokens[sample_idx]
                indices_of_named_entities, labels_IDs = self.calculate_indices_of_named_entities(
                    normalized_text, self.classes_list_, y[sample_idx])
                y_tokenized[sample_idx] = self.detect_token_labels(
                    tokenized_text, bounds_of_tokens_for_text, indices_of_named_entities, labels_IDs,
                    self.max_seq_length
                )
        if shapes_vocabulary is None:
            shapes_vocabulary_ = list(map(
                lambda it2: it2[0],
                filter(
                    lambda it1: (it1[1] >= 3) and (it1[0] not in {'[CLS]', '[SEP]', '[UNK]'}),
                    [(cur_shape, shapes_dict[cur_shape]) for cur_shape in sorted(list(shapes_dict.keys()))]
                )
            ))
            shapes_vocabulary_ += ['[CLS]', '[SEP]', '[UNK]']
            shapes_vocabulary_ = tuple(shapes_vocabulary_)
        else:
            shapes_vocabulary_ = shapes_vocabulary
        shapes_ = np.zeros((len(X), self.max_seq_length, len(shapes_vocabulary_)), dtype=np.float32)
        for sample_idx in range(n_samples):
            shape_ID = shapes_vocabulary_.index('[CLS]')
            shapes_[sample_idx][0][shape_ID] = 1.0
            for token_idx, cur_shape in enumerate(shapes[sample_idx]):
                if cur_shape in shapes_vocabulary_:
                    shape_ID = shapes_vocabulary_.index(cur_shape)
                else:
                    shape_ID = len(shapes_vocabulary_) - 1
                shapes_[sample_idx][token_idx + 1][shape_ID] = 1.0
            shape_ID = shapes_vocabulary_.index('[SEP]')
            shapes_[sample_idx][len(shapes[sample_idx]) + 1][shape_ID] = 1.0
        if self.use_shapes:
            X_tokenized[3] = np.concatenate((X_tokenized[3], shapes_), axis=-1)
        del shapes_, shapes
        return X_tokenized, (None if y is None else np.array(y_tokenized)), shapes_vocabulary_, \
               np.array(bounds_of_tokens, dtype=np.object)

    def get_params(self, deep=True) -> dict:
        return {'bert_hub_module_handle': self.bert_hub_module_handle, 'finetune_bert': self.finetune_bert,
                'lstm_units': self.lstm_units, 'batch_size': self.batch_size, 'max_seq_length': self.max_seq_length,
                'lr': self.lr, 'l2_reg': self.l2_reg, 'clip_norm': self.clip_norm, 'max_epochs': self.max_epochs,
                'patience': self.patience, 'validation_fraction': self.validation_fraction,
                'gpu_memory_frac': self.gpu_memory_frac, 'verbose': self.verbose, 'random_seed': self.random_seed,
                'udpipe_lang': self.udpipe_lang, 'use_shapes': self.use_shapes,
                'use_nlp_features': self.use_nlp_features}

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
            verbose=self.verbose, random_seed=self.random_seed, udpipe_lang=self.udpipe_lang,
            use_shapes=self.use_shapes, use_nlp_features=self.use_nlp_features
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_list_ = self.classes_list_
            result.shapes_list_ = self.shapes_list_
            result.tokenizer_ = self.tokenizer_
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
            verbose=self.verbose, random_seed=self.random_seed, udpipe_lang=self.udpipe_lang,
            use_shapes=self.use_shapes, use_nlp_features=self.use_nlp_features
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_list_ = self.classes_list_
            result.shapes_list_ = self.shapes_list_
            result.tokenizer_ = self.tokenizer_
            result.sess_ = self.sess_
        return result

    def __getstate__(self):
        return self.dump_all()

    def __setstate__(self, state: dict):
        self.load_all(state)

    def update_random_seed(self):
        if self.random_seed is None:
            self.random_seed = int(round(time.time()))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_random_seed(self.random_seed)

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
            params['tokenizer_'] = copy.deepcopy(self.tokenizer_)
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
            self.classes_list_ = copy.copy(new_params['classes_list_'])
            self.shapes_list_ = copy.copy(new_params['shapes_list_'])
            self.tokenizer_ = copy.deepcopy(new_params['tokenizer_'])
            self.update_random_seed()
            try:
                for idx in range(len(model_files)):
                    with open(tmp_file_names[idx], 'wb') as fp:
                        fp.write(new_params['model.' + model_files[idx]])
                self.load_model(os.path.join(tmp_dir_name, new_params['model_name_']))
            finally:
                for cur in tmp_file_names:
                    if os.path.isfile(cur):
                        os.remove(cur)
        else:
            self.set_params(**new_params)
        return self

    def initialize_bert_tokenizer(self) -> FullTokenizer:
        if self.bert_hub_module_handle is not None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
            config.gpu_options.allow_growth = True
            self.sess_ = tf.Session(config=config)
            bert_module = tfhub.Module(self.bert_hub_module_handle, trainable=True)
            tokenization_info = bert_module(signature='tokenization_info', as_dict=True)
            vocab_file, do_lower_case = self.sess_.run([tokenization_info['vocab_file'],
                                                        tokenization_info['do_lower_case']])
            tokenizer_ = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
            if hasattr(self, 'sess_'):
                for k in list(self.sess_.graph.get_all_collection_keys()):
                    self.sess_.graph.clear_collection(k)
                self.sess_.close()
                del self.sess_
            tf.reset_default_graph()
        else:
            if self.PATH_TO_BERT is None:
                raise ValueError('Path to the BERT model is not defined!')
            path_to_bert = os.path.normpath(self.PATH_TO_BERT)
            if not self.check_path_to_bert(path_to_bert):
                raise ValueError('`path_to_bert` is wrong! There are no BERT files into the directory `{0}`.'.format(
                    self.PATH_TO_BERT))
            if (os.path.basename(path_to_bert).find('_uncased_') >= 0) or \
                    (os.path.basename(path_to_bert).find('uncased_') >= 0):
                do_lower_case = True
            else:
                if os.path.basename(path_to_bert).find('_cased_') >= 0 or \
                        os.path.basename(path_to_bert).startswith('cased_'):
                    do_lower_case = False
                else:
                    do_lower_case = None
            if do_lower_case is None:
                raise ValueError('`{0}` is bad path to the BERT model, because a tokenization mode (lower case or no) '
                                 'cannot be detected.'.format(path_to_bert))
            tokenizer_ = FullTokenizer(vocab_file=os.path.join(path_to_bert, 'vocab.txt'), do_lower_case=do_lower_case)
            if self.verbose:
                bert_ner_logger.info('The BERT tokenizer has been loaded from a local drive. '
                                     '`do_lower_case` is {0}.'.format(do_lower_case))
        return tokenizer_

    def build_model(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
        config.gpu_options.allow_growth = True
        self.sess_ = tf.Session(config=config)
        input_ids = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32,
                                   name='input_ids')
        input_mask = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32,
                                    name='input_mask')
        segment_ids = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32,
                                     name='segment_ids')
        y_ph = tf.placeholder(shape=(self.batch_size, self.max_seq_length), dtype=tf.int32, name='y_ph')
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
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
            bert_model = BertModel(config=bert_config, is_training=self.finetune_bert, input_ids=input_ids,
                                   input_mask=input_mask, token_type_ids=segment_ids,
                                   use_one_hot_embeddings=False)
            sequence_output = bert_model.sequence_output
            tvars = tf.trainable_variables()
            init_checkpoint = os.path.join(self.PATH_TO_BERT, 'bert_model.ckpt')
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if self.verbose:
                bert_ner_logger.info('The BERT model has been loaded from a local drive.')
        if self.use_nlp_features or self.use_shapes:
            n_additional_features = 4
            if self.use_nlp_features:
                n_additional_features += (len(UNIVERSAL_POS_TAGS) + len(UNIVERSAL_DEPENDENCIES))
            if self.use_shapes:
                n_additional_features += len(self.shapes_list_)
            additional_features = tf.placeholder(
                shape=(self.batch_size, self.max_seq_length, n_additional_features), dtype=tf.float32,
                name='additional_features'
            )
            if self.verbose and self.use_shapes:
                bert_ner_logger.info('Number of shapes is {0}.'.format(len(self.shapes_list_)))
        else:
            additional_features = None
        n_tags = len(self.classes_list_) * 2 + 1
        he_init = tf.contrib.layers.variance_scaling_initializer(seed=self.random_seed)
        glorot_init = tf.keras.initializers.glorot_uniform(seed=self.random_seed)
        sequence_lengths = tf.reduce_sum(input_mask, axis=1)
        if self.lstm_units is None:
            if self.finetune_bert:
                logits = tf.layers.dense(
                    (tf.concat([sequence_output, additional_features], axis=-1) if additional_features is not None
                     else sequence_output),
                    n_tags, activation=None, kernel_regularizer=tf.nn.l2_loss,
                    kernel_initializer=he_init, name='outputs_of_NER')
            else:
                sequence_output_stop = tf.stop_gradient(sequence_output)
                logits = tf.layers.dense(
                    (tf.concat([sequence_output_stop, additional_features], axis=-1) if additional_features is not None
                     else sequence_output_stop),
                    n_tags, activation=None, kernel_regularizer=tf.nn.l2_loss,
                    kernel_initializer=he_init, name='outputs_of_NER')
        else:
            if self.finetune_bert:
                with tf.name_scope('bilstm_layer'):
                    rnn_cell = tf.keras.layers.LSTMCell(units=self.lstm_units, activation=tf.nn.tanh, dropout=0.3,
                                                        recurrent_dropout=0.0, kernel_initializer=glorot_init)
                    rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(rnn_cell, return_sequences=True))
                    if additional_features is not None:
                        rnn_output = rnn_layer(tf.concat([sequence_output, additional_features], axis=-1))
                    else:
                        rnn_output = rnn_layer(sequence_output)
            else:
                sequence_output_stop = tf.stop_gradient(sequence_output)
                with tf.name_scope('bilstm_layer'):
                    rnn_cell = tf.keras.layers.LSTMCell(units=self.lstm_units, activation=tf.nn.tanh, dropout=0.3,
                                                        recurrent_dropout=0.0, kernel_initializer=glorot_init)
                    rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(rnn_cell, return_sequences=True))
                    if additional_features is not None:
                        rnn_output = rnn_layer(tf.concat([sequence_output_stop, additional_features], axis=-1))
                    else:
                        rnn_output = rnn_layer(sequence_output_stop)
            logits = tf.layers.dense(rnn_output, n_tags, activation=None, kernel_regularizer=tf.nn.l2_loss,
                                     kernel_initializer=he_init, name='outputs_of_NER')
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, y_ph, sequence_lengths)
        loss_tensor = -log_likelihood
        base_loss = tf.reduce_mean(loss_tensor)
        regularization_loss = self.l2_reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        final_loss = base_loss + regularization_loss
        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.9, decay=0.9, epsilon=1e-10)
            if (self.lstm_units is None) or (self.clip_norm is None):
                train_op = optimizer.minimize(final_loss)
            else:
                grads_and_vars = optimizer.compute_gradients(final_loss)
                capped_gvs = [
                    (grad, var) if grad is None else (
                        tf.clip_by_norm(grad, 5.0, name='grad_clipping_{0}'.format(idx + 1)),
                        var
                    )
                    for idx, (grad, var) in enumerate(grads_and_vars)
                ]
                train_op = optimizer.apply_gradients(capped_gvs)
        with tf.name_scope('eval'):
            log_likelihood_eval_, _ = tf.contrib.crf.crf_log_likelihood(logits, y_ph, sequence_lengths,
                                                                        transition_params)
            seq_norm_eval = tf.contrib.crf.crf_log_norm(logits, sequence_lengths, transition_params)
            log_likelihood_eval = tf.reduce_mean(tf.cast(log_likelihood_eval_, tf.float32) /
                                                 tf.cast(seq_norm_eval, tf.float32))
        return train_op, log_likelihood_eval, logits, transition_params, input_mask

    def finalize_model(self):
        if hasattr(self, 'sess_'):
            for k in list(self.sess_.graph.get_all_collection_keys()):
                self.sess_.graph.clear_collection(k)
            self.sess_.close()
            del self.sess_
        tf.reset_default_graph()

    def save_model(self, file_name: str):
        saver = tf.train.Saver()
        saver.save(self.sess_, file_name)

    def load_model(self, file_name: str):
        if not hasattr(self, 'sess_'):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
            config.gpu_options.allow_growth = True
            self.sess_ = tf.Session(config=config)
        saver = tf.train.import_meta_graph(file_name + '.meta', clear_devices=True)
        saver.restore(self.sess_, file_name)

    @staticmethod
    def get_temp_model_name() -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='bert_crf.ckpt', delete=True) as fp:
            res = fp.name
        return res

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
        if 'udpipe_lang' not in kwargs:
            raise ValueError('`udpipe_lang` is not specified!')
        if not isinstance(kwargs['udpipe_lang'], str):
            raise ValueError('`udpipe_lang` is wrong! Expected `{0}`, got `{1}`.'.format(
                type('abc'), type(kwargs['udpipe_lang'])))
        if len(kwargs['udpipe_lang']) < 1:
            raise ValueError('`udpipe_lang` is wrong! Expected a nonepty string.')
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
        if kwargs['validation_fraction'] < 0.0:
            raise ValueError('`validation_fraction` is wrong! Expected a positive floating-point value greater than '
                             'or equal to 0.0, but {0} is not positive.'.format(kwargs['validation_fraction']))
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
        if 'use_shapes' not in kwargs:
            raise ValueError('`use_shapes` is not specified!')
        if (not isinstance(kwargs['use_shapes'], int)) and \
                (not isinstance(kwargs['use_shapes'], np.int32)) and \
                (not isinstance(kwargs['use_shapes'], np.uint32)) and \
                (not isinstance(kwargs['use_shapes'], bool)) and \
                (not isinstance(kwargs['use_shapes'], np.bool)):
            raise ValueError('`use_shapes` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['use_shapes'])))
        if 'use_nlp_features' not in kwargs:
            raise ValueError('`use_nlp_features` is not specified!')
        if (not isinstance(kwargs['use_nlp_features'], int)) and \
                (not isinstance(kwargs['use_nlp_features'], np.int32)) and \
                (not isinstance(kwargs['use_nlp_features'], np.uint32)) and \
                (not isinstance(kwargs['use_nlp_features'], bool)) and \
                (not isinstance(kwargs['use_nlp_features'], np.bool)):
            raise ValueError('`use_nlp_features` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['use_nlp_features'])))

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

    @staticmethod
    def calculate_indices_of_named_entities(source_text: str, classes_list: tuple,
                                            named_entities: Dict[str, List[tuple]]) -> \
            Tuple[np.ndarray, Dict[int, int]]:
        indices_of_named_entities = np.zeros((len(source_text),), dtype=np.int32)
        labels_to_classes = dict()
        label_ID = 1
        for ne_type in sorted(list(named_entities.keys())):
            class_id = classes_list.index(ne_type) + 1
            for ne_bounds in named_entities[ne_type]:
                for char_idx in range(ne_bounds[0], ne_bounds[1]):
                    indices_of_named_entities[char_idx] = label_ID
                labels_to_classes[label_ID] = class_id
                label_ID += 1
        return indices_of_named_entities, labels_to_classes

    @staticmethod
    def detect_token_labels(tokens: List[str], bounds_of_tokens: List[tuple], indices_of_named_entities: np.ndarray,
                            label_ids: dict, max_seq_length: int) -> np.ndarray:
        res = np.zeros((max_seq_length,), dtype=np.int32)
        n = min(len(bounds_of_tokens), max_seq_length - 2)
        for token_idx, cur in enumerate(bounds_of_tokens[:n]):
            distr = np.zeros((len(label_ids) + 1,), dtype=np.int32)
            for char_idx in range(cur[0], cur[1]):
                distr[indices_of_named_entities[char_idx]] += 1
            label_id = distr.argmax()
            if label_id > 0:
                res[token_idx + 1] = label_id
            del distr
        prev_label_id = 0
        for token_idx in range(max_seq_length):
            if token_idx >= n:
                break
            cur_label_id = res[token_idx + 1]
            if cur_label_id != prev_label_id:
                if tokens[token_idx].startswith('##'):
                    if prev_label_id > 0:
                        res[token_idx +1] = prev_label_id
                        cur_label_id = prev_label_id
                    else:
                        token_idx_ = token_idx - 1
                        while token_idx_ >= 0:
                            res[token_idx_ + 1] = cur_label_id
                            if not tokens[token_idx_].startswith('##'):
                                break
                            token_idx_ -= 1
            prev_label_id = cur_label_id
        prev_label_id = 0
        for token_idx in range(max_seq_length):
            cur_label_id = res[token_idx]
            if cur_label_id > 0:
                ne_id = label_ids[res[token_idx]]
                if cur_label_id == prev_label_id:
                    res[token_idx] = ne_id * 2 - 1
                else:
                    res[token_idx] = ne_id * 2
            prev_label_id = cur_label_id
        return res

    @staticmethod
    def get_subword_ID(subword: str) -> int:
        if subword == '[CLS]':
            return 0
        if subword == '[SEP]':
            return 1
        if subword.startswith('##'):
            return 2
        return 3

    @staticmethod
    def get_shape_of_string(src: str) -> str:
        if src in {'[UNK]', '[PAD]', '[CLS]', '[SEP]'}:
            return src
        shape = ''
        for idx in (range(2, len(src)) if src.startswith('##') else range(len(src))):
            if src[idx] in {'_', chr(11791)}:
                new_char = '_'
            elif src[idx].isalpha():
                if src[idx].isupper():
                    new_char = 'A'
                else:
                    new_char = 'a'
            elif src[idx].isdigit():
                new_char = 'D'
            elif src[idx] in {'.', ',', ':', ';', '-', '+', '!', '?', '#', '@', '$', '&', '=', '^', '`', '~', '*', '/',
                              '\\', '(', ')', '[', ']', '{', '}', "'", '"', '|', '<', '>'}:
                new_char = 'P'
            elif src[idx] in {chr(8213), chr(8212), chr(8211), chr(8210), chr(8209), chr(8208), chr(11834), chr(173),
                              chr(8722), chr(8259)}:
                new_char = '-'
            elif src[idx] in {chr(8220), chr(8221), chr(11842), chr(171), chr(187), chr(128631), chr(128630),
                              chr(128632), chr(12318), chr(12317), chr(12319)}:
                new_char = '"'
            elif src[idx] in {chr(39), chr(8216), chr(8217), chr(8218)}:
                new_char = "'"
            else:
                new_char = 'U'
            if len(shape) == 0:
                shape += new_char
            elif shape[-1] != new_char:
                shape += new_char
        return shape

    @staticmethod
    def check_X(X: Union[list, tuple, np.array], X_name: str):
        if (not hasattr(X, '__len__')) or (not hasattr(X, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not list-like object!'.format(X_name))
        if isinstance(X, np.ndarray):
            if len(X.shape) != 1:
                raise ValueError('`{0}` is wrong, because it is not 1-D list!'.format(X_name))
        n = len(X)
        for idx in range(n):
            if (not hasattr(X[idx], '__len__')) or (not hasattr(X[idx], '__getitem__')) or \
                    (not hasattr(X[idx], 'strip')) or (not hasattr(X[idx], 'split')):
                raise ValueError('Item {0} of `{1}` is wrong, because it is not string-like object!'.format(
                    idx, X_name))

    @staticmethod
    def check_Xy(X: Union[list, tuple, np.array], X_name: str, y: Union[list, tuple, np.array], y_name: str) -> tuple:
        BERT_NER.check_X(X, X_name)
        if (not hasattr(y, '__len__')) or (not hasattr(y, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not a list-like object!'.format(y_name))
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise ValueError('`{0}` is wrong, because it is not 1-D list!'.format(y_name))
        n = len(y)
        if n != len(X):
            raise ValueError('Length of `{0}` does not correspond to length of `{1}`! {2} != {3}'.format(
                X_name, y_name, len(X), len(y)))
        classes_list = set()
        for idx in range(n):
            if (not hasattr(y[idx], '__len__')) or (not hasattr(y[idx], 'items')) or (not hasattr(y[idx], 'keys')) or \
                    (not hasattr(y[idx], 'values')):
                raise ValueError('Item {0} of `{1}` is wrong, because it is not a dictionary-like object!'.format(
                    idx, y_name))
            for ne_type in sorted(list(y[idx].keys())):
                if (not hasattr(ne_type, '__len__')) or (not hasattr(ne_type, '__getitem__')) or \
                        (not hasattr(ne_type, 'strip')) or (not hasattr(ne_type, 'split')):
                    raise ValueError('Item {0} of `{1}` is wrong, because its key `{2}` is not a string-like '
                                     'object!'.format(idx, y_name, ne_type))
                if (ne_type == 'O') or (ne_type == 'o') or (ne_type == 'О') or (ne_type == 'о'):
                    raise ValueError('Item {0} of `{1}` is wrong, because its key `{2}` incorrectly specifies a named '
                                     'entity!'.format(idx, y_name, ne_type))
                if (not ne_type.isalpha()) or (not ne_type.isupper()):
                    raise ValueError('Item {0} of `{1}` is wrong, because its key `{2}` incorrectly specifies a named '
                                     'entity!'.format(idx, y_name, ne_type))
                classes_list.add(ne_type)
                if (not hasattr(y[idx][ne_type], '__len__')) or (not hasattr(y[idx][ne_type], '__getitem__')):
                    raise ValueError('Item {0} of `{1}` is wrong, because its value `{2}` is not a list-like '
                                     'object!'.format(idx, y_name, y[idx][ne_type]))
                for ne_bounds in y[idx][ne_type]:
                    if (not hasattr(ne_bounds, '__len__')) or (not hasattr(ne_bounds, '__getitem__')):
                        raise ValueError('Item {0} of `{1}` is wrong, because named entity bounds `{2}` are not '
                                         'specified as list-like object!'.format(idx, y_name, ne_bounds))
                    if len(ne_bounds) != 2:
                        raise ValueError('Item {0} of `{1}` is wrong, because named entity bounds `{2}` are not '
                                         'specified as 2-D list!'.format(idx, y_name, ne_bounds))
                    if (ne_bounds[0] < 0) or (ne_bounds[1] > len(X[idx])) or (ne_bounds[0] >= ne_bounds[1]):
                        raise ValueError('Item {0} of `{1}` is wrong, because named entity bounds `{2}` are '
                                         'incorrect!'.format(idx, y_name, ne_bounds))
        return tuple(sorted(list(classes_list)))
