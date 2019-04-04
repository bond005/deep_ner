import os
import copy
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from itertools import chain
from bert.tokenization import FullTokenizer
from typing import Dict, Union, List, Tuple
from nltk.tokenize.nist import NISTTokenizer

from .dataset_splitting import split_dataset


class BaseDataset:
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class NER_dataset(BaseDataset):
    PATH_TO_BERT = '/mnt/data/jupyter/zp_deep_ner/pretrained/rubert_cased_L-12_H-768_A-12_v1'

    def __init__(self, texts, annotations=None, max_seq_length=512, transforms=None, bert_hub_module_handle: Union[str, None]='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1', mode='train', shapes_list=None):

        self.texts = texts
        self.annotations = annotations

        self.texts_transformed = copy.deepcopy(texts)
        self.annotations_transformed = copy.deepcopy(annotations)

        self.transforms = transforms
        if annotations:
            self.classes_list_ = self.make_classes_list(annotations)
            NER_dataset.check_Xy(texts, 'X', annotations, 'y')
        self.bert_hub_module_handle = bert_hub_module_handle
        self.tokenizer_ = self.__initialize_bert_tokenizer()
        self.nltk_tokenizer_ = NISTTokenizer()
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.shapes_list_ = shapes_list

        # TODO remove it
        self.get_counter = 0
        self.recalculate_all()

        # print(len())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        if self.get_counter == 0:
            # self.recalculate_all()
            self.get_counter = len(self.texts)

        self.get_counter -= 1
        if self.mode == 'train':
            x = [self.X_tokenized[channel_idx][index] for channel_idx in range(len(self.X_tokenized))]
            y = self.y_tokenized[index]
            return x, y
        else:
            x = [self.X_tokenized[channel_idx][index] for channel_idx in range(len(self.X_tokenized))]
            return x

    def recalculate_all(self):

        self.texts_transformed = copy.deepcopy(self.texts)
        self.annotations_transformed = copy.deepcopy(self.annotations)

        if self.transforms is not None:
            for i, (text, ann) in enumerate(zip(self.texts_transformed, self.annotations_transformed)):
                self.texts_transformed[i],  self.annotations_transformed[i] = self.transforms.apply(text, ann)

        if self.mode == 'train':
            self.X_tokenized, self.y_tokenized, self.shapes_list_, self.bounds_of_tokens_for_training = self.tokenize_all(
                self.texts_transformed, self.annotations_transformed, shapes_vocabulary=self.shapes_list_)
        else:
            self.X_tokenized, _, _, self.bounds_of_tokens_for_training = self.tokenize_all(
                self.texts_transformed, shapes_vocabulary=self.shapes_list_)

    @staticmethod
    def make_classes_list(annotations):
        classes_list = list(set(chain(*annotations)))
        return classes_list

    @staticmethod
    def split_dataset(X, y, validation_fraction=0.1):
        if validation_fraction > 0.0:
            train_index, test_index = split_dataset(y, validation_fraction)
            X_train = [X[idx] for idx in train_index]
            y_train = [y[idx] for idx in train_index]
            X_val = [X[idx] for idx in test_index]
            y_val = [y[idx] for idx in test_index]
            del train_index, test_index
        else:
            X_train = X
            y_train = y
            X_val = None
            y_val = None

        return X_train, y_train, X_val, y_val

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
        NER_dataset.check_X(X, X_name)
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

    def __initialize_bert_tokenizer(self) -> FullTokenizer:
        if self.bert_hub_module_handle is not None:
            # config = tf.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
            # self.sess_ = tf.Session(config=config)
            self.sess_ = tf.Session()
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
            # TODO Add logger
            # if self.verbose:
            #     bert_ner_logger.info('The BERT tokenizer has been loaded from a local drive. '
            #                          '`do_lower_case` is {0}.'.format(do_lower_case))
        return tokenizer_

    def tokenize_all(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array]=None,
                     shapes_vocabulary: Union[tuple, None]=None) -> Tuple[List[np.ndarray], Union[np.ndarray, None],
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
        X_tokenized = [
            np.zeros((len(X), self.max_seq_length), dtype=np.int32),
            np.zeros((len(X), self.max_seq_length), dtype=np.int32),
            np.zeros((len(X), self.max_seq_length), dtype=np.int32),
            np.zeros((len(X), self.max_seq_length, 4), dtype=np.float32)
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
            for cur_word in self.nltk_tokenizer_.international_tokenize(source_text):
                found_idx_1 = source_text[start_pos:].find(cur_word)
                if found_idx_1 < 0:
                    raise ValueError('Text `{0}` cannot be tokenized!'.format(X[sample_idx]))
                cur_word_ = cur_word.lower() if self.tokenizer_.basic_tokenizer.do_lower_case else cur_word
                subwords = self.tokenizer_.tokenize(cur_word_)
                if '[UNK]' in subwords:
                    tokenized_text.append('[UNK]')
                    bounds_of_tokens_for_text.append((start_pos + found_idx_1, start_pos + found_idx_1 + len(cur_word)))
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
                            raise ValueError('Text `{0}` cannot be tokenized!'.format(X[sample_idx]))
                        tokenized_text.append(cur_subword)
                        bounds_of_tokens_for_text.append(
                            (
                                start_pos + found_idx_1 + start_pos_2 + found_idx_2,
                                start_pos + found_idx_1 + start_pos_2 + found_idx_2 + subword_len
                            )
                        )
                        start_pos_2 += (found_idx_2 + subword_len)
                start_pos += (found_idx_1 + len(cur_word))
            if len(tokenized_text) > (self.max_seq_length - 2):
                tokenized_text = tokenized_text[:(self.max_seq_length - 2)]
                bounds_of_tokens_for_text = bounds_of_tokens_for_text[:(self.max_seq_length - 2)]
            shapes_of_text = [self.get_shape_of_string(source_text[cur[0]:cur[1]]) for cur in bounds_of_tokens_for_text]

            if shapes_vocabulary is None:
                for cur_shape in shapes_of_text:
                    if cur_shape != '[UNK]':
                        shapes_dict[cur_shape] = shapes_dict.get(cur_shape, 0) + 1
            shapes.append(shapes_of_text)
            all_tokenized_texts.append(copy.copy(tokenized_text))
            bounds_of_tokens.append(copy.deepcopy(bounds_of_tokens_for_text))
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            token_IDs = self.tokenizer_.convert_tokens_to_ids(tokenized_text)
            for token_idx in range(len(token_IDs)):
                X_tokenized[0][sample_idx][token_idx] = token_IDs[token_idx]
                X_tokenized[1][sample_idx][token_idx] = 1
                X_tokenized[3][sample_idx][token_idx][self.get_subword_ID(tokenized_text[token_idx])] = 1.0
            del bounds_of_tokens_for_text, tokenized_text, token_IDs
        if y is None:
            y_tokenized = None
        else:
            y_tokenized = np.empty((len(y), self.max_seq_length), dtype=np.int32)
            for sample_idx in range(n_samples):
                source_text = X[sample_idx]
                tokenized_text = all_tokenized_texts[sample_idx]
                bounds_of_tokens_for_text = bounds_of_tokens[sample_idx]
                indices_of_named_entities, labels_IDs = self.calculate_indices_of_named_entities(
                    source_text, self.classes_list_, y[sample_idx])
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
        X_tokenized[3] = np.concatenate((X_tokenized[3], shapes_), axis=-1)
        del shapes_, shapes
        return X_tokenized, (None if y is None else np.array(y_tokenized)), shapes_vocabulary_, \
               np.array(bounds_of_tokens, dtype=np.object)

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
    def get_subword_ID(subword: str) -> int:
        if subword == '[CLS]':
            return 0
        if subword == '[SEP]':
            return 1
        if subword.startswith('##'):
            return 2
        return 3

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
                        res[token_idx + 1] = prev_label_id
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
