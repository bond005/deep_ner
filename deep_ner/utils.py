import codecs
import copy
import json
from logging import Logger
import os
from typing import Dict, Tuple, List, Union, Set
import warnings

from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np


def load_tokens_from_factrueval2016_by_paragraphs(text_file_name: str, tokens_file_name: str) -> \
        Tuple[Dict[int, Tuple[int, int, str]], str, tuple]:
    source_text = ''
    start_pos = 0
    tokens_and_their_bounds = dict()
    line_idx = 1
    bounds_of_paragraphs = []
    texts_of_paragraphs = []
    with codecs.open(text_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                texts_of_paragraphs.append(prep_line.lower())
            cur_line = fp.readline()
    paragraph_idx = 0
    paragraph_pos = 0
    with codecs.open(tokens_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(tokens_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) != 4:
                    raise ValueError(err_msg)
                try:
                    token_id = int(parts_of_line[0])
                except:
                    token_id = -1
                if token_id < 0:
                    raise ValueError(err_msg)
                try:
                    token_start = int(parts_of_line[1])
                except:
                    token_start = -1
                if token_start < len(source_text):
                    raise ValueError(err_msg)
                try:
                    token_len = int(parts_of_line[2])
                except:
                    token_len = -1
                if token_len < 0:
                    raise ValueError(err_msg)
                token_text = parts_of_line[3].strip()
                if len(token_text) != token_len:
                    raise ValueError(err_msg)
                if token_id in tokens_and_their_bounds:
                    raise ValueError(err_msg)
                while len(source_text) < token_start:
                    source_text += ' '
                source_text += token_text
                tokens_and_their_bounds[token_id] = (
                    token_start, token_start + token_len,
                    token_text
                )
                found_idx_in_paragraph = texts_of_paragraphs[paragraph_idx][paragraph_pos:].find(token_text.lower())
                if found_idx_in_paragraph < 0:
                    paragraph_idx += 1
                    paragraph_pos = 0
                    while paragraph_idx < len(texts_of_paragraphs):
                        if len(bounds_of_paragraphs) == 0:
                            bounds_of_paragraphs.append((0, start_pos))
                        else:
                            bounds_of_paragraphs.append((bounds_of_paragraphs[-1][1], start_pos))
                        found_idx_in_paragraph = texts_of_paragraphs[paragraph_idx].find(token_text.lower())
                        if found_idx_in_paragraph >= 0:
                            break
                        paragraph_idx += 1
                    if paragraph_idx >= len(texts_of_paragraphs):
                        raise ValueError(err_msg)
                else:
                    paragraph_pos += (found_idx_in_paragraph + len(token_text))
                start_pos = len(source_text)
            cur_line = fp.readline()
            line_idx += 1
    if len(texts_of_paragraphs) > 0:
        if len(bounds_of_paragraphs) > 0:
            bounds_of_paragraphs.append((bounds_of_paragraphs[-1][1], start_pos))
        else:
            bounds_of_paragraphs.append((0, start_pos))
    bounds_of_paragraphs_after_strip = []
    for cur_bounds in bounds_of_paragraphs:
        if cur_bounds[0] < cur_bounds[1]:
            source_paragraph_text = source_text[cur_bounds[0]:cur_bounds[1]]
            paragraph_text_after_strip = source_paragraph_text.strip()
            found_idx = source_paragraph_text.find(paragraph_text_after_strip)
            if found_idx > 0:
                paragraph_start = cur_bounds[0] + found_idx
            else:
                paragraph_start = cur_bounds[0]
            paragraph_end = paragraph_start + len(paragraph_text_after_strip)
            bounds_of_paragraphs_after_strip.append((paragraph_start, paragraph_end))
        else:
            bounds_of_paragraphs_after_strip.append(cur_bounds)
    return tokens_and_their_bounds, source_text, tuple(bounds_of_paragraphs_after_strip)


def load_tokens_from_factrueval2016_by_sentences(tokens_file_name: str) -> \
        Tuple[Dict[int, Tuple[int, int, str]], str, tuple]:
    source_text = ''
    tokens_and_their_bounds = dict()
    line_idx = 1
    bounds_of_sentences = []
    sentence_start = -1
    sentence_end = -1
    with codecs.open(tokens_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(tokens_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) != 4:
                    raise ValueError(err_msg)
                try:
                    token_id = int(parts_of_line[0])
                except:
                    token_id = -1
                if token_id < 0:
                    raise ValueError(err_msg)
                try:
                    token_start = int(parts_of_line[1])
                except:
                    token_start = -1
                if token_start < len(source_text):
                    raise ValueError(err_msg)
                try:
                    token_len = int(parts_of_line[2])
                except:
                    token_len = -1
                if token_len < 0:
                    raise ValueError(err_msg)
                token_text = parts_of_line[3].strip()
                if len(token_text) != token_len:
                    raise ValueError(err_msg)
                if token_id in tokens_and_their_bounds:
                    raise ValueError(err_msg)
                while len(source_text) < token_start:
                    source_text += ' '
                source_text += token_text
                tokens_and_their_bounds[token_id] = (
                    token_start, token_start + token_len,
                    token_text
                )
                if sentence_start < 0:
                    sentence_start = token_start
                sentence_end = token_start + token_len
            else:
                if (sentence_start >= 0) and (sentence_end >= 0):
                    bounds_of_sentences.append((sentence_start, sentence_end))
                sentence_start = -1
                sentence_end = -1
            cur_line = fp.readline()
            line_idx += 1
    if (sentence_start >= 0) and (sentence_end >= 0):
        bounds_of_sentences.append((sentence_start, sentence_end))
    return tokens_and_their_bounds, source_text, tuple(bounds_of_sentences)


def load_spans_from_factrueval2016(spans_file_name: str,
                                   tokens_dict: Dict[int, Tuple[int, int, str]]) -> Dict[int, List[int]]:
    spans = dict()
    line_idx = 1
    with codecs.open(spans_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(spans_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) < 9:
                    raise ValueError(err_msg)
                try:
                    span_id = int(parts_of_line[0])
                except:
                    span_id = -1
                if span_id < 0:
                    raise ValueError(err_msg)
                if span_id not in spans:
                    try:
                        found_idx = parts_of_line.index('#')
                    except:
                        found_idx = -1
                    if found_idx < 0:
                        raise ValueError(err_msg)
                    if (len(parts_of_line) - 1 - found_idx) < 2:
                        raise ValueError(err_msg)
                    if (len(parts_of_line) - 1 - found_idx) % 2 != 0:
                        raise ValueError(err_msg)
                    n = (len(parts_of_line) - 1 - found_idx) // 2
                    token_IDs = []
                    try:
                        for idx in range(found_idx + 1, found_idx + n + 1):
                            new_token_ID = int(parts_of_line[idx])
                            if new_token_ID in token_IDs:
                                token_IDs = []
                                break
                            if new_token_ID not in tokens_dict:
                                token_IDs = []
                                break
                            token_IDs.append(new_token_ID)
                            if token_IDs[-1] < 0:
                                token_IDs = []
                                break
                    except:
                        token_IDs = []
                    if len(token_IDs) == 0:
                        raise ValueError(err_msg)
                    spans[span_id] = token_IDs
                    del token_IDs
            cur_line = fp.readline()
            line_idx += 1
    return spans


def load_objects_from_factrueval2016(objects_file_name: str,
                                     spans_dict: Dict[int, List[int]]) -> Dict[int, Tuple[str, List[int]]]:
    objects = dict()
    line_idx = 1
    with codecs.open(objects_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(objects_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) < 5:
                    raise ValueError(err_msg)
                try:
                    object_id = int(parts_of_line[0])
                    if object_id in objects:
                        object_id = -1
                except:
                    object_id = -1
                if object_id < 0:
                    raise ValueError(err_msg)
                ne_type = parts_of_line[1].upper()
                if ne_type in {'PERSON', 'LOCATION', 'ORG', 'LOCORG'}:
                    if ne_type == 'LOCORG':
                        ne_type = 'LOCATION'
                    try:
                        found_idx = parts_of_line.index('#')
                    except:
                        found_idx = -1
                    if found_idx < 3:
                        raise ValueError(err_msg)
                    span_IDs = []
                    try:
                        for idx in range(2, found_idx):
                            new_span_ID = int(parts_of_line[idx])
                            if new_span_ID < 0:
                                span_IDs = []
                                break
                            if new_span_ID not in spans_dict:
                                span_IDs = []
                                break
                            if new_span_ID in span_IDs:
                                span_IDs = []
                                break
                            span_IDs.append(new_span_ID)
                    except:
                        span_IDs = []
                    if len(span_IDs) == 0:
                        raise ValueError(err_msg)
                    objects[object_id] = (ne_type, span_IDs)
                    del span_IDs
            cur_line = fp.readline()
            line_idx += 1
    return objects


def check_factrueval_tokenization(src_dir_name: str, split_by_paragraphs: bool):
    factrueval_files = dict()
    for cur_file_name in os.listdir(src_dir_name):
        if cur_file_name.endswith('.objects'):
            base_name = cur_file_name[:-len('.objects')]
        elif cur_file_name.endswith('.spans'):
            base_name = cur_file_name[:-len('.spans')]
        elif cur_file_name.endswith('.tokens'):
            base_name = cur_file_name[:-len('.tokens')]
        else:
            base_name = None
        if base_name is not None:
            if base_name in factrueval_files:
                assert cur_file_name not in factrueval_files[base_name]
                factrueval_files[base_name].append(cur_file_name)
            else:
                factrueval_files[base_name] = [cur_file_name]
    for base_name in factrueval_files:
        if len(factrueval_files[base_name]) != 3:
            raise ValueError('Files list for `{0}` is wrong!'.format(base_name))
        text_file_name = os.path.join(src_dir_name, base_name + '.txt')
        if not os.path.isfile(text_file_name):
            raise ValueError('File `{0}` does not exist!'.format(text_file_name))
        factrueval_files[base_name].append(text_file_name)
        factrueval_files[base_name] = sorted(factrueval_files[base_name])
    n_good = 0
    n_total = 0
    for base_name in sorted(list(factrueval_files.keys())):
        if split_by_paragraphs:
            tokens, text, paragraphs = load_tokens_from_factrueval2016_by_paragraphs(
                os.path.join(src_dir_name, base_name + '.txt'), os.path.join(src_dir_name, base_name + '.tokens')
            )
        else:
            tokens, text, paragraphs = load_tokens_from_factrueval2016_by_sentences(
                os.path.join(src_dir_name, base_name + '.tokens')
            )
        tokens_by_tokenizer = []
        for paragraph_start, paragraph_end in paragraphs:
            tokens_by_tokenizer += word_tokenize(text[paragraph_start:paragraph_end])
        tokens_by_factrueval = []
        for token_id in sorted(list(tokens.keys())):
            tokens_by_factrueval.append(tokens[token_id][2])
        tokens_by_tokenizer = tuple(tokens_by_tokenizer)
        tokens_by_factrueval = tuple(tokens_by_factrueval)
        if tokens_by_tokenizer == tokens_by_factrueval:
            print('')
            print('{0}'.format(base_name))
            print('All right!')
            print('')
            n_good += 1
        else:
            print('')
            print('{0}'.format(base_name))
            print('')
            print('true tokens:')
            print('{0}'.format(tokens_by_factrueval))
            print('')
            print('calculated tokens:')
            print('{0}'.format(tokens_by_tokenizer))
            print('')
        n_total += 1
    print('')
    print('Total number of texts is {0}.'.format(n_total))
    print('Number of correctly tokenized texts is {0}.'.format(n_good))


def factrueval2016_to_json(src_dir_name: str, dst_json_name: str, split_by_paragraphs: bool=True):
    factrueval_files = dict()
    for cur_file_name in os.listdir(src_dir_name):
        if cur_file_name.endswith('.objects'):
            base_name = cur_file_name[:-len('.objects')]
        elif cur_file_name.endswith('.spans'):
            base_name = cur_file_name[:-len('.spans')]
        elif cur_file_name.endswith('.tokens'):
            base_name = cur_file_name[:-len('.tokens')]
        else:
            base_name = None
        if base_name is not None:
            if base_name in factrueval_files:
                assert cur_file_name not in factrueval_files[base_name]
                factrueval_files[base_name].append(cur_file_name)
            else:
                factrueval_files[base_name] = [cur_file_name]
    for base_name in factrueval_files:
        if len(factrueval_files[base_name]) != 3:
            raise ValueError('Files list for `{0}` is wrong!'.format(base_name))
        text_file_name = os.path.join(src_dir_name, base_name + '.txt')
        if not os.path.isfile(text_file_name):
            raise ValueError('File `{0}` does not exist!'.format(text_file_name))
        factrueval_files[base_name].append(text_file_name)
        factrueval_files[base_name] = sorted(factrueval_files[base_name])
    train_data = []
    for base_name in sorted(list(factrueval_files.keys())):
        if split_by_paragraphs:
            tokens, text, paragraphs = load_tokens_from_factrueval2016_by_paragraphs(
                os.path.join(src_dir_name, base_name + '.txt'), os.path.join(src_dir_name, base_name + '.tokens')
            )
        else:
            tokens, text, paragraphs = load_tokens_from_factrueval2016_by_sentences(
                os.path.join(src_dir_name, base_name + '.tokens')
            )
        spans = load_spans_from_factrueval2016(os.path.join(src_dir_name, base_name + '.spans'), tokens)
        objects = load_objects_from_factrueval2016(os.path.join(src_dir_name, base_name + '.objects'), spans)
        named_entities = dict()
        if len(objects) > 0:
            for object_ID in objects:
                ne_type = objects[object_ID][0]
                tokens_of_ne = set()
                spans_of_ne = objects[object_ID][1]
                for span_ID in spans_of_ne:
                    tokens_of_ne |= set(spans[span_ID])
                tokens_of_ne = sorted(list(tokens_of_ne))
                if len(tokens_of_ne) > 0:
                    token_ID = tokens_of_ne[0]
                    ne_start = tokens[token_ID][0]
                    ne_end = tokens[token_ID][1]
                    for token_ID in tokens_of_ne[1:]:
                        if tokens[token_ID][0] < ne_start:
                            ne_start = tokens[token_ID][0]
                        if tokens[token_ID][1] > ne_end:
                            ne_end = tokens[token_ID][1]
                    if ne_type in named_entities:
                        named_entities[ne_type].append((ne_start, ne_end))
                    else:
                        named_entities[ne_type] = [(ne_start, ne_end)]
        train_data.append({'text': text, 'named_entities': named_entities, 'paragraph_bounds': paragraphs,
                           'base_name': base_name})
    with codecs.open(dst_json_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(train_data, fp, indent=4, ensure_ascii=False)


def recognized_factrueval2016_to_json(gold_dir_name: str, recognized_dir_name: str, dst_json_name: str):
    factrueval_files = dict()
    for cur_file_name in os.listdir(gold_dir_name):
        if cur_file_name.endswith('.objects'):
            base_name = cur_file_name[:-len('.objects')]
        elif cur_file_name.endswith('.spans'):
            base_name = cur_file_name[:-len('.spans')]
        elif cur_file_name.endswith('.tokens'):
            base_name = cur_file_name[:-len('.tokens')]
        elif cur_file_name.endswith('.txt'):
            base_name = cur_file_name[:-len('.txt')]
        else:
            base_name = None
        if base_name is not None:
            if base_name in factrueval_files:
                assert cur_file_name not in factrueval_files[base_name]
                factrueval_files[base_name].append(cur_file_name)
            else:
                factrueval_files[base_name] = [cur_file_name]
    for base_name in factrueval_files:
        factrueval_files[base_name] = sorted(factrueval_files[base_name])
        if len(factrueval_files[base_name]) != 4:
            raise ValueError('Files list for `{0}` is wrong!'.format(base_name))
    for base_name in factrueval_files:
        if not os.path.isfile(os.path.join(recognized_dir_name, base_name + '.task1')):
            raise ValueError('File `{0}` is not found!'.format(os.path.join(recognized_dir_name, base_name + '.task1')))
    train_data = []
    for base_name in sorted(list(factrueval_files.keys())):
        tokens, text, paragraphs = load_tokens_from_factrueval2016_by_paragraphs(
            os.path.join(gold_dir_name, base_name + '.txt'), os.path.join(gold_dir_name, base_name + '.tokens')
        )
        named_entities = dict()
        with codecs.open(os.path.join(recognized_dir_name, base_name + '.task1'), mode='r', encoding='utf-8',
                         errors='ignore') as fp:
            line_idx = 1
            cur_line = fp.readline()
            while len(cur_line) > 0:
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    err_msg = 'File `{0}`: line {1} is wrong!'.format(
                        os.path.join(recognized_dir_name, base_name + '.task1'), line_idx)
                    parts_of_line = prep_line.split()
                    if len(parts_of_line) != 3:
                        raise ValueError(err_msg)
                    ne_type = parts_of_line[0].upper()
                    if ne_type not in {'PER', 'LOC', 'ORG'}:
                        raise ValueError(err_msg)
                    if ne_type == 'PER':
                        ne_type = 'PERSON'
                    elif ne_type == 'LOC':
                        ne_type = 'LOCATION'
                    try:
                        ne_start = int(parts_of_line[1])
                    except:
                        ne_start = -1
                    if ne_start < 0:
                        raise ValueError(err_msg)
                    try:
                        ne_length = int(parts_of_line[2])
                    except:
                        ne_length = -1
                    if ne_length < 0:
                        raise ValueError(err_msg)
                    if ne_type in named_entities:
                        named_entities[ne_type].append((ne_start, ne_start + ne_length))
                    else:
                        named_entities[ne_type] = [(ne_start, ne_start + ne_length)]
                cur_line = fp.readline()
                line_idx += 1
        train_data.append({'text': text, 'named_entities': named_entities, 'paragraph_bounds': paragraphs})
    with codecs.open(dst_json_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(train_data, fp, indent=4, ensure_ascii=False)


def find_paragraph(bounds_of_paragraphs: List[Tuple[int, int]], entity_start_idx: int, entity_end_idx: int) -> int:
    paragraph_idx = -1
    for idx, bounds in enumerate(bounds_of_paragraphs):
        if (entity_start_idx >= bounds[0]) and (entity_start_idx < bounds[1]):
            if (entity_end_idx > entity_start_idx) and (entity_end_idx <= bounds[1]):
                paragraph_idx = idx
                break
    return paragraph_idx


def load_dataset_from_json(file_name: str) -> Tuple[List[str], List[Dict[str, List[Tuple[int, int]]]]]:

    def prepare_bounds(source_named_entities: Dict[str, List[List[int]]]) -> Dict[str, List[Tuple[int, int]]]:
        prepared_named_entities = dict()
        for cur_ne in source_named_entities:
            new_list = []
            prev_idx_ = -1
            for old_bounds in sorted(source_named_entities[cur_ne]):
                if prev_idx_ < 0:
                    new_list.append((old_bounds[0], old_bounds[1]))
                else:
                    if prev_idx_ >= old_bounds[0]:
                        new_list[-1] = (new_list[-1][0], old_bounds[1])
                    else:
                        new_list.append((old_bounds[0], old_bounds[1]))
                prev_idx_ = old_bounds[1]
            prepared_named_entities[cur_ne] = new_list
            del new_list
        return prepared_named_entities

    if not os.path.isfile(file_name):
        raise ValueError('The file `{0}` does not exist!'.format(file_name))
    X = []
    y = []
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError('The file `{0}` contains incorrect data! Expected a `{1}`, but got a `{2}`.'.format(
            file_name, type([1, 2]), type(data)))
    for sample_idx, sample_value in enumerate(data):
        if not isinstance(sample_value, dict):
            raise ValueError('{0} sample in the file `{1}` contains incorrect data! Expected a `{2}`, but got a '
                             '`{3}`.'.format(sample_idx, file_name, type({'a': 1, 'b': 2}), type(sample_value)))
        if 'text' not in sample_value:
            raise ValueError('{0} sample in the file `{1}` contains incorrect data! '
                             'The key `text` is not found!'.format(sample_idx, file_name))
        if 'named_entities' not in sample_value:
            raise ValueError('{0} sample in the file `{1}` contains incorrect data! '
                             'The key `named_entities` is not found!'.format(sample_idx, file_name))
        if 'paragraph_bounds' in sample_value:
            bounds_of_paragraphs = sample_value['paragraph_bounds']
            if len(sample_value) > 3:
                excess_keys = sorted(list(set(sample_value.keys()) - {'text', 'named_entities', 'paragraph_bounds',
                                                                      'base_name'}))
                if len(excess_keys) > 0:
                    raise ValueError('{0} sample in the file `{1}` contains incorrect data! Keys {2} are '
                                     'excess!'.format(sample_idx, file_name, excess_keys))
            if not isinstance(sample_value['paragraph_bounds'], list):
                raise ValueError(
                    '{0} sample in the file `{1}` contains incorrect data! Value of `paragraph_bounds` must be '
                    'a `{2}`, but it is a `{3}`.'.format(sample_idx, file_name, type([1, 2, 3]),
                                                         type(sample_value['text'])))
        else:
            bounds_of_paragraphs = None
            if len(sample_value) > 2:
                excess_keys = sorted(list(set(sample_value.keys()) - {'text', 'named_entities', 'base_name'}))
                if len(excess_keys) > 0:
                    raise ValueError('{0} sample in the file `{1}` contains incorrect data! Keys {2} are '
                                     'excess!'.format(sample_idx, file_name, excess_keys))
        if not isinstance(sample_value['text'], str):
            raise ValueError('{0} sample in the file `{1}` contains incorrect data! Value of `text` must be a `{2}`, '
                             'but it is a `{3}`.'.format(sample_idx, file_name, type('123'),
                                                         type(sample_value['text'])))
        if not isinstance(sample_value['named_entities'], dict):
            raise ValueError('{0} sample in the file `{1}` contains incorrect data! Value of `named_entities` must be '
                             'a `{2}`, but it is a `{3}`.'.format(sample_idx, file_name, type({'a': 1, 'b': 2}),
                                                                  type(sample_value['text'])))
        for entity_type in sample_value['named_entities']:
            if not isinstance(entity_type, str):
                raise ValueError(
                    '{0} sample in the file `{1}` contains incorrect data! Entity type `{2}` is wrong! Expected a '
                    '`{3}`, but got a `{4}`.'.format(sample_idx, file_name, entity_type, type('123'),
                                                     type(entity_type)))
            for entity_bounds in sample_value['named_entities'][entity_type]:
                if not isinstance(entity_bounds, list):
                    raise ValueError('{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for '
                                     'entity bounds.'.format(sample_idx, file_name, entity_bounds))
                if len(entity_bounds) != 2:
                    raise ValueError('{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for '
                                     'entity bounds.'.format(sample_idx, file_name, entity_bounds))
                if (entity_bounds[0] < 0) or (entity_bounds[0] >= len(sample_value['text'])):
                    raise ValueError('{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for '
                                     'entity bounds.'.format(sample_idx, file_name, entity_bounds))
                if (entity_bounds[1] <= entity_bounds[0]) or (entity_bounds[1] > len(sample_value['text'])):
                    raise ValueError('{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for '
                                     'entity bounds.'.format(sample_idx, file_name, entity_bounds))
        if bounds_of_paragraphs is not None:
            for paragraph_bounds in bounds_of_paragraphs:
                if not isinstance(paragraph_bounds, list):
                    raise ValueError(
                        '{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for paragraph '
                        'bounds.'.format(sample_idx, file_name, paragraph_bounds))
                if len(paragraph_bounds) != 2:
                    raise ValueError(
                        '{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for paragraph '
                        'bounds.'.format(sample_idx, file_name, paragraph_bounds))
                if (paragraph_bounds[0] < 0) or (paragraph_bounds[0] >= len(sample_value['text'])):
                    raise ValueError(
                        '{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for paragraph '
                        'bounds.'.format(sample_idx, file_name, paragraph_bounds))
                if (paragraph_bounds[1] <= paragraph_bounds[0]) or (paragraph_bounds[1] > len(sample_value['text'])):
                    raise ValueError(
                        '{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for paragraph '
                        'bounds.'.format(sample_idx, file_name, paragraph_bounds))
            bounds_of_paragraphs = [tuple(cur) for cur in bounds_of_paragraphs]
            text_by_paragraphs = list()
            entities_by_paragraphs = list()
            for paragraph_start, paragraph_end in bounds_of_paragraphs:
                text_by_paragraphs.append(sample_value['text'][paragraph_start:paragraph_end])
                entities_by_paragraphs.append(dict())
            for entity_type in sample_value['named_entities']:
                for entity_bounds in sorted(sample_value['named_entities'][entity_type]):
                    paragraph_idx = find_paragraph(bounds_of_paragraphs, entity_bounds[0], entity_bounds[1])
                    if paragraph_idx < 0:
                        raise ValueError('{0} sample in the file `{1}` contains incorrect data! {2} is wrong value for '
                                         'entity bounds.'.format(sample_idx, file_name, entity_bounds))
                    paragraph_start = bounds_of_paragraphs[paragraph_idx][0]
                    if entity_type in entities_by_paragraphs[paragraph_idx]:
                        entities_by_paragraphs[paragraph_idx][entity_type].append(
                            (entity_bounds[0] - paragraph_start, entity_bounds[1] - paragraph_start)
                        )
                    else:
                        entities_by_paragraphs[paragraph_idx][entity_type] = [
                            (entity_bounds[0] - paragraph_start, entity_bounds[1] - paragraph_start)
                        ]
            for paragraph_idx in range(len(bounds_of_paragraphs)):
                X.append(text_by_paragraphs[paragraph_idx])
                entities_by_paragraphs_ = dict()
                for entity_type in entities_by_paragraphs[paragraph_idx]:
                    entities_list = []
                    prev_idx = -1
                    for entity_bounds in sorted(entities_by_paragraphs[paragraph_idx][entity_type]):
                        if prev_idx < 0:
                            entities_list.append(entity_bounds)
                        else:
                            if prev_idx >= entity_bounds[0]:
                                entities_list[-1] = (entities_list[-1][0], entity_bounds[1])
                            else:
                                entities_list.append(entity_bounds)
                        prev_idx = entity_bounds[1]
                    entities_by_paragraphs_[entity_type] = entities_list
                    del entities_list
                y.append(entities_by_paragraphs_)
                del entities_by_paragraphs_
        else:
            X.append(sample_value['text'])
            y.append(prepare_bounds(sample_value['named_entities']))
    return X, y


def load_dataset_from_brat(brat_datadir_name: str, split_by_paragraphs: bool=True,
                           log: Union[Logger, None]=None) -> Tuple[List[str], List[Dict[str, List[Tuple[int, int]]]]]:
    all_annotation_files = sorted(list(filter(lambda it: it.endswith('.ann'), os.listdir(brat_datadir_name))))
    if len(all_annotation_files) == 0:
        raise ValueError('There are no annotation files into the directory `{0}`!'.format(brat_datadir_name))
    all_file_pairs = list()
    for annotation_file in all_annotation_files:
        text_file = annotation_file[:-3] + 'txt'
        if not os.path.isfile(os.path.join(brat_datadir_name, text_file)):
            raise ValueError('The annotation file `{0}` has not a corresponding text file!'.format(annotation_file))
        all_file_pairs.append(
            (
                os.path.join(brat_datadir_name, text_file),
                os.path.join(brat_datadir_name, annotation_file)
            )
        )
    del all_annotation_files
    texts = []
    entities = []
    for text_file, annotation_file in all_file_pairs:
        with codecs.open(text_file, mode='r', encoding='utf-8', errors='ignore') as fp:
            full_text = fp.read()
        entities_in_text = dict()
        with codecs.open(annotation_file, mode='r', encoding='utf-8', errors='ignore') as fp:
            cur_line = fp.readline()
            line_idx = 1
            while len(cur_line) > 0:
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    err_msg = 'File `{0}`: line {1} is wrong!'.format(annotation_file, line_idx)
                    parts_of_line = prep_line.split('\t')
                    if len(parts_of_line) != 3:
                        raise ValueError(err_msg)
                    entity_text = parts_of_line[2]
                    entity_description = parts_of_line[1]
                    parts_of_description = entity_description.split()
                    if len(parts_of_description) != 3:
                        raise ValueError(err_msg)
                    entity_type = parts_of_description[0]
                    try:
                        entity_start = int(parts_of_description[1])
                        entity_end = int(parts_of_description[2])
                    except:
                        entity_start = -1
                        entity_end = -1
                    if (entity_start < 0) or (entity_end <= entity_start):
                        raise ValueError(err_msg)
                    if entity_end > len(full_text):
                        raise ValueError(err_msg)
                    if full_text[entity_start:entity_end].strip() != entity_text.strip():
                        raise ValueError(err_msg)
                    if entity_type in entities_in_text:
                        entities_in_text[entity_type].append((entity_start, entity_end))
                    else:
                        entities_in_text[entity_type] = [(entity_start, entity_end)]
                cur_line = fp.readline()
                line_idx += 1
        for entity_type in sorted(list(entities_in_text.keys())):
            bounds_of_entities = sorted(entities_in_text[entity_type], key=lambda it: (it[0], it[1]))
            if len(bounds_of_entities) > 1:
                entity_idx = 1
                while entity_idx < len(bounds_of_entities):
                    if bounds_of_entities[entity_idx - 1][1] > bounds_of_entities[entity_idx][0]:
                        if log is None:
                            warnings.warn('File `{0}`, entity type `{1}`: bounds of entities are overlapped!'.format(
                                annotation_file, entity_type))
                        else:
                            log.warning('File `{0}`, entity type `{1}`: bounds of entities are overlapped!'.format(
                                annotation_file, entity_type))
                        if bounds_of_entities[entity_idx - 1][1] >= bounds_of_entities[entity_idx][1]:
                            del bounds_of_entities[entity_idx]
                        else:
                            bounds_of_entities[entity_idx] = (bounds_of_entities[entity_idx - 1][1],
                                                              bounds_of_entities[entity_idx][1])
                            entity_idx += 1
                    else:
                        entity_idx += 1
            entities_in_text[entity_type] = copy.deepcopy(bounds_of_entities)
            del bounds_of_entities
        soft_hyphen_idx = full_text.find('\xad')
        while soft_hyphen_idx >= 0:
            full_text = full_text[:soft_hyphen_idx] + full_text[(soft_hyphen_idx + 1):]
            for entity_type in sorted(list(entities_in_text.keys())):
                for entity_idx in range(len(entities_in_text[entity_type])):
                    entity_bounds = entities_in_text[entity_type][entity_idx]
                    if entity_bounds[0] > soft_hyphen_idx:
                        entity_bounds = (entity_bounds[0] - 1, entity_bounds[1] - 1)
                    elif entity_bounds[1] > soft_hyphen_idx:
                        entity_bounds = (entity_bounds[0], entity_bounds[1] - 1)
                    entities_in_text[entity_type][entity_idx] = entity_bounds
            start_pos = soft_hyphen_idx
            soft_hyphen_idx = full_text[start_pos:].find('\xad')
            if soft_hyphen_idx >= 0:
                soft_hyphen_idx += start_pos
        if split_by_paragraphs:
            paragraph_start = 0
            found_idx_1 = full_text.find('\n')
            found_idx_2 = full_text.find('\r')
            if (found_idx_1 >= 0) and (found_idx_2 >= 0):
                paragraph_end = min(found_idx_1, found_idx_2)
            elif found_idx_1 >= 0:
                paragraph_end = found_idx_1
            elif found_idx_2 >= 0:
                paragraph_end = found_idx_2
            else:
                paragraph_end = -1
            if paragraph_end < 0:
                full_text = full_text.replace('\n', ' ').replace('\r', ' ')
                texts.append(full_text)
                entities.append(entities_in_text)
            else:
                while paragraph_end >= 0:
                    entities_in_paragraph = dict()
                    for entity_type in entities_in_text:
                        for entity_bounds in entities_in_text[entity_type]:
                            if entity_bounds[0] >= paragraph_end:
                                break
                            if entity_bounds[1] <= paragraph_start:
                                continue
                            if (entity_bounds[0] >= paragraph_start) and (entity_bounds[1] <= paragraph_end):
                                entities_in_paragraph[entity_type] = entities_in_paragraph.get(entity_type, []) + \
                                                                     [
                                                                         (
                                                                             entity_bounds[0] - paragraph_start,
                                                                             entity_bounds[1] - paragraph_start
                                                                         )
                                                                     ]
                            else:
                                if entity_bounds[0] < paragraph_start:
                                    entity_start = paragraph_start
                                else:
                                    entity_start = entity_bounds[0]
                                if entity_bounds[1] > paragraph_end:
                                    entity_end = paragraph_end
                                else:
                                    entity_end = entity_bounds[1]
                                if full_text[entity_start:entity_end].strip() == \
                                        full_text[entity_bounds[0]:entity_bounds[1]].strip():
                                    entities_in_paragraph[entity_type] = entities_in_paragraph.get(entity_type, []) + \
                                                                         [
                                                                             (
                                                                                 entity_start - paragraph_start,
                                                                                 entity_end - paragraph_start
                                                                             )
                                                                         ]
                                else:
                                    raise ValueError('File `{0}`, entity type `{1}`: bounds of entities {2} are between'
                                                     ' paragraphs!'.format(annotation_file, entity_type, entity_bounds))
                    texts.append(full_text[paragraph_start:paragraph_end])
                    entities.append(entities_in_paragraph)
                    paragraph_start = -1
                    idx = paragraph_end + 1
                    while idx < len(full_text):
                        if full_text[idx] not in {'\r', '\n'}:
                            break
                        idx += 1
                    if idx < len(full_text):
                        paragraph_start = idx
                        found_idx_1 = full_text[paragraph_start:].find('\n')
                        found_idx_2 = full_text[paragraph_start:].find('\r')
                        if (found_idx_1 >= 0) and (found_idx_2 >= 0):
                            paragraph_end = min(found_idx_1, found_idx_2) + paragraph_start
                        elif found_idx_1 >= 0:
                            paragraph_end = found_idx_1 + paragraph_start
                        elif found_idx_2 >= 0:
                            paragraph_end = found_idx_2 + paragraph_start
                        else:
                            paragraph_end = -1
                    else:
                        paragraph_end = -1
                if paragraph_start >= 0:
                    paragraph_end = len(full_text)
                    entities_in_paragraph = dict()
                    for entity_type in entities_in_text:
                        for entity_bounds in entities_in_text[entity_type]:
                            if entity_bounds[0] >= paragraph_end:
                                break
                            if entity_bounds[1] <= paragraph_start:
                                continue
                            if (entity_bounds[0] >= paragraph_start) and (entity_bounds[1] <= paragraph_end):
                                entities_in_paragraph[entity_type] = entities_in_paragraph.get(entity_type, []) + \
                                                                     [
                                                                         (
                                                                             entity_bounds[0] - paragraph_start,
                                                                             entity_bounds[1] - paragraph_start
                                                                         )
                                                                     ]
                            else:
                                if entity_bounds[0] < paragraph_start:
                                    entity_start = paragraph_start
                                else:
                                    entity_start = entity_bounds[0]
                                if entity_bounds[1] > paragraph_end:
                                    entity_end = paragraph_end
                                else:
                                    entity_end = entity_bounds[1]
                                if full_text[entity_start:entity_end].strip() == \
                                        full_text[entity_bounds[0]:entity_bounds[1]].strip():
                                    entities_in_paragraph[entity_type] = entities_in_paragraph.get(entity_type, []) + \
                                                                         [
                                                                             (
                                                                                 entity_start - paragraph_start,
                                                                                 entity_end - paragraph_start
                                                                             )
                                                                         ]
                                else:
                                    raise ValueError('File `{0}`, entity type `{1}`: bounds of entities {2} are between'
                                                     ' paragraphs!'.format(annotation_file, entity_type, entity_bounds))
                    texts.append(full_text[paragraph_start:paragraph_end])
                    entities.append(entities_in_paragraph)
        else:
            full_text = full_text.replace('\n', ' ').replace('\r', ' ')
            texts.append(full_text)
            entities.append(entities_in_text)
    return texts, entities


def load_dataset_from_bio(file_name: str, paragraph_separators: Set[str]=None,
                          stopwords: Set[str]=None) -> Tuple[List[str], List[Dict[str, List[Tuple[int, int]]]]]:
    texts = []
    named_entities = []
    new_text = ''
    named_entities_for_new_text = dict()
    entity_start = -1
    entity_type = ''
    line_idx = 1
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) < 2:
                    raise ValueError(err_msg)
                token_text = parts_of_line[0]
                token_label = parts_of_line[-1]
                if not token_label.isupper():
                    raise ValueError(err_msg)
                if (token_label != 'O') and (not token_label.startswith('B-')) and (not token_label.startswith('I-')):
                    raise ValueError(err_msg)
                if (token_label != 'O') and (len(token_label) < 3):
                    raise ValueError(err_msg)
                if (paragraph_separators is not None) and (token_text in paragraph_separators):
                    if len(new_text) > 0:
                        if entity_start >= 0:
                            named_entities_for_new_text[entity_type] = named_entities_for_new_text.get(
                                entity_type, []
                            ) + [(entity_start, len(new_text))]
                        texts.append(new_text)
                        named_entities.append(named_entities_for_new_text)
                    new_text = ''
                    named_entities_for_new_text = dict()
                    entity_start = -1
                    entity_type = ''
                elif (stopwords is None) or (token_text not in stopwords):
                    if token_label == 'O':
                        if entity_start >= 0:
                            named_entities_for_new_text[entity_type] = named_entities_for_new_text.get(
                                entity_type, []
                            ) + [(entity_start, len(new_text))]
                            entity_start = -1
                            entity_type = ''
                    else:
                        if token_label.startswith('B-') and (entity_start >= 0):
                            named_entities_for_new_text[entity_type] = named_entities_for_new_text.get(
                                entity_type, []
                            ) + [(entity_start, len(new_text))]
                            entity_start = -1
                            entity_type = ''
                    if token_text.isalnum():
                        if len(new_text) == 0:
                            new_text = token_text
                        else:
                            new_text += (' ' + token_text)
                    else:
                        if token_text in {')', '}', ']', '>', '.', ',', '?', ':', ';'}:
                            new_text += token_text
                        else:
                            if (token_text == '%') and (len(new_text) > 0) and (new_text[-1].isdigit()):
                                new_text += token_text
                            elif (token_text == '\'') and (new_text.endswith('\'')):
                                new_text += token_text
                            else:
                                if len(new_text) == 0:
                                    new_text = token_text
                                else:
                                    new_text += (' ' + token_text)
                    if token_label.startswith('B-'):
                        entity_start = new_text.rfind(token_text)
                        entity_type = token_label[2:]
            else:
                if paragraph_separators is None:
                    if len(new_text) > 0:
                        if entity_start >= 0:
                            named_entities_for_new_text[entity_type] = named_entities_for_new_text.get(
                                entity_type, []
                            ) + [(entity_start, len(new_text))]
                        texts.append(new_text)
                        named_entities.append(named_entities_for_new_text)
                    new_text = ''
                    named_entities_for_new_text = dict()
                    entity_start = -1
                    entity_type = ''
            cur_line = fp.readline()
            line_idx += 1
    if len(new_text) > 0:
        if entity_start >= 0:
            named_entities_for_new_text[entity_type] = named_entities_for_new_text.get(entity_type, []) + \
                                                       [(entity_start, len(new_text))]
        texts.append(new_text)
        named_entities.append(named_entities_for_new_text)
    return texts, named_entities


def get_bio_label_of_token(source_text: str, token_bounds: Tuple[int, int],
                           named_entities: Dict[str, List[Tuple[int, int]]]) -> str:
    best_ne_type = ''
    best_ne_idx = None
    best_similarity = 0
    for ne_type in named_entities:
        for ne_idx, ne_bounds in enumerate(named_entities[ne_type]):
            if ne_bounds[1] <= token_bounds[0]:
                continue
            if ne_bounds[0] >= token_bounds[1]:
                continue
            if (ne_bounds[0] <= token_bounds[0]) and (ne_bounds[1] >= token_bounds[1]):
                new_similarity = token_bounds[1] - token_bounds[0]
            elif (ne_bounds[0] >= token_bounds[0]) and (ne_bounds[1] <= token_bounds[1]):
                new_similarity = ne_bounds[1] - ne_bounds[0]
            elif ne_bounds[0] <= token_bounds[0]:
                new_similarity = ne_bounds[1] - token_bounds[0]
            else:
                new_similarity = token_bounds[1] - ne_bounds[0]
            if new_similarity > best_similarity:
                best_similarity = new_similarity
                best_ne_type = ne_type
                best_ne_idx = ne_idx
    if (len(best_ne_type) == 0) or (best_ne_idx is None):
        return 'O'
    if best_similarity < ((token_bounds[1] - token_bounds[0]) // 2):
        return 'O'
    if token_bounds[0] <= named_entities[best_ne_type][best_ne_idx][0]:
        return 'B-' + best_ne_type
    if len(source_text[named_entities[best_ne_type][best_ne_idx][0]:token_bounds[0]].strip()) == 0:
        return 'B-' + best_ne_type
    return 'I-' + best_ne_type


def save_dataset_as_bio(source_file_name: str, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array],
                        result_file_name: str, stopwords: Set[str]=None):
    sample_idx = 0
    char_idx = 0
    line_idx = 1
    src_fp = None
    dst_fp = None
    is_new_line = True
    text_is_ended = False
    try:
        src_fp = codecs.open(source_file_name, mode='r', encoding='utf-8', errors='ignore')
        dst_fp = codecs.open(result_file_name, mode='w', encoding='utf-8', errors='ignore')
        cur_line = src_fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                is_new_line = False
                err_msg = 'File `{0}`: line {1} is wrong!'.format(source_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) < 2:
                    raise ValueError(err_msg)
                token_text = parts_of_line[0]
                if ((stopwords is None) or (token_text not in stopwords)) and (not text_is_ended):
                    found_idx = X[sample_idx][char_idx:].find(token_text)
                    if found_idx < 0:
                        if sample_idx < (len(X) - 1):
                            found_idx = X[sample_idx + 1].find(token_text)
                            if found_idx < 0:
                                text_is_ended = True
                            else:
                                sample_idx += 1
                                char_idx = found_idx
                        else:
                            raise ValueError(err_msg + ' Token `{0}` cannot be found in the text `{1}`!'.format(
                                token_text, X[sample_idx]))
                    else:
                        char_idx += found_idx
                    ne_label = get_bio_label_of_token(X[sample_idx], (char_idx, char_idx + len(token_text)),
                                                      y[sample_idx])
                    dst_fp.write('{0}\n'.format(' '.join(parts_of_line + [ne_label])))
                    char_idx += len(token_text)
                else:
                    dst_fp.write('{0}\n'.format(' '.join(parts_of_line + ['O'])))
            else:
                if not is_new_line:
                    dst_fp.write('\n')
                is_new_line = True
                text_is_ended = False
            cur_line = src_fp.readline()
            line_idx += 1
    finally:
        if src_fp is not None:
            src_fp.close()
        if dst_fp is not None:
            dst_fp.close()


def divide_dataset_by_sentences(X: Union[list, tuple, np.array], y: Union[list, tuple, np.array]) -> \
        Tuple[Union[list, tuple, np.array], Union[list, tuple, np.array]]:
    X_new = []
    y_new = []
    n_samples = len(X)
    for sample_idx in range(n_samples):
        sentences = sent_tokenize(X[sample_idx])
        start_idx = 0
        bounds_of_sentences = []
        for cur_sent in sentences:
            found_idx = X[sample_idx][start_idx:].find(cur_sent)
            if found_idx < 0:
                raise ValueError('The text `{0}` cannot be tokenized by sentences!'.format(X[sample_idx]))
            found_idx += start_idx
            bounds_of_sentences.append((found_idx, found_idx + len(cur_sent)))
            start_idx = (found_idx + len(cur_sent))
        for entity_type in y[sample_idx].keys():
            for entity_bounds in y[sample_idx][entity_type]:
                first_sentence_idx = -1
                min_distance = None
                for idx in range(len(bounds_of_sentences)):
                    if (bounds_of_sentences[idx][0] <= entity_bounds[0]) and \
                            (entity_bounds[0] < bounds_of_sentences[idx][1]):
                        first_sentence_idx = idx
                        break
                    if entity_bounds[0] < bounds_of_sentences[idx][0]:
                        if min_distance is None:
                            min_distance = bounds_of_sentences[idx][0] - entity_bounds[0]
                            first_sentence_idx = idx
                        else:
                            if (bounds_of_sentences[idx][0] - entity_bounds[0]) < min_distance:
                                min_distance = bounds_of_sentences[idx][0] - entity_bounds[0]
                                first_sentence_idx = idx
                if first_sentence_idx < 0:
                    raise ValueError('The `{0}` entity with bounds {1} cannot be found in the sentence list!'.format(
                        entity_type, entity_bounds))
                last_sentence_idx = first_sentence_idx + 1
                while last_sentence_idx < len(bounds_of_sentences):
                    if bounds_of_sentences[last_sentence_idx][0] >= entity_bounds[1]:
                        break
                    last_sentence_idx += 1
                bounds_of_united_sentence = (
                    bounds_of_sentences[first_sentence_idx][0],
                    bounds_of_sentences[last_sentence_idx - 1][1]
                )
                for _ in range(last_sentence_idx - first_sentence_idx - 1):
                    del bounds_of_sentences[first_sentence_idx]
                bounds_of_sentences[first_sentence_idx] = bounds_of_united_sentence
        entities_in_sentences = [dict() for _ in range(len(bounds_of_sentences))]
        for entity_type in y[sample_idx].keys():
            for entity_bounds in y[sample_idx][entity_type]:
                sentence_idx = -1
                min_distance = None
                for idx in range(len(bounds_of_sentences)):
                    if (bounds_of_sentences[idx][0] <= entity_bounds[0]) and \
                            (entity_bounds[0] < bounds_of_sentences[idx][1]):
                        sentence_idx = idx
                        break
                    if entity_bounds[0] < bounds_of_sentences[idx][0]:
                        if min_distance is None:
                            min_distance = bounds_of_sentences[idx][0] - entity_bounds[0]
                            sentence_idx = idx
                        else:
                            if (bounds_of_sentences[idx][0] - entity_bounds[0]) < min_distance:
                                min_distance = bounds_of_sentences[idx][0] - entity_bounds[0]
                                sentence_idx = idx
                if sentence_idx < 0:
                    raise ValueError('The `{0}` entity with bounds {1} cannot be found in the sentence list!'.format(
                        entity_type, entity_bounds))
                if (entity_bounds[0] >= bounds_of_sentences[sentence_idx][0]) and \
                        (entity_bounds[1] <= bounds_of_sentences[sentence_idx][1]):
                    new_entity_bounds = entity_bounds
                elif (entity_bounds[0] < bounds_of_sentences[sentence_idx][0]) and \
                        (entity_bounds[1] > bounds_of_sentences[sentence_idx][1]):
                    new_entity_bounds = bounds_of_sentences[sentence_idx]
                elif entity_bounds[0] < bounds_of_sentences[sentence_idx][0]:
                    new_entity_bounds = (bounds_of_sentences[sentence_idx][0], entity_bounds[1])
                else:
                    new_entity_bounds = (entity_bounds[0], bounds_of_sentences[sentence_idx][1])
                new_entity_bounds = (
                    new_entity_bounds[0] - bounds_of_sentences[sentence_idx][0],
                    new_entity_bounds[1] - bounds_of_sentences[sentence_idx][0],
                )
                entities_in_sentences[sentence_idx][entity_type] = \
                    entities_in_sentences[sentence_idx].get(entity_type, []) + [new_entity_bounds]
        for sentence_idx in range(len(entities_in_sentences)):
            for entity_type in entities_in_sentences[sentence_idx].keys():
                entities_in_sentences[sentence_idx][entity_type] = sorted(
                    entities_in_sentences[sentence_idx][entity_type]
                )
        X_new += [X[sample_idx][sent_start:sent_end] for sent_start, sent_end in bounds_of_sentences]
        y_new += entities_in_sentences
    if isinstance(X, tuple):
        X_new = tuple(X_new)
    elif isinstance(X, np.ndarray):
        X_new = np.array(X_new, dtype=object)
    if isinstance(y, tuple):
        y_new = tuple(y_new)
    elif isinstance(y, np.ndarray):
        y_new = np.array(y_new, dtype=object)
    return X_new, y_new
