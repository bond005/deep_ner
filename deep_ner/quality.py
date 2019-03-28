import copy
from typing import Dict, Union, List, Tuple

import numpy as np


def calc_similarity_between_entities(gold_entity: Tuple[int, int], predicted_entity: Tuple[int, int]) -> \
        Tuple[float, int, int, int]:
    if gold_entity[1] <= predicted_entity[0]:
        res = 0.0
        tp = 0
        fp = predicted_entity[1] - predicted_entity[0]
        fn = gold_entity[1] - gold_entity[0]
    elif predicted_entity[1] <= gold_entity[0]:
        res = 0.0
        tp = 0
        fp = predicted_entity[1] - predicted_entity[0]
        fn = gold_entity[1] - gold_entity[0]
    else:
        if (gold_entity[0] == predicted_entity[0]) and (gold_entity[1] == predicted_entity[1]):
            tp = gold_entity[1] - gold_entity[0]
            fp = 0
            fn = 0
            res = 1.0
        elif gold_entity[0] == predicted_entity[0]:
            if gold_entity[1] > predicted_entity[1]:
                tp = predicted_entity[1] - predicted_entity[0]
                fp = 0
                fn = gold_entity[1] - predicted_entity[1]
            else:
                tp = gold_entity[1] - gold_entity[0]
                fp = predicted_entity[1] - gold_entity[1]
                fn = 0
            res = tp / float(tp + fp + fn)
        elif gold_entity[1] == predicted_entity[1]:
            if gold_entity[0] < predicted_entity[0]:
                tp = predicted_entity[1] - predicted_entity[0]
                fp = 0
                fn = predicted_entity[0] - gold_entity[0]
            else:
                tp = gold_entity[1] - gold_entity[0]
                fp = gold_entity[0] - predicted_entity[0]
                fn = 0
            res = tp / float(tp + fp + fn)
        elif gold_entity[0] < predicted_entity[0]:
            if gold_entity[1] > predicted_entity[1]:
                tp = predicted_entity[1] - predicted_entity[0]
                fp = 0
                fn = (predicted_entity[0] - gold_entity[0]) + (gold_entity[1] - predicted_entity[1])
            else:
                tp = gold_entity[1] - predicted_entity[0]
                fp = predicted_entity[1] - gold_entity[1]
                fn = predicted_entity[0] - gold_entity[0]
            res = tp / float(tp + fp + fn)
        else:
            if gold_entity[1] < predicted_entity[1]:
                tp = gold_entity[1] - gold_entity[0]
                fp = (gold_entity[0] - predicted_entity[0]) + (predicted_entity[1] - gold_entity[1])
                fn = 0
            else:
                tp = predicted_entity[1] - gold_entity[0]
                fp = gold_entity[0] - predicted_entity[0]
                fn = gold_entity[1] - predicted_entity[1]
            res = tp / float(tp + fp + fn)
    return res, tp, fp, fn


def comb(n: int, k: int):
    d = list(range(0, k))
    yield d
    while True:
        i = k - 1
        while i >= 0 and d[i] + k - i + 1 > n:
            i -= 1
        if i < 0:
            return
        d[i] += 1
        for j in range(i + 1, k):
            d[j] = d[j - 1] + 1
        yield d


def find_pairs_of_named_entities(true_entities: List[int], predicted_entities: List[int],
                                 similarity_dict: Dict[Tuple[int, int], Tuple[float, int, int, int]]) -> \
        Tuple[float, List[Tuple[int, int]]]:
    best_similarity_sum = 0.0
    n_true = len(true_entities)
    n_predicted = len(predicted_entities)
    best_pairs = []
    if n_true == n_predicted:
        best_pairs = list(filter(lambda it1: it1 in similarity_dict, map(lambda it2: (it2, it2), range(n_true))))
        best_similarity_sum = sum(map(lambda it: similarity_dict[it][0], best_pairs))
    else:
        N_MAX_COMB = 10
        counter = 1
        if n_true < n_predicted:
            for c in comb(n_predicted, n_true):
                pairs = list(filter(
                    lambda it1: it1 in similarity_dict,
                    map(lambda it2: (it2, c[it2]), range(n_true))
                ))
                if len(pairs) > 0:
                    similarity_sum = sum(map(lambda it: similarity_dict[it][0], pairs))
                else:
                    similarity_sum = 0.0
                if similarity_sum > best_similarity_sum:
                    best_similarity_sum = similarity_sum
                    best_pairs = copy.deepcopy(pairs)
                del pairs
                counter += 1
                if counter > N_MAX_COMB:
                    break
            pairs = []
            used_indices = set()
            for true_idx in range(n_true):
                best_pred_idx = None
                best_similarity = -1.0
                for pred_idx in filter(lambda it: it not in used_indices, range(n_predicted)):
                    pair_candidate = (true_idx, pred_idx)
                    if pair_candidate in similarity_dict:
                        if similarity_dict[pair_candidate][0] > best_similarity:
                            best_similarity = similarity_dict[pair_candidate][0]
                            best_pred_idx = pred_idx
                if best_pred_idx is None:
                    break
                used_indices.add(best_pred_idx)
                pairs.append((true_idx, best_pred_idx))
            if len(pairs) > 0:
                similarity_sum = sum(map(lambda it: similarity_dict[it][0], pairs))
            else:
                similarity_sum = 0.0
            if similarity_sum > best_similarity_sum:
                best_similarity_sum = similarity_sum
                best_pairs = copy.deepcopy(pairs)
            del pairs
            del used_indices
        else:
            for c in comb(n_true, n_predicted):
                pairs = list(filter(
                    lambda it1: it1 in similarity_dict,
                    map(lambda it2: (c[it2], it2), range(n_predicted))
                ))
                if len(pairs) > 0:
                    similarity_sum = sum(map(lambda it: similarity_dict[it][0], pairs))
                else:
                    similarity_sum = 0.0
                if similarity_sum > best_similarity_sum:
                    best_similarity_sum = similarity_sum
                    best_pairs = copy.deepcopy(pairs)
                del pairs
                counter += 1
                if counter > N_MAX_COMB:
                    break
            pairs = []
            used_indices = set()
            for pred_idx in range(n_predicted):
                best_true_idx = None
                best_similarity = -1.0
                for true_idx in filter(lambda it: it not in used_indices, range(n_true)):
                    pair_candidate = (true_idx, pred_idx)
                    if pair_candidate in similarity_dict:
                        if similarity_dict[pair_candidate][0] > best_similarity:
                            best_similarity = similarity_dict[pair_candidate][0]
                            best_true_idx = true_idx
                if best_true_idx is None:
                    break
                used_indices.add(best_true_idx)
                pairs.append((best_true_idx, pred_idx))
            if len(pairs) > 0:
                similarity_sum = sum(map(lambda it: similarity_dict[it][0], pairs))
            else:
                similarity_sum = 0.0
            if similarity_sum > best_similarity_sum:
                best_similarity_sum = similarity_sum
                best_pairs = copy.deepcopy(pairs)
            del pairs
            del used_indices
    return best_similarity_sum, best_pairs


def calculate_prediction_quality(true_entities: Union[list, tuple, np.array],
                                 predicted_entities: List[Dict[str, List[Tuple[int, int]]]], classes_list: tuple) -> \
        Tuple[float, float, float, Dict[str, Tuple[float, float, float]]]:
    true_entities_ = []
    predicted_entities_ = []
    n_samples = len(true_entities)
    quality_by_entity_classes = dict()
    for sample_idx in range(n_samples):
        instant_entities = dict()
        for ne_class in true_entities[sample_idx]:
            entities_list = []
            for entity_bounds in true_entities[sample_idx][ne_class]:
                entities_list.append((entity_bounds[0], entity_bounds[1]))
            entities_list.sort()
            instant_entities[ne_class] = entities_list
            del entities_list
        true_entities_.append(instant_entities)
        del instant_entities
        instant_entities = dict()
        for ne_class in predicted_entities[sample_idx]:
            entities_list = []
            for entity_bounds in predicted_entities[sample_idx][ne_class]:
                entities_list.append((entity_bounds[0], entity_bounds[1]))
            entities_list.sort()
            instant_entities[ne_class] = entities_list
            del entities_list
        predicted_entities_.append(instant_entities)
        del instant_entities
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for ne_class in classes_list:
        tp_for_ne = 0
        fp_for_ne = 0
        fn_for_ne = 0
        for sample_idx in range(n_samples):
            if (ne_class in true_entities_[sample_idx]) and \
                    (ne_class in predicted_entities_[sample_idx]):
                n1 = len(true_entities_[sample_idx][ne_class])
                n2 = len(predicted_entities_[sample_idx][ne_class])
                similarity_dict = dict()
                for idx1, true_bounds in enumerate(true_entities_[sample_idx][ne_class]):
                    for idx2, predicted_bounds in enumerate(predicted_entities_[sample_idx][ne_class]):
                        similarity, tp, fp, fn = calc_similarity_between_entities(
                            true_bounds, predicted_bounds
                        )
                        if tp > 0:
                            similarity_dict[(idx1, idx2)] = (similarity, tp, fp, fn)
                similarity, pairs = find_pairs_of_named_entities(list(range(n1)), list(range(n2)), similarity_dict)
                tp_for_ne += sum(map(lambda it: similarity_dict[it][1], pairs))
                fp_for_ne += sum(map(lambda it: similarity_dict[it][2], pairs))
                fn_for_ne += sum(map(lambda it: similarity_dict[it][3], pairs))
                unmatched_std = sorted(list(set(range(n1)) - set(map(lambda it: it[0], pairs))))
                for idx1 in unmatched_std:
                    fn_for_ne += (true_entities_[sample_idx][ne_class][idx1][1] -
                                  true_entities_[sample_idx][ne_class][idx1][0])
                unmatched_test = sorted(list(set(range(n2)) - set(map(lambda it: it[1], pairs))))
                for idx2 in unmatched_test:
                    fp_for_ne += (predicted_entities_[sample_idx][ne_class][idx2][1] -
                                  predicted_entities_[sample_idx][ne_class][idx2][0])
            elif ne_class in true_entities_[sample_idx]:
                for entity_bounds in true_entities_[sample_idx][ne_class]:
                    fn_for_ne += (entity_bounds[1] - entity_bounds[0])
            elif ne_class in predicted_entities_[sample_idx]:
                for entity_bounds in predicted_entities_[sample_idx][ne_class]:
                    fp_for_ne += (entity_bounds[1] - entity_bounds[0])
        tp_total += tp_for_ne
        fp_total += fp_for_ne
        fn_total += fn_for_ne
        precision_for_ne = tp_for_ne / float(tp_for_ne + fp_for_ne) if tp_for_ne > 0 else 0.0
        recall_for_ne = tp_for_ne / float(tp_for_ne + fn_for_ne) if tp_for_ne > 0 else 0.0
        if (precision_for_ne + recall_for_ne) > 0.0:
            f1_for_ne = 2 * precision_for_ne * recall_for_ne / (precision_for_ne + recall_for_ne)
        else:
            f1_for_ne = 0.0
        quality_by_entity_classes[ne_class] = (f1_for_ne, precision_for_ne, recall_for_ne)
    precision = tp_total / float(tp_total + fp_total) if tp_total > 0 else 0.0
    recall = tp_total / float(tp_total + fn_total) if tp_total > 0 else 0.0
    if (precision + recall) > 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return f1, precision, recall, quality_by_entity_classes
