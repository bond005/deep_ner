from argparse import ArgumentParser
import logging
import os
import pickle
import sys
from typing import Union

try:
    from deep_ner.bert_ner import BERT_NER, bert_ner_logger
    from deep_ner.utils import load_dataset_from_bio, save_dataset_as_bio
    from deep_ner.quality import calculate_prediction_quality
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from deep_ner.bert_ner import BERT_NER, bert_ner_logger
    from deep_ner.utils import load_dataset_from_bio, save_dataset_as_bio
    from deep_ner.quality import calculate_prediction_quality


def train(train_file_name: str, valid_file_name: str, split_by_paragraphs: bool, bert_will_be_tuned: bool,
          lstm_layer_size: Union[int, None], max_epochs: int, batch_size: int, gpu_memory_frac: float,
          model_name: str) -> BERT_NER:
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            recognizer = pickle.load(fp)
        assert isinstance(recognizer, BERT_NER)
        print('The NER has been successfully loaded from the file `{0}`...'.format(model_name))
        print('')
    else:
        X_train, y_train = load_dataset_from_bio(
            train_file_name,
            paragraph_separators=({'-DOCSTART-'} if split_by_paragraphs else None)
        )
        X_val, y_val = load_dataset_from_bio(
            valid_file_name,
            paragraph_separators=({'-DOCSTART-'} if split_by_paragraphs else None)
        )
        print('The CoNLL-2003 data for training and validation have been loaded...')
        print('Number of samples for training is {0}.'.format(len(y_train)))
        print('Number of samples for validation is {0}.'.format(len(y_val)))
        print('')
        if BERT_NER.PATH_TO_BERT is None:
            bert_hub_module_handle = 'https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1'
        else:
            bert_hub_module_handle = None
        recognizer = BERT_NER(
            finetune_bert=bert_will_be_tuned, batch_size=batch_size, l2_reg=1e-3,
            bert_hub_module_handle=bert_hub_module_handle, lstm_units=lstm_layer_size, max_epochs=max_epochs,
            patience=3, gpu_memory_frac=gpu_memory_frac, verbose=True, random_seed=42,
            lr=1e-5 if bert_will_be_tuned else 1e-3
        )
        recognizer.fit(X_train, y_train, validation_data=(X_val, y_val))
        print('')
        print('The NER has been successfully fitted and saved into the file `{0}`...'.format(model_name))
        y_pred = recognizer.predict(X_val)
        print('F1 for validation data (with accounting of fuzzy mathing of true entities and predicted ones) is '
              '{0:.6f}.'.format(calculate_prediction_quality(y_val, y_pred, classes_list=recognizer.classes_list_)))
        print('')
    return recognizer


def recognize(test_file_name: str, split_by_paragraphs: bool, recognizer: BERT_NER, results_file_name: str):
    X_test, y_test = load_dataset_from_bio(
        test_file_name,
        paragraph_separators=({'-DOCSTART-'} if split_by_paragraphs else None)
    )
    print('The CoNLL-2003 data for final testing have been loaded...')
    print('Number of samples is {0}.'.format(len(y_test)))
    print('')
    y_pred = recognizer.predict(X_test)
    print('F1 for test data (with accounting of fuzzy mathing of true entities and predicted ones) is '
          '{0:.6f}.'.format(calculate_prediction_quality(y_test, y_pred, classes_list=recognizer.classes_list_)))
    print('')
    save_dataset_as_bio(test_file_name, X_test, y_pred, results_file_name, stopwords={'-DOCSTART-'})


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the NER model.')
    parser.add_argument('-t', '--train', dest='train_file_name', type=str, required=True,
                        help='Path to the file with CoNLL-2003 training data.')
    parser.add_argument('-e', '--test', dest='test_file_name', type=str, required=True,
                        help='Path to the file with CoNLL-2003 data for final testing.')
    parser.add_argument('-d', '--valid', dest='valid_file_name', type=str, required=True,
                        help='Path to the file with CoNLL-2003 data for validation.')
    parser.add_argument('-r', '--result', dest='result_name', type=str, required=True,
                        help='The file name into which all tokens and recognized BIO labels will be saved.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, required=False, default=10,
                        help='Maximal number of training epochs.')
    parser.add_argument('--lstm', dest='lstm_units', type=int, required=False, default=None,
                        help='The LSTM layer size (if it is not specified, than the LSTM layer is not used).')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the NER model.')
    parser.add_argument('--finetune_bert', dest='finetune_bert', required=False, action='store_true',
                        default=False, help='Will be the BERT and CRF finetuned together? Or the BERT will be frozen?')
    parser.add_argument('--path_to_bert', dest='path_to_bert', required=False, type=str,
                        default=None, help='Path to the BERT model (if it is not specified, than the standard '
                                           'large uncased BERT model from the TF-Hub will be used).')
    parser.add_argument('--text', dest='text_unit', type=str, choices=['sentence', 'paragraph'], required=False,
                        default='sentence', help='Text unit: sentence or paragraph.')
    args = parser.parse_args()

    train_file_name = os.path.normpath(args.train_file_name)
    test_file_name = os.path.normpath(args.test_file_name)
    valid_file_name = os.path.normpath(args.valid_file_name)
    if args.text_unit not in {'sentence', 'paragraph'}:
        raise ValueError('`{0}` is wrong value for the `text_unit` parameter!'.format(args.text_unit))
    if args.path_to_bert is None:
        path_to_bert = None
    else:
        path_to_bert = os.path.normpath(args.path_to_bert)
        if len(path_to_bert) == 0:
            raise ValueError('The BERT model cannot be contained into the current directory!')
        if not os.path.isdir(path_to_bert):
            raise ValueError('The directory `{0}` does not exist!'.format(path_to_bert))
    BERT_NER.PATH_TO_BERT = path_to_bert
    recognizer = train(train_file_name=train_file_name, valid_file_name=valid_file_name,
                       bert_will_be_tuned=args.finetune_bert, max_epochs=args.max_epochs, batch_size=args.batch_size,
                       gpu_memory_frac=args.gpu_memory_frac, model_name=os.path.normpath(args.model_name),
                       lstm_layer_size=args.lstm_units, split_by_paragraphs=(args.text_unit == 'paragraph'))
    recognize(test_file_name=test_file_name, recognizer=recognizer,
              results_file_name=os.path.normpath(args.result_name), split_by_paragraphs=(args.text_unit == 'paragraph'))


if __name__ == '__main__':
    bert_ner_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    bert_ner_logger.addHandler(handler)
    main()
