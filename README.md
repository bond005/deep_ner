[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

# deep_ner

**Deep-NER**: named entity recognizer based on deep neural networks and transfer learning.

The goal of this project is creation of a simple Python package with the sklearn-like interface for solution of different named entity recognition tasks in case number of labeled texts is very small (not greater than several thousands). Special neural network language models named as [ELMo](https://arxiv.org/abs/1802.05365) (**E**mbeddings from **L**anguage **Mo**dels) and [BERT](https://arxiv.org/abs/1810.04805) (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) ensure this possibility, because these language models were pre-trained on large text corpora and so they can select deep semantic features from text, reduce the influence of the homonymy problem and the like.


Installing
----------


For installation you need to Python 3.6 or later. To install this project on your local machine, you should run the following commands in the Terminal:

```
git clone https://github.com/bond005/deep_ner.git
cd deep_ner
pip install .
```

If you want to install the **Deep-NER** into a some virtual environment, than you don't need to use `sudo`, but before installing you have to activate this virtual environment (for example, using `source /path/to/your/python/environment/bin/activate` in the command prompt).

You can also run the tests

```
python setup.py test
```

Also, you can install the **Deep-NER** from the [PyPi](https://pypi.org/project/deep-ner) using the following command:

```
pip install deep-ner
```

Usage
-----


After installing the **Deep-NER** can be used as Python package in your projects. It includes two variants of NER: the **ELMo-NER** and the **BERT-NER**. Distributional semantics methods, based on the deep learning (BERT or ELMo), are used for semantic features extraction from some text, after that [**C**onditional **R**andom **F**ields (CRF)](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf) or their combinations with [**L**ong **S**hort **T**erm **M**emory networks (LSTM)](https://arxiv.org/abs/1508.01991) do the final classification of named entities in this text. Also, special linguistic information about tokens in the text can be used in addition to the distributional semantics, namely:

1) token shapes, which indicate the orphographic information (see [Finkel et al., 2005](https://nlp.stanford.edu/pubs/finkel2005boundaries.pdf));
2) part-of-speech tags, which represent the morphology of the specified language (the composition of these tags corresponds to the [Universal POS tags](http://universaldependencies.org/docs/u/pos/index.html));
3) structure of syntactic dependency tree, which is described using with [Universal Dependency Relations](https://universaldependencies.org/u/dep/).


You can create new named entity recognizer using the [cased English BERT](https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1) as follows:

```
from deep_ner.bert_ner import BERT_NER  # import the BERT-NER module
ner = BERT_NER(bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
               udpipe_lang='en', use_additional_features = True,
               validation_fraction=0.0)  # create new named entity recognizer for English language with BERT and additional features
```

If you want to use some special finetuned BERT, located on your disk, then you have to set path to the BERT directory in special variable (a Python class attribute) `BERT_NER.PATH_TO_BERT`. For example:

```
from deep_ner.bert_ner import BERT_NER  # import the BERT-NER module
BERT_NER.PATH_TO_BERT = '/path/to/the/unpacked/BERT'
ner = BERT_NER(validation_fraction=0.0, udpipe_lang='en')  # create new named entity recognizer using this BERT
```

**Important note**: name of directory with unpacked files of your BERT model must contain such subphrases as `cased` or `uncased` (for example, `cased_L-12_H-768_A-12`, `rubert_uncased_L-24_H-768_A-24` and the like). Presence of `cased` substring implies that the true case of processed texts is preserved, and occurrence of `uncased` substring is corresponded to processing of texts in lower-case only.

You can find archives with pre-trained BERT, appropriate for you, on websites of various scientific projects (so, good BERT model for Russian language is available as part of the DeepPavlov project http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz).

For building of NER based on the ELMo model you have to import another module and create object of the ELMo_NER class, like this:

```
from deep_ner.elmo_ner import ELMo_NER  # import the ELMo-NER module
ner = ELMo_NER(elmo_hub_module_handle='https://tfhub.dev/google/elmo/3', udpipe_lang='en')  # create new named entity recognizer for English language
```

Pre-trained ELMo for Russian language as customary TFHub module are granted by the iPavlov project: http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz

A neural network architecture of the **Deep-NER** consists of a deep pre-trained base (ELMo or BERT) and simple head trained from scratch (CRF or BiLSTM-CRF after BERT and CRF only after ELMo). For including BiLSTM layer in the **BERT-NER** you have to specify number of the LSTM units, for example:

```
from deep_ner.bert_ner import BERT_NER  # import the BERT-NER module
ner = BERT_NER(bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1', lstm_units=256, validation_fraction=0.0, udpipe_lang='en')  # create new BERT-NER with BiLSTM-CRF head instead of simple CRF head
```


For training of all types of the **Deep-NER** (based on BERT or ELMo) you have to prepare list of short texts and do manual labeling of these texts by named entities. Result of text labeling should like as list of dictionaries, corresponding to list of source texts. Each dictionary in this list contains named entities and their bounds (initial and final character indices) in source text. For example:

```
texts_for_training = [
    'MIPT is one of the leading Russian universities in the areas of physics and technology.',
    'MIPT has a very rich history. Its founders included academicians Pyotr Kapitsa, Nikolay Semenov and Sergey Khristianovich.',
    'The main MIPT campus is located in Dolgoprudny, a northern suburb of Moscow.'
]
named_entity_labels_for_training = [
    {
        'ORG': [(0, 4)]
    },
    {
        'ORG': [(0, 4)],
        'PERSON': [(65, 78), (80, 95), (100, 121)]
    },
    {
        'LOCATION': [(35, 46), (69, 75)],
        'ORG': [(9, 13)]
    }
]
ner.fit(texts_for_training, named_entity_labels_for_training)
``` 

Predicted named entities for specified texts list also are presented in same format (as list of dictionaries):

```
texts_for_testing = [
    'Novosibirsk state university is located in the world-famous scientific center – Akademgorodok.',
    '"It’s like in the woods!" – that's what people say when they come to Akademgorodok for the first time.',
    'NSU’s new building will remind you of R2D2, the astromech droid and hero of the Star Wars saga'
]
results_of_prediction = ner.predict(texts_for_testing)
```

Quality evaluating of trained NER is based on examination all possible correspondences between a predicted labeling of named entities and the gold-standard labeling and choosing the best matches. After than special F1-score is calculated for all matched pairs "predicted entity" - "gold-standard entity" (fuzzy matching is taken into consideration too):

```
true_labels_for_testing = [
    {
        'LOCATION': [(80, 93)],
        'ORG': [(0, 28)]
    },
    {
        'LOCATION': [(69, 82)]
    },
    {
        'ORG': [(0, 3)],
        'PERSON': [(38, 42)]
    }
]
f1, precision, recall = ner.calculate_prediction_quality(true_labels_for_testing,
                                                         true_labels_for_testing,
                                                         ner.classes_list_)
``` 

If you want to train or evaluate Deep-NER using labeled dataset created by the [Brat annotation tool](http://brat.nlplab.org/), then you can load this dataset for  the Deep-NER by means of the function `load_dataset_from_brat` as follows:

```
from deep_ner.utils import load_dataset_from_brat

texts_for_training, named_entity_labels_for_training = load_dataset_from_brat('/path/to/the/directory/with/brat/results')

``` 

You can serialize and de-serialize any object of the **BERT-NER** or the **ELMo-NER** class using the `pickle` module from Python’s standard library:

```
import pickle

with open('path/to/file/with/model', 'wb') as fp:
    pickle.dump(ner, fp)

with open('path/to/file/with/model', 'rb') as fp:
    yet_another_ner = pickle.load(fp)
```


#### Note 1


All named entity labels must be uppercased texts without spaces. Also `O` is inadmissible label of named entity (this is a special label for background tokens in text, which are not part of any named entity). For example, `ORG`, `FIRST_NAME` and `LOCATION` are correct labels of named entities, and `Location`, `FIRST NAME` and `O` are wrong labels of named entities.


#### Note 2


You have to use short texts such as sentences or small paragraphs, because long texts will be processed worse. If you train the **Deep-NER** on corpus of long texts, then the training can be converged slowly. If you use the **Deep-NER**, trained on short texts, for recognizing of long text, then only some initial words of this text can be tagged, and remaining words at the end of text will not be considered by algorithm. Besides, you need to use a very large volume of RAM for processing of long texts.

For solving of above-mentioned problem you can split long texts by shorter sentences using well-known NLP libraries such as [NLTK](http://www.nltk.org/api/nltk.tokenize.html?highlight=sent_tokenize#nltk.tokenize.sent_tokenize) or [SpaCy](https://spacy.io/api/token#is_sent_start). Also, for splitting of long texts together with their manual annotations you can use the special function `divide_dataset_by_sentences` from the `utils` module of this package.


#### Note 3


A neural base (BERT or ELMo) can be tuned together with a CRF-based (or LSTM-CRF-based) neural head, or the neural base can be frozen, i.e. it is used as feature extractor only. The special argument of the constructor, named as `finetune_bert` for BERT and `finetune_elmo` for ELMo, specifies a corresponding mode. About tuning and freezing, you can read [an interesting paper, written by Sebastian Ruder and his colleagues](https://arxiv.org/abs/1903.05987).


Demo
----

### Demo for Russian language (the FactRuEval-2016 dataset)


In the `demo` subdirectory you can see **demo_elmo_factrueval2016.py** and **demo_bert_factrueval2016.py** - examples of experiments on the FactRuEval-2016 text corpus, which is part of special competition devoted to named entity recognition and fact extraction in Russian (it is described in the paper [FactRuEval 2016: Evaluation of Named Entity Recognition and Fact Extraction Systems for Russian](http://www.dialog-21.ru/media/3430/starostinaetal.pdf)).

In first example, based on the ELMo, you can train and evaluate the Russian NER with the DeepPavlov's ELMo model:

```
PYTHONPATH=$PWD python -u demo/demo_elmo_factrueval2016.py \
    -d /home/user/factRuEval-2016 \
    -m /home/user/FactRuEval2016_results/elmo_and_crf.pkl \
    -r /home/user/FactRuEval2016_results/results_of_elmo_and_crf \
    --max_epochs 1000 --batch 32
```

where:

- `/home/user/factRuEval-2016` is path to the FactRuEval-2016 repository cloned from https://github.com/dialogue-evaluation/factRuEval-2016;
- `/home/user/FactRuEval2016_results/elmo_and_crf.pkl` is path to binary file into which the ELMo-NER will be written after its training;
- `/home/user/FactRuEval2016_results/results_of_elmo_and_crf` is path to directory with recognition results.

In second example, based on the BERT, you can train and evaluate the Russian model of NER with means of [multilingual BERT](https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1) as base neural model and CRF (or BiLSTM-CRF) as final classifier.

You can run this example on the command prompt in this way ('frozen' BERT and BiLSTM-CRF head):

```
PYTHONPATH=$PWD python -u demo/demo_bert_factrueval2016.py \
    -d /home/user/factRuEval-2016 \
    -m /home/user/FactRuEval2016_results/bert_and_crf.pkl \
    -r /home/user/FactRuEval2016_results/results_of_bert_and_crf \
    --max_epochs 1000 --lstm 256 --batch 32
```

or in that way ('unfrozen' BERT and CRF head):

```
PYTHONPATH=$PWD python -u demo/demo_bert_factrueval2016.py \
    -d /home/user/factRuEval-2016 \
    -m /home/user/FactRuEval2016_results/bert_lstm_and_crf.pkl \
    -r /home/user/FactRuEval2016_results/results_of_bert_lstm_and_crf \
    --max_epochs 1000 --batch 8 --finetune_bert
```

where:

- `/home/user/factRuEval-2016` is path to the FactRuEval-2016 repository cloned from https://github.com/dialogue-evaluation/factRuEval-2016;
- `/home/user/FactRuEval2016_results/tuned_bert_and_crf.pkl` is path to binary file into which the BERT-NER will be written after its training;
- `/home/user/FactRuEval2016_results/results_of_tuned_bert_and_crf` is path to directory with recognition results.

In first from above-mentioned ways you will train the neural head only, and BERT will be 'frozen'. But in second way you will train both the BERT base and the CRF head. Second way is more hard and time-consuming, but it allows you to achieve better results. 

After recognition results calculation we can use the special FactRuEval-2016 script for evaluating of these results:

```
cd /home/user/factRuEval-2016/scripts
python t1_eval.py -t ~/FactRuEval2016_results/results_of_tuned_bert_and_crf -s ../testset/ -l
``` 

Quality score calculated by this script may differ from value returned by the `calculate_prediction_quality` method of the `quality` module.

### Demo for English language (the CoNLL-2003 dataset)

There is special demo script **demo_bert_conll2003.py** in the `demo` subdirectory. This demo script shows apllying of the `BERT_NER` for named entity recognition on the data of well-known labeled corpus [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/). Execution way of this script is same as one for the FactRuEval-2016 demo script, except that three things:

- parameter `-d`, which specifies path to the FactRuEval-2016 repository, is not used for CoNLL-2003;
- three parameters `-t`, `-d` and `-e` are used instead of above-mentioned parameter for specifing of data files (these parameters correspond to pathes to the training data file, validation data file and the final testing data file);
- parameter `-r` does not point to the directory, but it points to the file with recognition results which will be generated by the demo script.

You can found prepared files with the CoNLL-2003 data for training, validation and final testing in the archive http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz.

Internal structure of the results file, generated after completion of the demo script, is described on the web-page [Output Example conlleval](https://www.clips.uantwerpen.be/conll2000/chunking/output.html). Also, example of using of evaluation script named as `conlleval` is adduced on this page. This evaluation script is written with Perl, and it is avaliable for free downloading by the web link https://www.clips.uantwerpen.be/conll2003/ner/bin/conlleval. I recommend to use this script that to evaluate your  results for the CoNLL-2003 dataset, in spite of the fact that the script **demo_bert_conll2003.py** does evaluation of recognition results too (algorithms of F1 calculation in the CoNLL-2003 script and in my demo script are different).

Breaking Changes
-----

**Breaking changes in version 0.0.5**
- Tokenization bug in **BERT_NER** and **ELMo_NER** has been fixed.

**Breaking changes in version 0.0.4**
- List of additional features has been expanded(morphological and syntactical features have been added, besides word shapes, i.e. orphographic features). Also, you have gotten possibility to enable/disable these additional features by the constructor parameter. 

**Breaking changes in version 0.0.3**
- a little misprint in the setup.py has been fixed.

**Breaking changes in version 0.0.2**
- serialization and deserialization with the `pickle` have been improved, and dependencies have been updated.

**Breaking changes in version 0.0.1**
- initial (alpha) version of the Deep-NER has been released.

License
-------

The **Deep-NER** (`deep-ner`) is Apache 2.0 - licensed.


Acknowledgment
--------------


The work was supported by National Technology Initiative and PAO Sberbank project ID 0000000007417F630002.