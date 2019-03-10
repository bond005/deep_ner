from setuptools import setup, find_packages

import elmo_ner


long_description = '''
ELMo-NER
============

ELMo-NER: named entity recognizer based on ELMo and CRF.

The goal of this project is creation of a simple Python package with
the sklearn-like interface for solution of different named entity 
recognition tasks in case number of labeled texts is very small (not 
greater than several thousands). Special neural network language model 
named as `ELMo<https://arxiv.org/pdf/1802.05365.pdf>`_ (Embeddings from Language Models)
ensures this possibility, because this language model was pre-trained on
large text corpora and so it can select deep semantic features from text.

Getting Started
---------------

Installing
~~~~~~~~~~

To install this project on your local machine, you should run the
following commands in Terminal:

.. code::

    git clone https://github.com/bond005/elmo_ner.git
    cd elmo_ner
    sudo python setup.py install

You can also run the tests:

.. code::

    python setup.py test

Usage
~~~~~

After installing the ELMo-NER can be used as Python package in your
projects. For example:

.. code::

    from elmo_ner import ELMo_NER  # import the ELMo-NER package
    ner = ELMo_NER('https://tfhub.dev/google/elmo/2')  # create new named entity recognizer

To see the work of the ELMo-NER on the FactRuEval-2016 dataset, you can
run a demo

.. code::
 
    PYTHONPATH=$PWD python -u demo/demo_factrueval2016.py -d /home/user/factRuEval-2016 -m /home/user/FactRuEval2016_results/elmo_and_crf.pkl -r /home/user/FactRuEval2016_results/results_of_elmo_and_crf --max_epochs 1000 --batch 32

or (with saving model after its training):

.. code::
 
    PYTHONPATH=$PWD python -u demo/demo_factrueval2016.py -d /home/user/factRuEval-2016 -m /home/user/FactRuEval2016_results/elmo_and_crf.pkl -r /home/user/FactRuEval2016_results/results_of_elmo_and_crf --max_epochs 1000 --batch 8 --finetune_elmo

In this demo, the ELMo-NER learns to recognize the named entities LOCATION,
ORG and PERSON in Russian texts. If you specify the ``--finetune_elmo``
flag, then the CRF head will be trained with the ELMo base, else the ELMo
base will be frozen.

'''

setup(
    name='elmo-ner',
    version=elmo_ner.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='ELMo-NER: named entity recognizer based on ELMo and CRF',
    long_description=long_description,
    url='https://github.com/bond005/elmo_ner',
    author='Ivan Bondarenko',
    author_email='bond005@yandex.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['elmo', 'ner', 'crf', 'nlp', 'tensorflow', 'scikit-learn'],
    install_requires=['nltk==3.4', 'numpy==1.16.1', 'scikit-learn==0.20.2', 'scipy==1.2.0',
                      'tensorboard==1.12.2', 'tensorflow-gpu==1.12.0'],
    test_suite='tests'
)
