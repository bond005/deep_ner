from setuptools import setup, find_packages

import deep_ner


long_description = '''
Deep-NER
============

Named entity recognizer based on ELMo or BERT as feature extractor and
CRF as final classifier.

The goal of this project is creation of a simple Python package with
the sklearn-like interface for solution of different named entity 
recognition tasks in case number of labeled texts is very small (not 
greater than several thousands). Special neural network language models 
named as ELMo (Embeddings from Language Models) and BERT (Bidirectional
Encoder Representations from Transformers) ensure this possibility,
because these language model were pre-trained on large text corpora and
so they can select deep semantic features from text.
'''

setup(
    name='deep-ner',
    version=deep_ner.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='Deep-NER: named entity recognizer based on ELMo or BERT as embeddings and CRF as final classifier',
    long_description=long_description,
    url='https://github.com/bond005/deep_ner',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['elmo', 'bert', 'ner', 'crf', 'nlp', 'tensorflow', 'scikit-learn'],
    install_requires=['nltk==3.4.5', 'numpy==1.18.1', 'scikit-learn==0.22.1', 'scipy==1.4.1', 'tensorboard==1.15.0',
                      'tensorflow==1.15.0', 'tensorflow-hub==0.8.0', 'bert-tensorflow==1.0.1', 'spacy-udpipe==0.2.0',
                      'spacy==2.2.3', 'pymorphy2==0.8', 'rusenttokenize==0.0.5'],
    test_suite='tests'
)
