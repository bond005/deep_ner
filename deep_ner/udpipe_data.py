import re
from typing import Set

import spacy_udpipe
from spacy_udpipe import UDPipeLanguage


UNIVERSAL_POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON',
                      'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
UNIVERSAL_DEPENDENCIES = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'auxpass', 'case', 'cc', 'ccomp',
                          'compound', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'discourse', 'dislocated',
                          'dobj', 'expl', 'fixed', 'flat', 'foreign', 'goeswith', 'gov', 'iobj', 'list', 'mark', 'mwe',
                          'name', 'neg', 'nmod', 'nsubj', 'nsubjpass', 'nummod', 'obj', 'obl', 'orphan', 'parataxis',
                          'pass', 'punct', 'relcl', 'remnant', 'reparandum', 'root', 'vocative', 'xcomp']


def create_udpipe_pipeline(lang: str) -> UDPipeLanguage:
    try:
        pipeline = spacy_udpipe.load(lang)
    except:
        spacy_udpipe.download(lang)
        pipeline = spacy_udpipe.load(lang)
    if pipeline is None:
        del pipeline
        raise ValueError('The `{0}` language cannot be loaded for the UDPipe!')
    return pipeline


def prepare_dependency_tag(source_tag: str) -> Set[str]:
    re_for_splitting = re.compile('[:\-]+')
    tags = {source_tag.lower().replace(':', '').replace('-', '')}
    for cur in filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip().lower(),
                                                    re_for_splitting.split(source_tag))):
        tags.add(cur)
    return tags
