from itertools import chain

from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer

lemmer = None


def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        # As default pos in lemmatization is Noun
        return wn.NOUN


def synonym(word, pos):
    """

    :params word: str:
    :return:
    """
    synonyms = wn.synsets(word, get_wordnet_pos(pos))
    lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))

    return lemmas


def hypernym(word, pos):
    """

    :params word: str:
    :return:
    """
    synonyms = wn.synsets(word, get_wordnet_pos(pos))
    hypernyms = []
    for syn in synonyms:
        hyper = syn.hypernyms()
        hypernyms += hyper
    lemmas = [h.offset() for h in hypernyms]

    return lemmas


def lemmatize(word, pos):
    global lemmer
    if lemmer is None:
        lemmer = WordNetLemmatizer()

    return lemmer.lemmatize(word, get_wordnet_pos(pos))


def make_entities():
    with open('../data/wordnet/wordnet-entities.txt', 'w') as f:
        for s in wn.all_synsets():
            f.write('{}\t{}'.format(str(s.offset()), s.name()))
            f.write('\n')


# make_entities()

