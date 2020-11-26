import razdel

from functools import lru_cache

from nltk import wordpunct_tokenize
from string import punctuation
from pymorphy2 import MorphAnalyzer


morphAnalyzer = MorphAnalyzer()


@lru_cache(50_000)
def morph_parse(w):
    parses = morphAnalyzer.parse(w)
    if not parses:
        return None
    return parses[0]


@lru_cache(50_000)
def word2pos(w):
    parse = morph_parse(w)
    if not parse:
        return None
    if not parse.tag:
        return None
    if not parse.tag.POS:
        return None
    return parse.tag.POS


def tokenize(text):
    return [t for t in wordpunct_tokenize(text.lower()) if not all(c in punctuation for c in t)]


@lru_cache(maxsize=50_000)
def word2lemma(word):
    parse = morph_parse(word)
    if not parse:
        return word
    if parse.normal_form:
        return parse.normal_form
    return word


def prepare_definition(text, first_sentence=False):
    """ replace all words in the definition with uppercase lemmas """
    if first_sentence:
        text = list(razdel.sentenize(text))[0].text
    if '—' not in text:
        return ''
    l, r = text.split('—', maxsplit=1)
    prepared = ' '.join([word2lemma(w.text) for w in razdel.tokenize(r)]).upper()
    return prepared
