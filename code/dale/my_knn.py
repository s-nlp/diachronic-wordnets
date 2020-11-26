import xmltodict
import numpy as np
import pandas as pd

from functools import lru_cache
from collections import Counter, defaultdict
from tqdm.auto import tqdm

from nlp import morph_parse, word2pos, tokenize, morphAnalyzer


def normalize(v, epsilon=1e-10):
    return v / (sum(v**2)**0.5 + epsilon)


class BaseSentenceEmbedder:
    def __init__(self, n=300, normalize_word=True, pos_weights=None, default_weight=1):
        self.n = n
        self.normalize_word = normalize_word
        self.pos_weights = pos_weights
        self.default_weight = default_weight

    def get_word_vec(self, word):
        raise NotImplementedError()

    @lru_cache(maxsize=1024)
    def __call__(self, text):
        tokens = tokenize(text)
        weights = self.get_word_weights(tokens)
        vecs = [self.get_word_vec(word) for word in tokens]
        if self.normalize_word:
            vecs = [normalize(v) for v in vecs]
        if len(vecs) == 0:
            return np.zeros(self.n)
        return normalize(sum([vec * weight for vec, weight in zip(vecs, weights)]))

    def get_word_weights(self, tokens):
        if not self.pos_weights:
            return [1] * len(tokens)
        return [self.pos_weights.get(word2pos(t), self.default_weight) for t in tokens]


class SentenceEmbedder(BaseSentenceEmbedder):
    def __init__(self, ft, **kwargs):
        super(SentenceEmbedder, self).__init__(**kwargs)
        self.ft = ft

    def get_word_vec(self, word):
        return self.ft[word]


class SynsetStorage:
    def __init__(self, id2synset, ids, ids_long, texts_long, word2sense, forbidden_id=None):
        self.id2synset = id2synset
        self.ids = ids
        self.ids_long = ids_long
        self.texts_long = texts_long
        self.word2sense = word2sense
        self.forbidden_id = forbidden_id or set()

    @classmethod
    def construct(cls, synsets_raw, forbidden_words=None):
        id2synset = {v['@id']: v for v in synsets_raw['synsets']['synset']}

        word2sense = defaultdict(set)

        for synset_id, synset in id2synset.items():
            senses = synset['sense']
            if not isinstance(senses, list):
                senses = [senses]
            texts = {sense['#text'] for sense in senses}
            texts.add(synset['@ruthes_name'])
            for text in texts:
                word2sense[text].add(synset_id)
        print('number of texts:', len(word2sense))

        forbidden_id = set()
        for word in (forbidden_words or {}):
            if word in word2sense:
                for sense_id in word2sense[word]:
                    forbidden_id.add(sense_id)
        print('forbidden senses are', len(forbidden_id))

        ids = sorted(id2synset.keys())
        ids_long = []
        texts_long = []
        for id in ids:
            s = id2synset[id]
            senses = s['sense']
            if not isinstance(senses, list):
                senses = [senses]
            texts = {sense['#text'] for sense in senses}
            texts.add(s['@ruthes_name'])

            # исключаем все слова, омонимичные с тем, что есть в тестовой выборке
            senses = {synset_id for w in texts for synset_id in word2sense[w]}
            if senses.intersection(forbidden_id):
                continue

            if len(texts) > 1:
                texts.add(' ; '.join(sorted(texts)))
            for text in sorted(texts):
                ids_long.append(id)
                texts_long.append(text)
        print('numer of ids', len(ids), 'long list is', len(ids_long))
        return cls(
            id2synset=id2synset,
            ids=ids,
            ids_long=ids_long,
            texts_long=texts_long,
            word2sense=word2sense,
            forbidden_id=forbidden_id,
        )

    def get_synset_name(self, synset_id):
        return self.id2synset.get(synset_id, {}).get('@ruthes_name', '-')


def make_rel_df(rel_n_raw, id2synset):
    rel_df = pd.DataFrame(rel_n_raw['relations']['relation'])
    rel_df['parent'] = rel_df['@parent_id'].apply(lambda x: id2synset[x]['@ruthes_name'])
    rel_df['child'] = rel_df['@child_id'].apply(lambda x: id2synset.get(x, {}).get('@ruthes_name'))
    return rel_df


class RelationStorage:
    def __init__(self, forbidden_id=None):
        self.id2hyponym = defaultdict(set)
        self.id2hypernym = defaultdict(set)
        self.forbidden_id = forbidden_id or set()  # forbidden_id = set(ttest.SYNSET_ID)

    def add_pair(self, hypo_id, hyper_id, max_depth=100500):
        if max_depth <= 0:
            return
        if hypo_id in self.id2hyponym[hyper_id]:
            # the pair is already here
            return
        if hypo_id in self.id2hypernym[hyper_id]:
            raise ValueError('{} is already a hypernym of {}, so it cannot become its hyponym'.format(hypo_id, hyper_id))
        for next_hypo in self.id2hyponym[hypo_id]:
            self.add_pair(next_hypo, hyper_id, max_depth=max_depth-1)
        for next_hyper in self.id2hypernym[hyper_id]:
            self.add_pair(hypo_id, next_hyper, max_depth=max_depth-1)
        self.id2hyponym[hyper_id].add(hypo_id)
        self.id2hypernym[hypo_id].add(hyper_id)

    def construct_relations(self, rel_df):
        self.id2hyponym = defaultdict(set)
        self.id2hypernym = defaultdict(set)

        hypo_df = rel_df[rel_df['@name'] == 'hyponym']
        for r, row in tqdm(hypo_df.iterrows()):
            hypo_id = row['@child_id']
            hyper_id = row['@parent_id']
            if hypo_id not in self.forbidden_id and hyper_id not in self.forbidden_id:
                self.add_pair(hypo_id, hyper_id, max_depth=1)  # во второй версии поставим максимальную глубину, равную 2

        print(len(self.id2hyponym))
        print(max(len(c) for c in self.id2hyponym.values()))
        print(max(len(c) for c in self.id2hypernym.values()))
        print(sum(len(c) for c in self.id2hypernym.values()))


def hypotheses_knn(
        text,
        synset_storage: SynsetStorage,
        rel_storage: RelationStorage,
        index=None,
        text2vec=None,
        k=10,
        verbose=False,
        decay=0,
        grand_mult=1,
        result_size=10,
        return_hypotheses=False,
        neighbor_scorer=None,
        indexer=None,
):
    ids_list = synset_storage.ids_long
    texts_list = synset_storage.texts_long
    # todo: distance decay
    if indexer:
        distances, indices = indexer.query(text, k=k)
    else:
        vec = text2vec(text)
        distances, indices = index.query(vec.reshape(1, -1), k=k)
    hypotheses = Counter()
    for i, d in zip(indices.ravel(), distances.ravel()):
        hypers = rel_storage.id2hypernym.get(ids_list[i], set())

        if neighbor_scorer is not None:
            neighbor_score = neighbor_scorer(text, texts_list[i])
        else:
            neighbor_score = 1
        base_score = np.exp(-d ** decay) * neighbor_score
        if verbose:
            print(d, 1, ids_list[i], texts_list[i], len(hypers), np.exp(-d ** decay),  base_score)
        for parent in hypers:
            hypotheses[parent] += base_score
            for grandparent in rel_storage.id2hypernym.get(parent, set()):
                hypotheses[grandparent] += base_score * grand_mult
    if return_hypotheses:
        return hypotheses
    if verbose:
        print(len(hypotheses))
    result = []
    for hypo, cnt in hypotheses.most_common(result_size):
        if verbose:
            print(cnt, hypo, synset_storage.id2synset[hypo]['@ruthes_name'])
        result.append(hypo)
    return result


def prepare_submission(words, hypotheses, id2synset):
    result_nouns = []
    result_hyperonyms = []
    result_hyper_names = []
    for n, h in zip(words, hypotheses):
        for hypo in h:
            result_nouns.append(n)
            result_hyperonyms.append(hypo)
            result_hyper_names.append(id2synset[hypo]['@ruthes_name'])
    result_df = pd.DataFrame({'noun': result_nouns, 'result': result_hyperonyms, 'result_text': result_hyper_names})
    return result_df


def dict2submission(word2hypotheses, id2synset):
    result_nouns = []
    result_hyperonyms = []
    result_hyper_names = []
    for n, h in word2hypotheses.items():
        for hypo in h:
            result_nouns.append(n)
            result_hyperonyms.append(hypo)
            result_hyper_names.append(id2synset[hypo]['@ruthes_name'])
    result_df = pd.DataFrame({'noun': result_nouns, 'result': result_hyperonyms, 'result_text': result_hyper_names})
    return result_df


def prepare_storages(synsets_filename, relations_filename, forbidden_words=None):
    with open(synsets_filename, 'r', encoding='utf-8') as f:
        synsets_raw = xmltodict.parse(f.read(), process_namespaces=True)
    with open(relations_filename, 'r', encoding='utf-8') as f:
        rel_raw = xmltodict.parse(f.read(), process_namespaces=True)

    synset_storage = SynsetStorage.construct(synsets_raw, forbidden_words=forbidden_words or set())

    rel_df = make_rel_df(rel_raw, synset_storage.id2synset)
    rel_storage = RelationStorage(forbidden_id=synset_storage.forbidden_id)
    rel_storage.construct_relations(rel_df)

    return synset_storage, rel_storage, rel_df


def get_transitivity(text):
    tags = [morphAnalyzer.parse(t)[0].tag for t in tokenize(text)]
    trans = None
    for tag in tags:
        if tag.transitivity is not None:
            trans = tag.transitivity
            break
    return trans


def transitivity_rerank(hypos, word, id2synset, mul=1.01):
    trans = get_transitivity(word)
    if trans is None:
        return hypos
    new_hypos = Counter()
    for hypo, score in hypos.items():
        hypo_text = id2synset[hypo]['@ruthes_name']
        ttrans = get_transitivity(hypo_text)
        new_score = score
        if ttrans is None:
            pass
        elif ttrans == trans:
            new_score *= mul
        else:
            new_score /= mul
        new_hypos[hypo] = new_score
    return new_hypos


def get_top(cntr, n=10):
    return [k for k, v in cntr.most_common(n)]


class W2VWrapper(BaseSentenceEmbedder):
    POS_MAP = {
        'INFN': 'VERB',
        'ADJF': 'ADJ',
        'ADVB': 'ADV',
    }
    POS_MISS = {
        'PREP'
    }

    def __init__(self, w2v, morph=morph_parse, add_pos=True, **kwargs):
        super(W2VWrapper, self).__init__(**kwargs)
        self.w2v = w2v
        self.morph = morph
        self.prefix2word = self.make_prefixes()
        self.add_pos = add_pos

    def make_prefixes(self):
        prefix2word = defaultdict(set)
        for w in self.w2v.vocab.keys():
            for n in range(1, len(w) - 2):
                prefix2word[w[:-n]].add(w)
        prefix2word = {k: v for k, v in prefix2word.items()}
        return prefix2word

    def find_prefix(self, word, min_len=2):
        mapping = self.prefix2word
        if word in mapping:
            return mapping[word]
        for i in range(1, len(word) - min_len):
            t = mapping.get(word[:-i])
            if t is not None:
                return t
        return None

    def get_text_vec(self, text, verbose=False):
        toks = tokenize(text)
        vecs = []
        for tok in toks:
            vecs.append(self.get_word_vec(word=tok, verbose=verbose))
        if not vecs:
            return np.zeros(self.w2v.vectors.shape[1])
        return normalize(sum(vecs))
    
    def get_word_vec(self, word, verbose=False):
        parse = self.morph(word)
        if not parse:
            return np.zeros(self.n)
        tag = parse.tag
        if tag is None:
            print('none POS for word "{}"'.format(word))
            tag = '-'
        new_pos = self.POS_MAP.get(tag.POS, tag.POS or '-')
        if new_pos in self.POS_MISS:
            return np.zeros(self.n)
        key = word
        key2 = parse.normal_form or word
        if self.add_pos:
            key = key + '_' + new_pos
            key2 = key2 + '_' + new_pos 
        if verbose:
            print(key, key2, self.find_prefix(key))
        if key in self.w2v.vocab:
            return self.w2v[key]
        elif key2 in self.w2v.vocab:
            return self.w2v[key2]
        else:
            keys = self.find_prefix(key)
            if keys:
                return sum([self.w2v[k] for k in keys]) / len(keys)
        return np.zeros(self.n)


class MixedIndexer:
    def __init__(self, models, vectorizers, proportions, coefs):
        self.models = models
        self.vectorizers = vectorizers
        self.proportions = proportions
        self.coefs = coefs

    def query(self, text, k=10):
        all_distances, all_indices = [], []
        for m, v, p, c in zip(self.models, self.vectorizers, self.proportions, self.coefs):
            dis, ind = m.query(v(text).reshape(1, -1), k=max(1, int(k * p)))
            all_distances.append(dis * c)
            all_indices.append(ind)
        # return all_distances, all_indices
        return np.concatenate(all_distances, axis=1), np.concatenate(all_indices, axis=1)
