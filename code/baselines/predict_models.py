import json
import re
from abc import abstractmethod, ABC
from collections import Counter
from operator import itemgetter

import numpy as np
from gensim.models import KeyedVectors
from scipy import spatial


class Model(ABC):
    def __init__(self, params):
        self.w2v_synsets = KeyedVectors.load_word2vec_format(params['synsets_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    def predict_hypernyms(self, neologisms, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        return {neologism: self.compute_candidates(neologism, get_hypernym_fn,
                                                   get_hyponym_fn, get_taxonomy_name_fn,
                                                   topn) for neologism in neologisms}

    def get_score(self, neologism, candidate, count):
        return count * self.get_similarity(neologism, candidate)

    def get_similarity(self, neologism, candidate):
        v1 = self.w2v_data[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    @abstractmethod
    def compute_candidates(self, neologisms, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        pass


# ---------------------------------------------------------------------------------------------
# Baseline Model
# ---------------------------------------------------------------------------------------------

class BaselineModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10) -> list:
        return list(map(itemgetter(0), self.generate_associates(neologism, topn)))

    def generate_associates(self, neologism, topn=10) -> list:
        return self.w2v_synsets.similar_by_vector(self.w2v_data[neologism], topn)


# ---------------------------------------------------------------------------------------------
# Hypernym of Co-Hypernyms Model
# ---------------------------------------------------------------------------------------------

class HCHModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        return self.compute_hchs(neologism, get_hypernym_fn, topn=10)[:topn]

    def compute_hchs(self, neologism, compute_hypernyms, topn=10) -> list:
        associates = map(itemgetter(0), self.generate_associates(neologism, topn))
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs


# ---------------------------------------------------------------------------------------------
# Ranked Model
# ---------------------------------------------------------------------------------------------

class RankedModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]

        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        sorted_hypernyms = reversed(sorted(all_hypernyms.items(), key=lambda x: self.get_score(neologism, *x)))

        return [i[0] for i in sorted_hypernyms][:topn]


# ---------------------------------------------------------------------------------------------
# LR Model
# ---------------------------------------------------------------------------------------------

def distance2vote(d, a=3.0, b=5.0, y=1.0):
    sim = np.maximum(0, 1 - d ** 2 / 2)
    return np.exp(-d ** a) * y * sim ** b


def compute_distance(s):
    return np.sqrt(2 * (1 - s))


class LRModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)
        self.wiktionary = self.__get_wiktionary(params['wiki_path'])
        self.wiki_model = KeyedVectors.load_word2vec_format(params['wiki_vectors_path'], binary=False)

        self.delete_bracets = re.compile(r"\(.+?\)")
        if params['language'] == 'ru':
            self.pattern = re.compile("[^А-я \-]")
        else:
            self.pattern = re.compile("[^A-z \-]")

    def __get_wiktionary(self, path):
        wiktionary = {}
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                wiktionary[data['word']] = {"hypernyms": data['hypernyms'], "synonyms": data['synonyms'],
                                            "meanings": data['meanings']}
        return wiktionary

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]
        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        associates = self.generate_associates(neologism, 100)
        votes = Counter()
        for associate, similarity in associates:
            distance = compute_distance(similarity)
            for hypernym in get_hypernym_fn(associate):
                votes[hypernym] += distance2vote(distance)
                for second_order in get_hypernym_fn(hypernym):
                    votes[second_order] += distance2vote(distance, y=0.5)

        sorted_hypernyms = Counter()
        for candidate in all_hypernyms + votes:
            count = all_hypernyms.get(candidate, 1)
            similarity, wiki_similarity, in_synonyms, \
            in_hypernyms, in_definition, not_in_hypernyms, \
            not_in_synonyms, not_in_definition, not_wiki_similarity = self.compute_weights(neologism, candidate,
                                                                                           get_taxonomy_name_fn)

            hyponym_count = votes.get(candidate, 0.0)
            if hyponym_count == 0.0:
                not_hyponym_count = 1.0
            else:
                not_hyponym_count = 0.0

            score = count * similarity * 3.34506023 + wiki_similarity * 6.82626317 + not_wiki_similarity * 1.87866841 + \
                    in_synonyms * 2.01482447 + not_in_synonyms * -0.36626615 + in_hypernyms * 1.42307518 + \
                    not_in_hypernyms * 0.22548314 + in_definition * 2.06956417 + -0.42100585 * not_in_definition + \
                    hyponym_count * 13.73838828 + not_hyponym_count * 0.0


            sorted_hypernyms[candidate] = score

        return [i[0] for i in sorted_hypernyms.most_common(topn)]

    def compute_weights(self, neologism, candidate, get_taxonomy_name_fn):
        similarity = self.get_similarity(neologism, candidate)
        wiki_similarity = 0.0
        not_wiki_similarity = 0.0
        in_synonyms = 0.0
        in_hypernyms = 0.0
        in_definition = 0.0
        not_in_synonyms = 0.0
        not_in_hypernyms = 0.0
        not_in_definition = 0.0

        candidate_words = self.delete_bracets.sub("", get_taxonomy_name_fn(candidate)).split(',')
        if neologism.lower() in self.wiktionary:
            wiktionary_data = self.wiktionary[neologism.lower()]

            if any([candidate_word.lower() in wiktionary_data['hypernyms'] for candidate_word in candidate_words]):
                in_hypernyms = 1.0
            else:
                not_in_hypernyms = 1.0

            if any([candidate_word.lower() in wiktionary_data['synonyms'] for candidate_word in candidate_words]):
                in_synonyms = 1.0
            else:
                not_in_synonyms = 1.0

            if any([any([candidate_word.lower() in i for candidate_word in candidate_words])
                    for i in wiktionary_data['meanings']]):
                in_definition = 1.0
            else:
                not_in_definition = 1.0

            wiki_similarities = []
            for wiki_hypernym in wiktionary_data['hypernyms']:
                wiki_hypernym = wiki_hypernym.replace("|", " ").replace('--', '')
                wiki_hypernym = self.pattern.sub("", wiki_hypernym)
                if not all([i == " " for i in wiki_hypernym]):
                    wiki_similarities.append(self.compute_similarity(wiki_hypernym.replace(" ", "_"), candidate))
            if wiki_similarities:
                wiki_similarity = sum(wiki_similarities) / len(wiki_similarities)
            else:
                not_wiki_similarity = 1.0
        else:
            not_wiki_similarity = 1.0

        return similarity, wiki_similarity, in_synonyms, in_hypernyms, in_definition, not_in_hypernyms, \
               not_in_synonyms, not_in_definition, not_wiki_similarity

    def compute_similarity(self, neologism, candidate):
        v1 = self.wiki_model[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)
