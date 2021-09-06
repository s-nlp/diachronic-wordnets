import pandas as pd
import codecs
import json
from collections import defaultdict, OrderedDict
import numpy as np
from tqdm import tqdm
import gensim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import shutil
import os
import sys
import random
# todo in separate file

def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0].upper()
            hypernyms = read_fn(line_split[1])
            vocab[word].append(hypernyms)
    return vocab


class VectorsWithHash:
    def __init__(self, vectors):
        self.vectors = vectors
        self.hash = {}

    def __contains__(self, w):
        return w in self.vectors

    def similarity(self, w1, w2):
        key = f'{w1}[SEP]{w2}'
        if key in self.hash:
            return self.hash[key]

        res = self.vectors.similarity(w1, w2)
        self.hash[key] = res
        return res

    def most_similar(self, word, topn=10000):
        return self.vectors.most_similar(word, topn=topn)

class HypernymPredictModel():
    def __init__(self, config):
        pass

    def train(self, train_words):
        pass

    def predict(self, new_words):
        pass

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass

    def _init_model_state(self, config):
        self.word_type = config['pos']
        self.topk = config['topk']
        pass

    def _load_resources(self, config):
        self._load_thesaurus(config)
        self._load_vectors(config)
        self._load_wkt(config)

    def _load_thesaurus(self, config):
        ThesClass = EnWordNet if config['lang'] == 'en' else RuThes if config['ruthes'] else RuWordNet
        self.thesaurus = ThesClass(config['thesaurus_dir'])
        
    def _load_vectors(self, config):
        embeddings_path = config['embeddings_path']
        try:
            self.vector_model = VectorsWithHash(gensim.models.KeyedVectors.load(embeddings_path))
        except:
            self.vector_model = VectorsWithHash(gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False))

    def _load_wkt(self, config):
        self.wiktionary = wkt.load_wiktionary(config['wiktionary_dump_path'], self.vector_model)

    def _calculate_features(self, word):
        candidate2features = self._calculate_candidates(word)

        candidate_col = []
        features = []
        for synset_id in candidate2features:
            synset = self.thesaurus.synsets[synset_id]

            init_features = self._calculate_init_features(synset_id, candidate2features)
            wkt_features = calculate_wiktionary_features(word, synset, self.wiktionary)
            synset_features = calculate_synset_similarity(word, synset, self.vector_model, None)

            candidate_col.append(synset_id)
            features.append(init_features + wkt_features + synset_features)

        features = np.array(features)
        columns = {f'f{i}': features[:,i] for i in range(features.shape[1])}
        columns['cand'] = candidate_col

        return pd.DataFrame(columns)

    def _calculate_candidates(self, word):
        most_similar_words = self.vector_model.most_similar(word, topn=10000) # must be larger then topk
        most_similar_words = filter_most_sim_by_type(word, most_similar_words, self.word_type, self.thesaurus)
        most_similar_words = most_similar_words[:self.topk]

        candidates = []
        for cand_word in most_similar_words:
            if cand_word in self.thesaurus.sense2synid:
                for synid in self.thesaurus.sense2synid[cand_word]:
                    if self.thesaurus.synsets[synid].synset_type != self.word_type:
                        continue
                    
                    candidates.append([synid, 0])
                    for h in self.thesaurus.synsets[synid].rels.get('hypernym', []):
                        candidates.append([h.synset_id, 1])
                        for hh in h.rels.get('hypernym', []):
                            candidates.append([hh.synset_id, 2])
                            
        
        candidate2features = {}
        for synset_id, level in candidates:
            if synset_id not in candidate2features:
                candidate2features[synset_id] = [0, []]
            candidate2features[synset_id][0] += 1
            candidate2features[synset_id][1].append(level)

        return candidate2features


    def _calculate_init_features(synset_id, candidate2features):
        features = []
        init_features = candidate2features[synset_id]

        features.append(init_features[0])
        features.append(np.log2(2 + init_features[0]))

        features.append(np.min(init_features[1]))
        features.append(np.mean(init_features[1]))
        features.append(np.max(init_features[1]))

        return features



'''
def load_wiktionary(wiktionary_dump_path, vectors, wkt):
    title2docs = {key.replace(' ', '_'): val for key, val in wkt.get_title2docs(wiktionary_dump_path).items() if key in vectors}
    for title in title2docs:
        docs_info = []
        for doc in title2docs[title]:
            docs_info.append(wkt.parse_wiktionary(doc['text']))
        title2docs[title] = docs_info
    return title2docs
'''
def load_wiktionary(wiktionary_dump_path, vectors):
    title2docs = {key.replace(' ', '_'): val for key, val in wkt.get_title2docs(wiktionary_dump_path).items() if key in vectors}
    for title in title2docs:
        docs_info = []
        for doc in title2docs[title]:
            docs_info.append(wkt.parse_wiktionary(doc['text']))
        title2docs[title] = docs_info
    return title2docs

def calc_most_sim(embeddings, thesaurus, test_df, word_type, topk):
    all_most_similar = []
    words = []
    for word in test_df['word']:
        if word not in embeddings:
            continue
        most_similar_words = embeddings.most_similar(word, topn=10000) # must be larger then topk
        most_similar_words = filter_most_sim_by_type(word, most_similar_words, word_type, thesaurus)
        most_similar_words = most_similar_words[:topk]

        all_most_similar.append(most_similar_words)
        words.append(word)

    df = pd.DataFrame(columns=['word', 'most_similar'])
    df['word'] = words
    df['most_similar'] = all_most_similar
    return df

def filter_most_sim_by_type(word, most_similar_words, word_type, thesaurus):
    filtered_word_list = []
    banned_words = set()
    for synid in thesaurus.sense2synid.get(word, []):
        synset = thesaurus.synsets[synid]
        banned_words.update(synset.synset_words)

    for w, score in most_similar_words:
        w = w.replace('ё', 'е')
        if w not in thesaurus.sense2synid:
            continue

        if w in banned_words:
            continue

        found_sense = False
        inst_hypernyms = []
        for synid in thesaurus.sense2synid[w]:
            if thesaurus.synsets[synid].synset_type == word_type:
                found_sense = True
            if 'instance hypernym' in thesaurus.synsets[synid].rels:
                inst_hypernyms.append(1)
            else:
                inst_hypernyms.append(0)

        if found_sense is True and sum(inst_hypernyms) != len(inst_hypernyms):
            filtered_word_list.append([w, score])

    return filtered_word_list

def load_word2patterns(word2patterns_path):
    word2patterns = {}
    with codecs.open(word2patterns_path, 'r', 'utf-8') as file_descr:
        for line in file_descr.read().split('\n'):
            if len(line) == 0:
                continue
            target_word, cand_word, pattern_count, one_sent_count = line.split('\t')
            if target_word not in word2patterns:
                word2patterns[target_word] = {}
            word2patterns[target_word][cand_word] = [int(pattern_count), int(one_sent_count)]
    return word2patterns


def get_word_patterns_features(target_word, cand_word, word2patterns_syn):
    word_pattern_features = []
    if len(word2patterns_syn) > 0:
        syn_pattern_count = 0
        syn_one_sent_count = 0
        syn_pattern_score = 1
        if target_word in word2patterns_syn and w in word2patterns_syn[target_word]:
            syn_pattern_count = word2patterns_syn[target_word][cand_word][0]
            syn_one_sent_count = word2patterns_syn[target_word][cand_word][1]
            syn_pattern_score = 1 + syn_pattern_count / (syn_one_sent_count + 2)
        word_pattern_features = [np.log2(1 + syn_pattern_count), np.log2(1 + syn_one_sent_count), syn_pattern_score]
        
    return word_pattern_features

def get_synset_patterns_features(target_word, synset, word2patterns_hyp):
    synset_pattern_features = []
    if len(word2patterns_hyp) > 0:
        hyp_pattern_features = []
        for s_word in set(synset.synset_words):
            if target_word in word2patterns_hyp and s_word in word2patterns_hyp[target_word]:
                hyp_pattern_count = word2patterns_hyp[target_word][s_word][0]
                hyp_one_sent_count = word2patterns_hyp[target_word][s_word][1]
                hyp_pattern_score = 1 + hyp_pattern_count / (hyp_one_sent_count + 2)
                hyp_pattern_features.append([np.log2(1 + hyp_pattern_count), np.log2(1 + hyp_one_sent_count), hyp_pattern_score])
        if len(hyp_pattern_features) == 0:
            hyp_pattern_features.append([0, 0, 0])
        max_hyp_pattern_features = np.max(hyp_pattern_features, axis=0)
        min_hyp_pattern_features = np.min(hyp_pattern_features, axis=0)
        synset_pattern_features += max_hyp_pattern_features.tolist()
        synset_pattern_features += min_hyp_pattern_features.tolist()
        
    return synset_pattern_features

'''         
def get_synset_bert_features(target_word, synset, bert_model, thesaurus):
    synset_bert_features = []
    if bert_model is not None:
        synsets_ids = [synset.synset_id]
        for hypo in synset.rels.get('hyponym', []):
            synsets_ids.append(hypo.synset_id)
        synsets_ids = sorted(synsets_ids)
        bert_probs = get_synsets_probs(bert_model, target_word, synsets_ids, thesaurus)
        synset_bert_prob = bert_probs[0][1]
        if bert_probs.shape[0] > 1:
            hyponyms_bert_prob =  np.max(bert_probs[1:], axis=0)[1]
        else:
            hyponyms_bert_prob = 0.0
        synset_bert_features += [synset_bert_prob, hyponyms_bert_prob]
        
    return synset_bert_features
'''
def get_synset_bert_features(target_word, synset, bert_model, thesaurus):
    synset_bert_features = []
    if bert_model is not None:
        synset_bert_features.append(bert_model.predict(target_word, ';'.join(synset.synset_words), return_prob=True)[0][1])
    
    return synset_bert_features

def get_synset_candidates_(cand_word, thesaurus, word_type):
    candidates = []
    if cand_word in thesaurus.sense2synid:
        for synid in thesaurus.sense2synid[cand_word]:
            if thesaurus.synsets[synid].synset_type != word_type:
                continue
            #if 'instance hypernym' in thesaurus.synsets[synid].rels:
            #    continue
            
            for h in thesaurus.synsets[synid].rels.get('hypernym', []):
                candidates.append([h, 1])
                for hh in h.rels.get('hypernym', []):
                    candidates.append([hh, 2])
                    
    return candidates

def get_init_synset_features(synset, level):
    synset_features = [-level]
    return synset_features
            

def get_candidates_with_features(target_word, cand_word, word_features, thesaurus, vectors, word_type, word2patterns_syn={}, word2patterns_hyp={}, bert_model=None, wiktionary={}):
    predict = OrderedDict()
    
    word_features += get_word_patterns_features(target_word, cand_word, word2patterns_syn)
    candidates = get_synset_candidates_(cand_word, thesaurus, word_type)

    for s, level in candidates:
        if s.synset_type != word_type:
            continue
        synset_features = get_init_synset_features(s, level)
        synset_features += get_synset_patterns_features(target_word, s, word2patterns_hyp)
        synset_features += get_synset_bert_features(target_word, s, bert_model, thesaurus)
        synset_features += calculate_wiktionary_features(target_word, s, wiktionary)
        synset_features += calculate_synset_similarity(target_word, s, vectors, thesaurus)
        
        if s.synset_id not in predict:
            predict[s.synset_id] = [s.synset_name, np.array(word_features), np.array(synset_features)]
        else:
            predict[s.synset_id][1] = np.vstack([predict[s.synset_id][1], np.array(word_features)])
            predict[s.synset_id][2] = np.vstack([predict[s.synset_id][2], np.array(synset_features)])

    return predict


def get_synset_candidates(most_sim, thesaurus, vectors, word_type, predict_topk=20, word2patterns_syn={}, word2patterns_hyp={}, bert_model=None, wiktionary={}):
    synset_candidates = OrderedDict()
    for _, row in tqdm(most_sim.iterrows()):
        target_word = row['word']
        most_similar_words = row['most_similar'][:predict_topk]
        predict = OrderedDict()
        for data in most_similar_words:
            w = data[0]
            word_features = data[1:]

            candidates_features = get_candidates_with_features(target_word, w, word_features, thesaurus, vectors, word_type,
                                                               word2patterns_syn, word2patterns_hyp, bert_model, wiktionary)
            for synset_id in candidates_features:
                if synset_id not in predict:
                    predict[synset_id] = candidates_features[synset_id]
                else:
                    predict[synset_id][1] = np.vstack([predict[synset_id][1], candidates_features[synset_id][1]])
                    predict[synset_id][2] = np.vstack([predict[synset_id][2], candidates_features[synset_id][2]])
        synset_candidates[target_word] = predict

    return synset_candidates

def get_train_true_info(df):
    reference = {}
    w2true = {}
    for _, row in df.iterrows():
        word = row['word']
        target_gold = json.loads(row['target_gold'])
        reference[word] = target_gold
        w2true[word] = set()
        for t in target_gold:
            w2true[word].update(t)

    return reference, w2true

def get_features_df(synset_candidates, w2true=None):
    all_features = []
    for w in synset_candidates:
        for cand in synset_candidates[w]:

            word_features = synset_candidates[w][cand][1]
            synset_features = synset_candidates[w][cand][2]
            if len(word_features.shape) == 1:
                word_features.resize((1, word_features.shape[0]))

            if len(synset_features.shape) == 1:
                synset_features.resize((1, synset_features.shape[0]))

            count = word_features.shape[0]
            log2count = np.log2(2 + count)
            mean_synset_features = np.mean(synset_candidates[w][cand][2], axis=0).tolist()
            max_synset_features = np.max(synset_candidates[w][cand][2], axis=0).tolist()
            max_word_features = np.max(word_features, axis=0).tolist()
            mean_word_features = np.mean(word_features, axis=0).tolist()
            min_word_features = np.min(word_features, axis=0).tolist()
            all_features.append([w, cand, int(w in w2true and cand in w2true[w]), count, log2count] + mean_synset_features + max_word_features  + mean_word_features + min_word_features)

    df = pd.DataFrame(all_features, columns=['word', 'cand', 'label'] + [f'f{i}' for i in range(len(all_features[0]) - 3)])
    df.fillna(0.0, inplace=True) 
    return df

def train_predict_cv(df_features, ref, folds=5):
    features_len = len(df_features.columns) - 3
    features = [f'f{i}' for i in range(features_len)]
    kf = KFold(n_splits=folds)

    results = []
    models = []
    for train_index, test_index in kf.split(df_features['word'].unique()):
        train_words = df_features['word'].unique()[train_index]
        test_words = df_features['word'].unique()[test_index]

        train_df = df_features[df_features['word'].apply(lambda x: x in train_words)]
        test_df = df_features[df_features['word'].apply(lambda x: x in test_words)]

        clf = LogRegScaler()

        X_train = train_df[features]
        X_test = test_df[features]
        y_train = train_df['label']
        y_test = test_df['label']

        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)

        test_df['predict'] = y_pred[:,1]
        test_df = test_df.sort_values(by=['word', 'predict'], ascending=False)

        cur_ref = {w: ref[w] for w in ref if w in set(test_words)}
        mean_ap, mean_rr, w2ap = get_score(cur_ref, from_df_to_pred(test_df), k=10)
        eval_res = [mean_ap, mean_rr]

        models.append(clf)

        print(eval_res)
        results.append(eval_res)

    print(f'Averaged results = {np.mean(results, axis=0)}')
    return models, features

def get_score_list(reference, submitted, k=10):
    random.seed(42)
    max_items_len = int(len(reference) * 0.8)
    all_words = list(reference)
    map_list, mrr_list = [], []

    for _ in range(30):
        random.shuffle(all_words)
        _80_percent_words = all_words[:max_items_len]
        smaller_reference = dict(filter(lambda x: x[0] in _80_percent_words, reference.items()))
        mean_ap, mean_rr = get_score(smaller_reference, submitted, k=k)
        map_list.append(mean_ap)
        mrr_list.append(mean_rr)

    #write_to_file(os.path.splitext(submitted_path)[0]+"_map_scores.json", map_list)
    #write_to_file(os.path.splitext(submitted_path)[0]+"_mrr_scores.json", mrr_list)

    return map_list, mrr_list

def get_score(reference, predicted, k=10, return_addition_info=False):
    ap_sum = 0
    rr_sum = 0

    w2res = {}
    for neologism in reference:
        reference_hypernyms = reference.get(neologism, [])
        predicted_hypernyms = predicted.get(neologism, [])
        ap = compute_ap(reference_hypernyms, predicted_hypernyms, k)
        rr = compute_rr([j for i in reference_hypernyms for j in i], predicted_hypernyms, k)
        ap_sum += ap
        rr_sum += rr
        w2res[neologism] = ap

    results = [ap_sum / len(reference), rr_sum / len(reference)]
    if return_addition_info:
        results.append(w2res)
    return results


def compute_ap(actual, predicted, k=10):
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted = set()
    skipped = 0
    for i, p in enumerate(predicted):
        if p in already_predicted:
            skipped += 1
            continue
        for parents in actual:
            if p in parents:
                num_hits += 1.0
                score += num_hits / (i + 1.0 - skipped)
                already_predicted.update(parents)
                break

    return score / min(len(actual), k)


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0


def from_df_to_pred(df):
    pred = {}
    for i, row in df.iterrows():
        word = row['word']
        if word not in pred:
            pred[word] = []
        cand = row['cand']
        pred[word].append(cand)
    return pred

def calculate_wiktionary_features(target_word, synset, wiktionary):
    # 1 feature for direct syn, 1 feature for hypo syn, 1 feature for direct hyper, 1 feature for hypo hyper
    if len(wiktionary) == 0:
        return []
    
    features = [0] * 6
    direct_syn_feature_idx = 0
    hypo_syn_feature_idx = 1
    direct_hyper_feature_idx = 2
    hypo_hyper_feature_idx = 3
    direct_meaning_feature_idx = 4
    hypo_meaning_feature_idx = 5

    def get_all_wkt(word, wiktionary, tag):
        all_words = set([word])
        for wikt_doc_info in wiktionary.get(word, []):
            all_words.update(wikt_doc_info[tag])
        return all_words

    tw_synonyms = get_all_wkt(target_word, wiktionary, 'synonym')
    tw_hypernyms = get_all_wkt(target_word, wiktionary, 'hypernym')
    tw_meaning = []
    for wikt_doc_info in wiktionary.get(target_word, []):
        tw_meaning.append('_'.join(wikt_doc_info['meaning']).replace(' ', '_'))
    tw_meaning = '_'.join(tw_meaning)

    synset_synonyms = set()
    for word in synset.synset_words:
        synset_synonyms.update(get_all_wkt(word, wiktionary, 'synonym'))

    hypo_synonyms = set()
    hypo_words = set()
    for hypo in synset.rels.get('hyponym', []):
        hypo_words.update(hypo.synset_words)
        for word in hypo.synset_words:
            hypo_synonyms.update(get_all_wkt(word, wiktionary, 'synonym'))

    features[direct_syn_feature_idx] = int(len(tw_synonyms.intersection(synset_synonyms)) > 0)
    features[hypo_syn_feature_idx] = int(len(tw_synonyms.intersection(hypo_synonyms)) > 0)
    features[direct_hyper_feature_idx] = int(len(tw_hypernyms.intersection(set(synset.synset_words))) > 0)
    features[hypo_hyper_feature_idx] = int(len(tw_hypernyms.intersection(hypo_words)) > 0)

    for w in synset_synonyms:
        if w in tw_meaning:
            features[direct_meaning_feature_idx] = 1

    for w in hypo_synonyms:
        if w in tw_meaning:
            features[hypo_meaning_feature_idx] = 1

    return features

def calculate_synset_similarity(w, synset, vectors, thesaurus):
    sim_list = []
    hyp_sim_list = []
    f_lists = [[], [], [], []]
    for synset_word in synset.synset_words:
        if synset_word in vectors:
            f_lists[0].append(vectors.similarity(w, synset_word))
    for hyponym in synset.rels.get('hyponym', []):
        hyponym_sim = []
        for synset_word in hyponym.synset_words:
            if synset_word in vectors:
                hyponym_sim.append(vectors.similarity(w, synset_word))
        if len(hyponym_sim) == 0:
            hyponym_sim.append(0)
        f_lists[1].append(np.max(hyponym_sim))
        f_lists[2].append(np.mean(hyponym_sim))
        f_lists[3].append(np.min(hyponym_sim))
    results = []
    for f_list in f_lists:
        if len(f_list) == 0:
            f_list.append(0)
        results.append(np.max(f_list))
        results.append(np.mean(f_list))
        results.append(np.min(f_list))
    return results

class LogRegScaler():
    def __init__(self):
        self.clf = LogisticRegression(solver='liblinear', max_iter=1000, C=0.2, class_weight='balanced')
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        y_pred = self.clf.predict_proba(X)
        return y_pred

def predict_test(test_df, thesaurus, vector_model, models, features, pos, save_path,
                 word2patterns_syn={}, word2patterns_hyp={}, bert_model=None, reference_path=None, predict_synset_cand_count=20, lang='ru', wiktionary={}):
    #print('Calculating test most sim')
    most_sim = calc_most_sim(vector_model, thesaurus, test_df, pos, 100)

    #print('Calculating test candidates with features')
    synset_candidates = get_synset_candidates(most_sim, thesaurus, vector_model, pos, predict_synset_cand_count,
                                              word2patterns_syn, word2patterns_hyp, bert_model, wiktionary)
    test_df_features = get_features_df(synset_candidates, w2true={})
    print(test_df_features.iloc[:2])
    #print('Predict')
    test_df_features[f'predict'] = [0] * test_df_features.shape[0]
    for model in models:
        test_df_features[f'predict'] += model.predict_proba(test_df_features[features])[:,1]
    test_df_features[f'predict'] /= len(models)

    test_df_features = test_df_features.sort_values(by=['word', 'predict'], ascending=False)
    test_df_features['word'] = test_df_features['word'].apply(lambda x: x.upper())
    if lang == 'en':
        test_df_features['cand'] = test_df_features['cand'].apply(lambda x: thesaurus.synsets[x].synset_name)
    test_df_features.to_csv('temp.csv', sep='\t', index=False, header=False)
    if save_path is not None:
        create_submit_df(test_df_features).to_csv(save_path, sep='\t', index=False, header=False)

    if reference_path is not None:
        submitted = read_dataset(save_path)
        reference = read_dataset(reference_path, json.loads)
        map_score, mrr_score, _ = get_score(reference, submitted)
        print(f'Results for {save_path}:\n\tMAP = {map_score}\n\tMRR = {mrr_score}')

    return test_df_features

def create_submit_df(df, topk=10):
    target_words = []
    predict = []
    probas = []
    counter = 0
    last_w = ''
    for i, row in df.iterrows():
        if row['word'] != last_w:
            counter = 0
            last_w = row['word']
        if counter < 10:
            target_words.append(row['word'].upper())
            predict.append(row['cand'])
            probas.append(row['predict'])
        counter += 1
    return pd.DataFrame({'word': target_words, 'cand': predict, 'prob': probas})
#
