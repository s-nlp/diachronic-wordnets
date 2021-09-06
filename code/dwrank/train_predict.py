import pandas as pd
import codecs
import json
import gensim
import os
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from collections import OrderedDict

from taxoenrich.models import EnWordNet, RuWordNet
from taxoenrich.utils import get_score, VectorsWithHash, LogRegScaler, get_score_list

import argparse

from collections import defaultdict, OrderedDict
import json

def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0].upper()
            hypernyms = read_fn(line_split[1])
            vocab[word].append(hypernyms)
    return vocab

class HypernymPredictModel():
    def __init__(self, config):
        self.config = config
        self._init_model_state(config)
        self._load_resources(config)

    def train(self, train_df):
        self.thesaurus.filter_thesaurus(train_df['word'])

        features = [self._calculate_features(w) for w in tqdm(train_df['word'])]
        features = [f for f in features if f is not None]

        train_features_df = pd.concat(features).reset_index(drop=True)

        ref, word2true = self._get_train_true_info(train_df)
        self._add_true(train_features_df, word2true)

        self.models, self.features = self._train_predict_cv(train_features_df, ref, 3)

    def predict(self, new_words):
        features = [self._calculate_features(w) for w in tqdm(new_words)]
        features = [f for f in features if f is not None]

        test_features_df = pd.concat(features).reset_index(drop=True)

        test_features_df[f'predict'] = [0] * test_features_df.shape[0]
        for model in self.models:
            test_features_df[f'predict'] += model.predict_proba(test_features_df[self.features])[:,1]
        test_features_df[f'predict'] /= len(self.models)

        test_features_df = test_features_df.sort_values(by=['word', 'predict'], ascending=False)
        test_features_df['word'] = test_features_df['word'].apply(lambda x: x.upper())
        if self.lang == 'en':
            test_features_df['cand'] = test_features_df['cand'].apply(lambda x: self.thesaurus.synsets[x].synset_name)

        return self._create_predict_df(test_features_df)

    def save(self, model_path):
        joblib.dump({'models': self.models, 'features': self.features}, model_path)

    def load(self, model_path):
        model_info = joblib.load(model_path)
        self.models = model_info['models']
        self.features = model_info['features']

    def _init_model_state(self, config):
        self.word_type = config['pos']
        self.topk = config['topk']
        self.wkt_on = config['wkt']
        self.lang = config['lang']
        self.search_by_word = config['search_by_word']

        global wkt
        if self.lang == 'en':
            import wiktionary_processing.utils_en as wkt
        else:
            import wiktionary_processing.utils as wkt

    def _load_resources(self, config):
        self._load_thesaurus(config)
        self._load_vectors(config)
        self._load_wkt(config)

    def _load_thesaurus(self, config):
        print('Loading Thesaurus')
        ThesClass = EnWordNet if config['lang'] == 'en' else RuThes if config['ruthes'] else RuWordNet
        self.thesaurus = ThesClass(config['thesaurus_dir'])
        
    def _load_vectors(self, config):
        print('Loading Vectors')
        embeddings_path = config['embeddings_path']
        try:
            self.vector_model = VectorsWithHash(gensim.models.KeyedVectors.load(embeddings_path))
        except:
            self.vector_model = VectorsWithHash(gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False))

    def _load_wkt(self, config):
        self.wiktionary = {}
        if self.wkt_on:
            print('Loading Wiktionary')
            self.wiktionary = wkt.load_wiktionary(config['wiktionary_dump_path'], self.vector_model)

    def _calculate_features(self, word):
        candidate2features = self._calculate_candidates(word)
        if len(candidate2features) == 0:
            return None
        candidate_col = []
        features = []
        for synset_id in candidate2features:
            if synset_id not in self.thesaurus.synsets:
                continue
            synset = self.thesaurus.synsets[synset_id]

            init_features = self._calculate_init_features(synset_id, candidate2features)
            wkt_features = self._calculate_wiktionary_features(word, synset)
            synset_features = self._calculate_synset_similarity(word, synset)

            candidate_col.append(synset_id)
            features.append(init_features + wkt_features + synset_features)

        features = np.array(features)

        columns = OrderedDict()
        columns['word'] = [word] * len(candidate_col)
        columns['cand'] = candidate_col

        for i in range(features.shape[1]):
            columns[f'f{i}'] = features[:,i]

        return pd.DataFrame(columns)

    def _calculate_candidates(self, word):
        if word not in self.vector_model:
            return {}

        most_similar_words = self.vector_model.most_similar(word, topn=10000) # must be larger then topk
        
        if self.search_by_word:
            most_similar_words = self._filter_most_sim_by_type(word, most_similar_words)
            most_similar_words = most_similar_words[:self.topk]

            candidates = []
            for cand_word, similarity in most_similar_words:
                if cand_word in self.thesaurus.sense2synid:
                    for synid in self.thesaurus.sense2synid[cand_word]:
                        if self.thesaurus.synsets[synid].synset_type != self.word_type:
                            continue
                        
                        candidates.append([synid, 0, similarity])
                        for h in self.thesaurus.synsets[synid].rels.get('hypernym', []):
                            candidates.append([h.synset_id, 1, similarity])
                            for hh in h.rels.get('hypernym', []):
                                candidates.append([hh.synset_id, 2, similarity])
        else:
            most_similar_words = self._filter_most_sim(word, most_similar_words)
            most_similar_words = most_similar_words[:self.topk]
            candidates = []
            for synid, _ in most_similar_words:
                if self.thesaurus.synsets[synid].synset_type != self.word_type:
                    continue
                
                candidates.append([synid, 0])
                for h in self.thesaurus.synsets[synid].rels.get('hypernym', []):
                    candidates.append([h.synset_id, 1])
                    for hh in h.rels.get('hypernym', []):
                        candidates.append([hh.synset_id, 2])
                            
        
        candidate2features = {}
        for cand_info in candidates:
            synset_id = cand_info[0]
            features = cand_info[1:]
            if synset_id not in candidate2features:
                candidate2features[synset_id] = [0]
                for f in features:
                    candidate2features[synset_id].append([])

            candidate2features[synset_id][0] += 1
            for i, f in enumerate(features):
                candidate2features[synset_id][i + 1].append(f)

        return candidate2features


    def _calculate_init_features(self, synset_id, candidate2features):
        features = []
        init_features = candidate2features[synset_id]

        features.append(init_features[0])
        features.append(np.log2(2 + init_features[0]))

        features.append(np.min(init_features[1]))
        features.append(np.mean(init_features[1]))
        features.append(np.max(init_features[1]))

        return features

    def _calculate_wiktionary_features(self, target_word, synset):
        # 1 feature for direct syn, 1 feature for hypo syn, 1 feature for direct hyper, 1 feature for hypo hyper
        if len(self.wiktionary) == 0:
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

        tw_synonyms = get_all_wkt(target_word, self.wiktionary, 'synonym')
        tw_hypernyms = get_all_wkt(target_word, self.wiktionary, 'hypernym')
        tw_meaning = []
        for wikt_doc_info in self.wiktionary.get(target_word, []):
            tw_meaning.append('_'.join(wikt_doc_info['meaning']).replace(' ', '_'))
        tw_meaning = '_'.join(tw_meaning)

        synset_synonyms = set()
        for word in synset.synset_words:
            synset_synonyms.update(get_all_wkt(word, self.wiktionary, 'synonym'))

        hypo_synonyms = set()
        hypo_words = set()
        for hypo in synset.rels.get('hyponym', []):
            hypo_words.update(hypo.synset_words)
            for word in hypo.synset_words:
                hypo_synonyms.update(get_all_wkt(word, self.wiktionary, 'synonym'))

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

    def _calculate_synset_similarity(self, w, synset):
        f_lists = [[], [], [], []]
        for synset_word in synset.synset_words:
            if synset_word in self.vector_model:
                f_lists[0].append(self.vector_model.similarity(w, synset_word))
        for hyponym in synset.rels.get('hyponym', []):
            hyponym_sim = []
            for synset_word in hyponym.synset_words:
                if synset_word in self.vector_model:
                    hyponym_sim.append(self.vector_model.similarity(w, synset_word))
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

    @staticmethod
    def _get_train_true_info(train_df):
        reference = {}
        w2true = {}
        for _, row in train_df.iterrows():
            word = row['word']
            target_gold = json.loads(row['target_gold'])
            reference[word] = target_gold
            w2true[word] = set()
            for t in target_gold:
                w2true[word].update(t)

        return reference, w2true

    @staticmethod
    def _add_true(df, word2true):
        true_col = []
        for i, row in df.iterrows():
            word = row['word']
            cand = row['cand']
            label = int(cand in word2true[word])
            true_col.append(label)

        df['label'] = true_col

    def _train_predict_cv(self, df_features, ref, folds=3):
        non_features_col_num = 3
        features_len = len(df_features.columns) - non_features_col_num
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
            mean_ap, mean_rr = get_score(cur_ref, self._from_df_to_pred(test_df), k=10)
            eval_res = [mean_ap, mean_rr]

            models.append(clf)

            print(eval_res)
            results.append(eval_res)

        print(f'Averaged results = {np.mean(results, axis=0)}')
        return models, features

    def _filter_most_sim_by_type(self, word, most_similar_words):
        filtered_word_list = []
        banned_words = set()
        for synid in self.thesaurus.sense2synid.get(word, []):
            synset = self.thesaurus.synsets[synid]
            banned_words.update(synset.synset_words)

        for w, score in most_similar_words:
            w = w.replace('ё', 'е')
            if w not in self.thesaurus.sense2synid:
                continue

            if w in banned_words:
                continue

            found_sense = False
            for synid in self.thesaurus.sense2synid[w]:
                if self.thesaurus.synsets[synid].synset_type == self.word_type:
                    found_sense = True

            if found_sense is True:
                filtered_word_list.append([w, score])

        return filtered_word_list

    def _filter_most_sim(self, word, most_similar_words):
        filtered_word_list = []
        for w, score in most_similar_words:
            if not w.startswith('__s'):
                continue
            
            synid = w[len('__s'):]
            if synid in self.thesaurus.synsets and self.thesaurus.synsets[synid].synset_type == self.word_type:
                filtered_word_list.append([synid, score])

        return filtered_word_list

    @staticmethod
    def _from_df_to_pred(df):
        pred = {}
        for i, row in df.iterrows():
            word = row['word']
            if word not in pred:
                pred[word] = []
            cand = row['cand']
            pred[word].append(cand)
        return pred

    @staticmethod
    def _create_predict_df(df, topk=10):
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



def calculate_synset_vectors(vector_model, thesaurus, black_list=[]):
    word2vec = {}
    for synid in tqdm(thesaurus.synsets):
        synset = thesaurus.synsets[synid]
        synset_word_id = f'__s{synid}'
        synset_vectors = []
        for word in synset.synset_words:
            if word in vector_model and word not in black_list:
                synset_vectors.append(vector_model.vectors[word])

        if len(synset_vectors) > 0:
            word2vec[synset_word_id] = np.mean(synset_vectors, axis=0)

    #for word in vector_model.vectors.vocab:
    #    word2vec[word] = vector_model.vectors[word]

    return word2vec


def reinit_vector_model(model, black_list_words):
    word2vec = calculate_synset_vectors(model.vector_model, model.thesaurus, black_list_words)
    temp_w2w_path = './_w2v_tmp.wv'
    with codecs.open(temp_w2w_path, 'w', 'utf-8') as file_descr:
        wv_size = word2vec[list(word2vec.keys())[0]].shape[0]
        nwords = len({w: word2vec[w] for w in word2vec if word2vec[w].shape[0] == wv_size})
        nwords += len(model.vector_model.vectors.vocab)

        print(nwords)
        file_descr.write(f'{nwords} {wv_size}')
        for w in tqdm(word2vec):
            if word2vec[w].shape[0] != wv_size:
                continue
            vector = ' '.join([str(val) for val in word2vec[w]])
            file_descr.write(f'\n{w} {vector}')

        for w in model.vector_model.vectors.vocab:
            vector = ' '.join([str(val) for val in model.vector_model.vectors[w]])
            file_descr.write(f'\n{w} {vector}')

    model._load_vectors({'embeddings_path': temp_w2w_path})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang')
    parser.add_argument('--pos')
    parser.add_argument('--wkt', action='store_true')
    parser.add_argument('--embeddings_path')
    parser.add_argument('--result_paths', nargs='+', default=None)
    parser.add_argument('--search_by_word', action='store_true')
    parser.add_argument('--exec_flags', nargs='+', default=None, type=int)
    args = parser.parse_args()

    config = {
        'pos': args.pos,
        'topk': 40,
        'lang': args.lang,
        'ruthes': False,
        'wkt': args.wkt,
        'embeddings_path': args.embeddings_path,
        'search_by_word': args.search_by_word,
        'ruthes': False
    }

    # hardcoded for now
    # todo: to configs \ params
    if args.lang == 'en':
        config['thesaurus_dir'] = 'data/wordnets/WordNet-1.6'
        config['wiktionary_dump_path'] = 'data/wiki/enwiktionary-20210201-pages-articles-multistream.xml'
        if args.pos == 'N':
            train_path = 'data/tasks/train_nouns_en_1.6_0.05_14_01_21.tsv'
            
            assert len(args.result_paths) == 3
            predict_data = []
            if args.exec_flags is None or args.exec_flags[0]:
                predict_data.append({'thesaurus_dir': 'data/wordnets/WordNet-1.6',
                'test_path': 'data/tasks/nouns_en.1.6-3.0.tsv', 'result_path': args.result_paths[0]})
            if args.exec_flags is None or args.exec_flags[1]:
                predict_data.append({'thesaurus_dir': 'data/wordnets/WordNet-1.7',
                'test_path': 'data/tasks/nouns_en.1.7-3.0.tsv', 'result_path': args.result_paths[1]})
            if args.exec_flags is None or args.exec_flags[2]:
                predict_data.append({'thesaurus_dir': 'data/wordnets/WordNet-2.0',
                'test_path': 'data/tasks/nouns_en.2.0-3.0.tsv', 'result_path': args.result_paths[2]})
        elif args.pos == 'V':
            train_path = 'data/tasks/train_verbs_en_1.6_0.3_14_01_21.tsv'
            assert len(args.result_paths) == 3
            predict_data = []
            if args.exec_flags is None or args.exec_flags[0]:
                predict_data.append({'thesaurus_dir': 'data/wordnets/WordNet-1.6',
                'test_path': 'data/tasks/verbs_en.1.6-3.0.tsv', 'result_path': args.result_paths[0]})
            if args.exec_flags is None or args.exec_flags[1]:
                predict_data.append({'thesaurus_dir': 'data/wordnets/WordNet-1.7',
                'test_path': 'data/tasks/verbs_en.1.7-3.0.tsv', 'result_path': args.result_paths[1]})
            if args.exec_flags is None or args.exec_flags[2]:
                predict_data.append({'thesaurus_dir': 'data/wordnets/WordNet-2.0',
                'test_path': 'data/tasks/verbs_en.2.0-3.0.tsv', 'result_path': args.result_paths[2]})


    elif args.lang == 'ru':
        config['thesaurus_dir'] = 'data/wordnets/RuWordNet'
        config['wiktionary_dump_path'] = 'data/wiki/ruwiktionary-20210201-pages-articles-multistream.xml'

        if args.pos == 'N':
            train_path = 'data/tasks/test_nouns_0.2_fixed_12_07_20.tsv'

            assert len(args.result_paths) == 2

            predict_data = []
            predict_data.append({'thesaurus_dir': 'data/wordnets/RuWordNet',
                'test_path': 'data/tasks/ruwordnet_non-restricted_nouns.tsv', 'result_path': args.result_paths[0]})
            predict_data.append({'thesaurus_dir': 'data/wordnets/RuWordNet',
                'test_path': 'data/tasks/nouns_private_subgraphs.tsv', 'result_path': args.result_paths[1]})
        elif args.pos == 'V':
            train_path = 'data/tasks/test_verbs_0.2_11_08_20.tsv'

            assert len(args.result_paths) == 2

            predict_data = []
            predict_data.append({'thesaurus_dir': 'data/wordnets/RuWordNet',
                'test_path': 'data/tasks/ruwordnet_non-restricted_verbs.tsv', 'result_path': args.result_paths[0]})
            predict_data.append({'thesaurus_dir': 'data/wordnets/RuWordNet',
                'test_path': 'data/tasks/verbs_private_subgraphs.tsv', 'result_path': args.result_paths[1]})
            

    print(config)
    model = HypernymPredictModel(config)

    train_df = pd.read_csv(train_path).rename(columns={'target_word': 'word'})
    train_df['word'] = train_df['word'].apply(lambda x: x.lower())

    if not args.search_by_word:
        reinit_vector_model(model, train_df['word'])

    model.train(train_df)

    for predict_setup in predict_data:
        model._load_thesaurus({'thesaurus_dir': predict_setup['thesaurus_dir'], 'lang': args.lang, 'ruthes': False})
        if not args.search_by_word:
            model._load_vectors({'embeddings_path': args.embeddings_path})
            reinit_vector_model(model, [])
        model.topk = 20

        test_path = predict_setup['test_path']
        predict_df = pd.read_csv(test_path, header=None, sep='\t').rename(columns={0: 'word'})
        predict_df['word'] = predict_df['word'].apply(lambda x: x.lower())

        pred = model.predict(list(set(predict_df['word'])))
        pred.to_csv(predict_setup['result_path'], sep='\t', index=False, header=False)

        submitted = read_dataset(predict_setup['result_path'])
        reference = read_dataset(test_path, json.loads)
        map, mrr = get_score_list(reference, submitted)

        print(predict_setup['result_path'])
        print(f'MAP: {np.mean(map)} +- {np.std(map)}')
        print(f'MRR: {np.mean(mrr)} +- {np.std(mrr)}')
