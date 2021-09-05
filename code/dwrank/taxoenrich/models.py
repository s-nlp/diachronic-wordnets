import os
import codecs
import json
from tqdm import tqdm
import pymorphy2
import xml.etree.ElementTree as ET
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
import numpy as np
import pickle
import codecs

class SynSet:
    def __init__(self, synset_id, synset_name, synset_type, words):
        self.synset_type = synset_type
        self.synset_name = synset_name
        self.synset_id = synset_id
        self.synset_words = words
        self.rels = {}
        self.total_hyponyms = None

    def add_rel(self, synset, rel_type):
        if rel_type not in self.rels:
            self.rels[rel_type] = []

        self.rels[rel_type].append(synset)

    def _calc_total_hyponyms(self):
        if self.total_hyponyms is not None:
            return self.total_hyponyms

        hyponym_list = self.rels.get('hyponym', [])
        self.total_hyponyms = len(hyponym_list) + sum(list(map(lambda x: x._calc_total_hyponyms(), hyponym_list)))

        return self.total_hyponyms


class WordNet:
    def __init__(self):
        self.synsets = {}
        self.senses = set()
        self.sense2synid = {}

    def _calc_synset_children(self):
        for synset_id in self.synsets:
            try:
                self.synsets[synset_id]._calc_total_hyponyms()
            except:
                print(synset_id)
        top_synsets = []
        for synset_id in self.synsets:
            hypernym_count = len(self.synsets[synset_id].rels.get('hypernym', []))
            if hypernym_count == 0:
                top_synsets.append(synset_id)

    def filter_thesaurus(self, words):
        for w in words:
            if w not in self.sense2synid:
                continue
            synsets_ids = self.sense2synid[w]
            for synset_id in synsets_ids:
                try:
                    synset = self.synsets[synset_id]
                    parents = synset.rels.get('hypernym', [])
                    childs = synset.rels.get('hyponym', [])
                    for parent_synset in parents:
                        del parent_synset.rels['hyponym'][parent_synset.rels['hyponym'].index(synset)]
                    for child_synset in childs:
                        del child_synset.rels['hypernym'][child_synset.rels['hypernym'].index(synset)]

                    for parent_synset in parents:
                        for child_synset in childs:
                            if child_synset in parent_synset.rels['hyponym']:
                                continue
                            parent_synset.rels['hyponym'].append(child_synset)
                            child_synset.rels['hypernym'].append(parent_synset)

                    del self.synsets[synset_id]

                    for sense in synset.synset_words:
                        if sense in self.senses:
                            self.senses.remove(sense)
                        if sense in self.sense2synid:
                            del self.sense2synid[sense]
                except Exception as e:
                    print(synset_id)
                    print(e)
        #self._calc_synset_children()

class EnWordNet(WordNet):
    def __init__(self, wordnet_root):
        super().__init__()
        self.rel_map = {
            '@': 'hypernym',
            '@i': 'hypernym',
            '~': 'hyponym',
            '~i': 'hyponym'
            }
        self._load_wordnet(wordnet_root)

    def _load_wordnet(self, wordnet_root):
        self._load_synsets(wordnet_root)
        #self._calc_synset_children()

    def _load_index(self, wordnet_root):
        synsets_paths = {
            'N': os.path.join(wordnet_root, os.path.join('dict', 'index.noun')),
            'A': os.path.join(wordnet_root, os.path.join('dict', 'index.adj')),
            'V': os.path.join(wordnet_root, os.path.join('dict', 'index.verb')),
            'R': os.path.join(wordnet_root, os.path.join('dict', 'index.adv')),
        }

        index = {t: {} for t in synsets_paths}
        for synset_type, synset_path in synsets_paths.items():
            with codecs.open(synset_path, 'r', 'utf-8') as file:
                for line in file:
                    if line.startswith('\t') or line.startswith(' '):
                        continue
                    line_content = line.strip().split()
                    word = line_content[0]
                    synset_count = int(line_content[2])
                    ptr_count = int(line_content[3])
                    index[synset_type][word] = line_content[4 + ptr_count + 2:]
                    assert len(line_content[4 + ptr_count + 2:]) == synset_count

        return index


    def _load_synsets(self, wordnet_root, black_list_synsets=None, black_list_senses=None):
        synsets_paths = {
            'N': os.path.join(wordnet_root, os.path.join('dict', 'data.noun')),
            'A': os.path.join(wordnet_root, os.path.join('dict', 'data.adj')),
            'V': os.path.join(wordnet_root, os.path.join('dict', 'data.verb')),
            'R': os.path.join(wordnet_root, os.path.join('dict', 'data.adv')),
        }
        index = self._load_index(wordnet_root)
        for synset_type, synset_path in synsets_paths.items():
            with codecs.open(synset_path, 'r', 'utf-8') as file:
                for line in file:
                    if line.startswith('\t') or line.startswith(' '):
                        continue
                    try:
                        synset_info = self._read_line(line)
                        synset_id = synset_info['id']
                        synset_name = synset_info['name']
                        synset_name_idx = index[synset_type][synset_name].index(synset_info['id'][:-1]) + 1
                        synset_name = f'{synset_name}.{synset_type.lower()}.{self._to2digit(synset_name_idx)}'
                        synset_words = synset_info['words']

                    except Exception as e:
                        print(line)
                        raise e

                    if black_list_synsets is not None and synset_id in black_list_synsets:
                        self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, set())
                    else:
                        if black_list_senses is not None:
                            synset_words = [w for w in synset_words if w not in black_list_senses]

                        self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, set(synset_words))
                        self.senses.update(synset_words)
                        for sense in synset_words:
                            if sense not in self.sense2synid:
                                self.sense2synid[sense] = []
                            self.sense2synid[sense].append(synset_id)

                    for rel_type, rel_synset_id in synset_info['rels']:
                        self.synsets[synset_id].add_rel(rel_synset_id, rel_type)

        for synset_id in self.synsets:
            for rel_type in self.synsets[synset_id].rels:
                for i, rel_synset_id in enumerate(self.synsets[synset_id].rels[rel_type]):
                    self.synsets[synset_id].rels[rel_type][i] = self.synsets[rel_synset_id]

    def _read_line(self, line):
        synset_info = {}

        ID_IDX = 0
        POS_IDX = 2
        W_LEN_IDX = 3
        WORDS_SHIFT = 4

        line_content = line.strip().split()
        synset_id = line_content[ID_IDX]
        pos = line_content[POS_IDX]
        if pos == 's':
            pos = 'a'
        synset_id += pos

        synset_info['id'] = synset_id

        synset_words_len = int(line_content[W_LEN_IDX], 16)
        synset_words = []
        
        for i in range(synset_words_len):
            synset_words.append(line_content[WORDS_SHIFT + i * 2])


        synset_info['words'] = []
        for w in synset_words:
            if '(' in w:
                w = w[:w.find('(')]
            w = w.lower()
            if w not in synset_info['words']:
                synset_info['words'].append(w)
            
        synset_name = synset_info['words'][0]
        synset_info['name'] = synset_name

        RELS_SHIFT = WORDS_SHIFT + synset_words_len * 2 + 1
        rels_count = int(line_content[RELS_SHIFT - 1])
        cur_rel_shift = RELS_SHIFT
        rels = []
        while len(rels) != rels_count and cur_rel_shift < len(line_content) - 2 and line_content[cur_rel_shift] != '|':
            rel_type = self.rel_map.get(line_content[cur_rel_shift], line_content[cur_rel_shift])

            rel_synset_id = line_content[cur_rel_shift + 1]
            pos = line_content[cur_rel_shift + 2]
            if pos == 's':
                pos = 'a'
            rel_synset_id += pos
            rels.append((rel_type, rel_synset_id))

            cur_rel_shift += 4
            '''
            while cur_rel_shift < len(line_content)  - 2 and line_content[cur_rel_shift] == '+':
                rel_synset_id = line_content[cur_rel_shift + 1]
                pos = line_content[cur_rel_shift + 2]
                if pos == 's':
                    pos = 'a'
                rel_synset_id += pos
                rels.append((rel_type, rel_synset_id))
                cur_rel_shift += 4
            '''
        assert len(rels) == rels_count
        rels = [rel for rel in rels if rel[1][-1] == synset_id[-1]]
        synset_info['rels'] = rels

        return synset_info

    def _to2digit(self, num):
        return '0' * (2 - len(str(num))) + str(num)


class RuWordNet(WordNet):
    def __init__(self, wordnet_root):
        super().__init__()
        self._load_wordnet(wordnet_root)

    def _load_wordnet(self, wordnet_root):
        self._load_synsets(wordnet_root)
        self._load_rels(wordnet_root)
        #self._calc_synset_children()

    def _load_synsets(self, wordnet_root, black_list_synsets=None, black_list_senses=None):
        synsets_paths = {
            'N': os.path.join(wordnet_root, 'synsets.N.xml'),
            'A': os.path.join(wordnet_root, 'synsets.A.xml'),
            'V': os.path.join(wordnet_root, 'synsets.V.xml')
        }

        morph_analizer = pymorphy2.MorphAnalyzer()
        for synset_type, synset_path in synsets_paths.items():
            root = ET.parse(synset_path).getroot()
            for synset in root.getchildren():
                synset_name = synset.attrib['ruthes_name'].lower()
                synset_id = synset.attrib['id']
                if black_list_synsets is not None and synset_id in black_list_synsets:
                    self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, set())
                    continue
                synset_words = set()
                for sense in synset.getchildren():
                    word = sense.text.lower().replace('ё', 'е')
                    split_word = word.split()
                    split_word = [morph_analizer.parse(w)[0].normal_form.replace('ё', 'е') for w in split_word]
                    sense = '_'.join(split_word)
                    if black_list_senses is not None and sense in black_list_senses:
                        continue
                    synset_words.add(sense)

                self.senses.update(synset_words)
                self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, synset_words)
                for sense in synset_words:
                    if sense not in self.sense2synid:
                        self.sense2synid[sense] = []
                    self.sense2synid[sense].append(synset_id)

    def _load_rels(self, wordnet_root):
        synsets_rels_paths = {
            'N': os.path.join(wordnet_root, 'synset_relations.N.xml'),
            'A': os.path.join(wordnet_root, 'synset_relations.A.xml'),
            'V': os.path.join(wordnet_root, 'synset_relations.V.xml')
        }

        for synset_rel_type, synset_rels_path in synsets_rels_paths.items():
            root = ET.parse(synset_rels_path).getroot()
            for synset_rel in root.getchildren():
                synset_id = synset_rel.attrib['parent_id']
                rel_synset_id = synset_rel.attrib['child_id']
                rel_type = synset_rel.attrib['name']
                if rel_type not in ['hyponym', 'hypernym', 'instance hypernym']:
                    continue

                if rel_type == 'instance hypernym':
                    rel_type = 'hypernym'

                if synset_id not in self.synsets or rel_synset_id not in self.synsets:
                    continue

                self.synsets[synset_id].add_rel(self.synsets[rel_synset_id], rel_type)

class RuThes(WordNet):
    def __init__(self, concepts_path):
        super().__init__()
        self._load_synsets(concepts_path)
        self._load_rels(concepts_path)
        self._calc_hypo_rels()

    def _load_synsets(self, concepts_path):
        with codecs.open(concepts_path, 'r', 'utf-8') as file:
            for line in tqdm(file):
                concept_info = json.loads(line)
                self._add_concept(concept_info)

    def _load_rels(self, concepts_path):
        with codecs.open(concepts_path, 'r', 'utf-8') as file:
            for line in tqdm(file):
                concept_info = json.loads(line)
                self._add_rel(concept_info)


    def _add_concept(self, concept_info):
        synset_id = concept_info['conceptid']
        synset_name = concept_info['conceptstr']
        synset_words = concept_info['synonyms']

        #concept_domain = concept_info['domainmask']

        synset_type = 'N' # todo calculate POS

        #print(synset_words)

        #[{'textentryid': 1149044, 'textentrystr': 'ФРЕЙМВОРК APACHE MAVEN', 'lementrystr': 'ФРЕЙМВОРК APACHE MAVEN', 'isambig': '0', 'isarguable': '0'}, {'textentryid': 1149958, 'textentrystr': 'MAVEN', 'lementrystr': 'MAVEN', 'isambig': '0', 'isarguable': '0'}, {'textentryid': 1167907, 'textentrystr': 'APACHE MAVEN', 'lementrystr': 'APACHE MAVEN', 'isambig': '0', 'isarguable': '0'}]
        synset_words = [sense_info['lementrystr'].lower().replace(' ', '_') for sense_info in synset_words]
        self.senses.update(synset_words)
        self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, synset_words)
        for sense in synset_words:
            if sense not in self.sense2synid:
                self.sense2synid[sense] = []
            self.sense2synid[sense].append(synset_id)

    def _add_rel(self, concept_info):
        synset_id = concept_info['conceptid']
        for rel in concept_info['relats']:
            if rel['relationstr'] == 'ВЫШЕ':
                self.synsets[synset_id].add_rel(self.synsets[rel['conceptid']], 'hypernym')

    def _calc_hypo_rels(self):
        for synset_id, synset in tqdm(self.synsets.items()):
            for hyper in synset.rels.get('hypernym', []):
                hyper.add_rel(self.synsets[synset_id], 'hyponym')


class BertCls():
    def __init__(self, model_dir, max_length=16, no_cuda=False, max_bs=256, sent_mode=False):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir, do_lower_case=False)
        self.no_cuda = no_cuda
        self.device = torch.device("cpu") if self.no_cuda else torch.device("cuda")
        self.model_dir = model_dir
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.max_length = max_length
        self.max_bs = max_bs
        self.sent_mode = sent_mode
        self.hash = {}
        if os.path.exists(os.path.join(model_dir, '.hash')):
            print('Load hash')
            self.hash = pickle.load(codecs.open(os.path.join(model_dir, '.hash'), 'rb'))
        self.model.eval()

    def save_hash(self):
        pickle.dump(self.hash, codecs.open(os.path.join(self.model_dir, '.hash'), 'wb'))

    #def predict_synset(self, target_word, synset_words, return_prob=False):

    def predict(self, target_word, cand_word, return_prob=False):
        print(target_word, cand_word)
        key = ' '.join([target_word, cand_word])
        if key in self.hash:
            if return_prob is True:
                return self.hash[key]
            else:
                return np.argmax(self.hash[key], axis=1)

        with torch.no_grad():
            input_ids, attention_mask, token_type_ids = self._tokenize(target_word, cand_word)
            model_input = {"input_ids": torch.tensor(input_ids).view(1, -1).to(self.device),
                  "attention_mask": torch.tensor(attention_mask).view(1, -1).to(self.device),
                  "token_type_ids": torch.tensor(token_type_ids).view(1, -1).to(self.device)}
            output = self.model(**model_input)[0]
            probs = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        self.hash[key] = probs
        if return_prob is True:
            return probs
        else:
            return np.argmax(probs, axis=1)

    def _tokenize(self, target_word, cand_word):
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        sent = f'Понятие {target_word} - это тип понятия {cand_word}'
        if self.sent_mode is True:
            inputs = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=self.max_length,)
        else:
            inputs = self.tokenizer.encode_plus(target_word, cand_word, add_special_tokens=True, max_length=self.max_length,)

        print(inputs)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return input_ids, attention_mask, token_type_ids
        '''
        inputs = {"input_ids": torch.tensor(input_ids).view(1, -1),
                  "attention_mask": torch.tensor(attention_mask).view(1, -1),
                  "token_type_ids": torch.tensor(token_type_ids).view(1, -1)}
        return inputs
        '''

    def predict_multiple(self, target_word, cand_words, return_prob=False):
        key = ' '.join([target_word] + sorted(cand_words))
        if key in self.hash:
            if return_prob is True:
                return self.hash[key]
            else:
                return np.argmax(self.hash[key], axis=1)
        probs = []
        batch_count = len(cand_words) // self.max_bs
        if batch_count  != len(cand_words) / self.max_bs:
            batch_count += 1

        for i in range(batch_count):
            batch_cand_words = cand_words[i * self.max_bs: min((i+1) * self.max_bs, len(cand_words))]
            total_input_ids, total_attention_mask, total_token_type_ids = [], [], []
            for cand_word in batch_cand_words:
                input_ids, attention_mask, token_type_ids = self._tokenize(target_word, cand_word)
                total_input_ids.append(input_ids)
                total_attention_mask.append(attention_mask)
                total_token_type_ids.append(token_type_ids)
            model_input = {"input_ids": torch.tensor(total_input_ids).to(self.device),
                    "attention_mask": torch.tensor(total_attention_mask).to(self.device),
                    "token_type_ids": torch.tensor(total_token_type_ids).to(self.device)}
            with torch.no_grad():
                output = self.model(**model_input)[0]
                batch_probs = torch.nn.functional.softmax(output, dim=1)
            probs.append(batch_probs.detach().cpu().numpy())

        probs = np.vstack(probs)

        self.hash[key] = probs
        if return_prob is True:
            return probs
        return np.argmax(probs, axis=1)
