import argparse
from collections import defaultdict

from nltk.corpus import WordNetCorpusReader
from tqdm import tqdm
import numpy as np

from ruwordnet.ruwordnet_reader import RuWordnet
from vectorizers.bert_model import BertPretrained
from fasttext_vectorize_en import compute_synsets_from_wordnets


class BertVectorizer:
    def __init__(self, model_path):
        self.bert = BertPretrained(model_path)

    # -------------------------------------------------------------
    # get ruwordnet
    # -------------------------------------------------------------

    def vectorize_groups(self, synsets, output_path, to_upper=True):
        vectors = {synset: np.mean([np.mean(sentence_vectors, 0) for sentence_vectors in
                                    self.bert.vectorize_sentences([i.split() for i in texts])], 0)
                   for synset, texts in tqdm(synsets.items())}
        self.save_as_w2v(vectors, output_path, to_upper)

    # -------------------------------------------------------------
    # get dataset
    # -------------------------------------------------------------

    def vectorize_data(self, data, output_path, upper):
        batch = [self.bert.vectorize_sentences([d.split("_")])[0] for d in tqdm(data)]
        vectors = {word: np.mean(sentence_vectors, 0) for sentence_vectors, word in zip(batch, data)}
        self.save_as_w2v(vectors, output_path, upper=upper)

    # -------------------------------------------------------------
    # save results
    # -------------------------------------------------------------

    @staticmethod
    def save_as_w2v(dictionary, output_path, upper=True):
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{len(dictionary)} {list(dictionary.values())[0].shape[-1]}\n")
            for word, vector in dictionary.items():
                vector_line = " ".join(map(str, vector))
                if upper:
                    word = word.upper()
                w.write(f"{word} {vector_line}\n")


def read_file(filename, lower=True):
    with open(filename, 'r', encoding='utf-8') as f:
        dataset = f.read()
        if lower:
            dataset = dataset.lower()
        return dataset.split("\n")[:-1]


def parse_args():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--bert_path', type=str, dest="bert_path", help='bert model dir')
    parser.add_argument('--output_path', type=str, dest="output_path", help='output_path')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "wordnet" command
    ruwordnet_parser = subparsers.add_parser('wordnet', help='wordnet help')
    ruwordnet_parser.add_argument('--wordnet_old', type=str, help='wordnet old database path')
    ruwordnet_parser.add_argument('--wordnet_new', type=str, help='wordnet new database path')
    ruwordnet_parser.add_argument('--pos', choices='nv', help="choose pos-tag to subset wordnet")

    # create the parser for the "ruwordnet" command
    wordnet_parser = subparsers.add_parser('ruwordnet', help='ruwordnet help')
    wordnet_parser.add_argument('--ruwordnet_path', type=str, help='ruwordnet database path')
    wordnet_parser.add_argument('--pos', choices='NV', help="choose pos-tag to subset ruwordnet")

    # create the parser for the "data" command
    data_parser = subparsers.add_parser('data', help='data help')
    data_parser.add_argument('--data_path', type=str, dest="data_path", help='path to test data')
    data_parser.add_argument('--upper', action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bert_vectorizer = BertVectorizer(args.bert_path)

    if 'ruwordnet_path' in args:
        ruwordnet = RuWordnet(args.ruwordnet_path, None)
        synsets = defaultdict(list)
        for sense_id, synset_id, text in ruwordnet.get_all_senses():
            if synset_id.endswith(args.pos):
                synsets[synset_id].append(text.lower())
        bert_vectorizer.vectorize_groups(synsets, args.output_path, to_upper=False)

    if 'wordnet_old' in args:
        wn_old = WordNetCorpusReader(args.wordnet_old, None)
        wn_new = WordNetCorpusReader(args.wordnet_new, None)
        synsets = compute_synsets_from_wordnets(wn_old, wn_new, args.pos)
        bert_vectorizer.vectorize_groups(synsets, args.output_path, to_upper=False)

    if "data_path" in args:
        data = read_file(args.data_path, lower=args.upper)
        bert_vectorizer.vectorize_data(data, args.output_path, upper=args.upper)
