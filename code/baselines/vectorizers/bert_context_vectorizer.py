import argparse
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np
import json

from vectorizers.bert_model import BertPretrained


class BertContextVectorizer:
    def __init__(self, model_path):
        self.bert = BertPretrained(model_path)

    # -------------------------------------------------------------
    # update vectors
    # -------------------------------------------------------------

    def update_vectors(self, current_vectors, text_path, output_path, batch_size):
        counter = 0
        batch = []
        position_batch = []

        with open(text_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                counter += 1
                tokens, positions = json.loads(line)
                tokens = tokens.split()
                if len(self.bert.tokenize(tokens)[0]) > 510:
                    continue
                batch.append(tokens)
                position_batch.append(positions)
                if counter % batch_size == 0:
                    for word, vector in self.get_vectors(batch, position_batch):
                        if not any(np.isnan(vector)) and word in current_vectors:
                                current_vectors[word][0] = current_vectors[word][0] + vector
                                current_vectors[word][1] += 1
                    batch = []
                    position_batch = []

            for word, vector in self.get_vectors(batch, position_batch):
                if not any(np.isnan(vector)) and word in current_vectors:
                        current_vectors[word][0] = current_vectors[word][0] + vector
                        current_vectors[word][1] += 1

        self.save_as_w2v_mean(current_vectors, output_path)

    # -------------------------------------------------------------
    # get vectors
    # -------------------------------------------------------------

    def get_vectors(self, sentences, indices):
        word_vectors = []
        batch = self.bert.vectorize_sentences(sentences)

        for sent_vectors, tokens, sent_indices in zip(batch, sentences, indices):
            assert sent_vectors.shape[0] == len(tokens)
            word_vectors.extend([(synset, self.get_avg_vector(sent_vectors, borders, synset, tokens))
                                 for synset, borders in sent_indices])
        return word_vectors

    @staticmethod
    def get_avg_vector(vectors, borders, s, t):
        start, end = borders
        return np.mean(vectors[start:end], 0)

    # -------------------------------------------------------------
    # save vectors
    # -------------------------------------------------------------

    @staticmethod
    def save_as_w2v_mean(dictionary, output_path):
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{len(dictionary)} {list(dictionary.values())[0][0].shape[-1]}\n")
            for word, (vector, count) in dictionary.items():
                mean_vector = vector / count if count != 0 else vector
                vector_line = " ".join(map(str, mean_vector))
                w.write(f"{word.upper()} {vector_line}\n")


def get_vectors(filepath):
    w2v = KeyedVectors.load_word2vec_format(filepath, binary=False)
    return {word: [w2v[word], 1] for word in w2v.vocab}


def parse_args():
    parser = argparse.ArgumentParser(prog='BERT context vectorizer')
    parser.add_argument('--bert_path', type=str, dest="bert_path", help='bert model dir')
    parser.add_argument('--vectors_path', type=str, dest="vectors_path", help='vectors_path')
    parser.add_argument('--output_path', type=str, dest="output_path", help='output_path')
    parser.add_argument('--texts_dir', type=str, dest="texts_dir", help='texts_dir')
    parser.add_argument('--batch_size', type=int, dest='batch_size', help='batch size', default=16)
    return parser.parse_args()


if __name__ == '__main__':
    # --bert_path "models/pretrained_berts/rubert_cased"
    # --texts_dir "models/parsed_sentences_with_occurrences"
    # --vectors_path "models/vectors/bert/ru/tokens/nouns_public.txt"
    # --output_path "models/vectors/bert/ru/tokens/nouns_public_context.txt"
    args = parse_args()
    bcv = BertContextVectorizer(args.bert_path)
    vectors = get_vectors(args.vectors_path)

    print(f"Processing {args.texts_dir}")
    bcv.update_vectors(vectors, args.texts_dir, args.output_path, args.batch_size)
