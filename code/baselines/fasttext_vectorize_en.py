import os
import sys
from collections import defaultdict

from nltk.corpus import WordNetCorpusReader

from vectorizers.fasttext_vectorizer import FasttextVectorizer


def compute_synsets_from_wordnets(wordnet, pos):
    synsets = set(wordnet.all_synsets(pos))
    return extract_senses(synsets)


def extract_senses(synsets) -> dict:
    result = defaultdict(list)
    for synset in synsets:
        for lemma in synset.lemmas():
            result[synset.name()].append(lemma.name().replace("_", " "))
    return result


def process_data(vectorizer, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    vectorizer.vectorize_multiword_data(dataset, output_file, to_upper=False)


def main():
    # python fasttext_vectorize_en.py models/cc.en.300.bin ../../datasets/WNs 2.0 models/vectors/fasttext/en ../../datasets/en

    if len(sys.argv) < 7:
        raise Exception(
            "Required arguments: <fasttext-path> <wn-dir> <old-version> <new-version> <vector-path> <input-path>")

    ft = FasttextVectorizer(sys.argv[1])
    old = sys.argv[3]
    new = sys.argv[4]
    wn2 = WordNetCorpusReader(os.path.join(sys.argv[2], "WN" + old), None)
    vector_path = sys.argv[5]
    if not os.path.exists(vector_path):
        os.makedirs(vector_path)
    data_path = sys.argv[6]

    for pos in ['nouns', 'verbs']:
        synsets = compute_synsets_from_wordnets(wn2, pos[0])
        ft.vectorize_groups(synsets, os.path.join(vector_path, f"{pos}_wordnet_fasttext_{old}-{new}.txt"), False)
        process_data(ft, os.path.join(data_path, f"no_labels_{pos}_en.{old}-{new}.tsv"),
                     os.path.join(vector_path, f"{pos}_fasttext_{old}-{new}.txt"))


if __name__ == '__main__':
    main()
