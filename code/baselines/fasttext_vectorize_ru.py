import os
import sys
from collections import defaultdict

from ruwordnet.ruwordnet_reader import RuWordnet
from vectorizers.fasttext_vectorizer import FasttextVectorizer


def process_data(vectorizer, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = f.read().lower().split("\n")[:-1]
    vectorizer.vectorize_words(dataset, output_file)


if __name__ == '__main__':
    # python3 fasttext_vectorize_ru.py models/cc.ru.300.bin ../../datasets/ruwordnet.db \
    # models/vectors/fasttext/ru/ ../../datasets/ru/

    if len(sys.argv) < 5:
        raise Exception("The following arguments required:<fasttext_path> <ruwordnetdb_path> <output_path> <data_path>")

    ft = FasttextVectorizer(sys.argv[1])
    ruwordnet = RuWordnet(db_path=sys.argv[2], ruwordnet_path=None)
    vector_path = sys.argv[3]
    if not os.path.exists(vector_path):
        os.makedirs(vector_path)
    data_path = sys.argv[4]

    # ----------------------
    # vectorize synsets
    # ----------------------
    noun_synsets = defaultdict(list)
    verb_synsets = defaultdict(list)

    for sense_id, synset_id, text in ruwordnet.get_all_senses():
        if synset_id.endswith("N"):
            noun_synsets[synset_id].append(text.lower())
        elif synset_id.endswith("V"):
            verb_synsets[synset_id].append(text.lower())

    ft.vectorize_groups(noun_synsets, os.path.join(vector_path, "ruwordnet_nouns.txt"))
    ft.vectorize_groups(verb_synsets, os.path.join(vector_path, "ruwordnet_verbs.txt"))

    # ----------------------
    # vectorize data
    # ----------------------

    paths = [("ruwordnet_non-restricted-nouns_no_labels.tsv", "non-restricted_nouns_fasttext.txt"),
             ("ruwordnet_non-restricted-verbs_no_labels.tsv", "non-restricted_verbs_fasttext.txt"),
             ("nouns_public_no_labels.tsv", "nouns_public.txt"),
             ("verbs_public_no_labels.tsv", "verbs_public.txt"),
             ("nouns_private_no_labels.tsv", "nouns_private.txt"),
             ("verbs_private_no_labels.tsv", "verbs_private.txt")]

    for input_path, output_path in paths:
        process_data(ft, os.path.join(data_path, input_path),
                     os.path.join(vector_path, output_path))
