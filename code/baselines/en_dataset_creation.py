import json
import os
import sys
from itertools import combinations

import networkx as nx
from nltk.corpus import WordNetCorpusReader


def extract_new_lemmas(synsets, wordnet, pos):
    new_lemmas = set([lemma.name() for synset in synsets for lemma in synset.lemmas()])
    old_lemmas = set(wordnet.all_lemma_names(pos))
    unique_lemmas = new_lemmas.difference(old_lemmas)
    return unique_lemmas


def cluster_hypernyms(hypernyms):
    G = nx.Graph()
    for hypernym in hypernyms:
        G.add_node(hypernym)
    for h1, h2 in combinations(hypernyms, 2):
        if (h2 in h1.hypernyms()) or (h1 in h2.hypernyms()):
            G.add_edge(h1, h2)
    return list(nx.connected_components(G))


def generate_gold(lemmas, wordnet_new, wordnet_old, pos):
    result = {}
    for lemma in lemmas:
        synsets = wordnet_new.synsets(lemma, pos)
        all_hypernyms = []
        for synset in synsets:
            direct_hypernyms = synset.hypernyms() + synset.instance_hypernyms()
            all_hypernyms.extend(direct_hypernyms)
            second_order_hypernyms = [hypernym for synset in direct_hypernyms for hypernym in synset.hypernyms()]
            all_hypernyms.extend(second_order_hypernyms)
        if all([hypernym in wordnet_old for hypernym in all_hypernyms]) and all_hypernyms:
            result[lemma] = cluster_hypernyms(all_hypernyms)
    return result


def save(gold_dict, outdir, outputfile):
    with open(os.path.join(outdir, outputfile), 'w', encoding='utf-8') as w, \
            open(os.path.join(outdir, "no_labels_" + outputfile), 'w', encoding='utf-8') as f:
        for lemma, hypernyms in gold_dict.items():
            f.write(lemma + "\n")
            for group in hypernyms:
                names = [hypernym.name() for hypernym in group]
                w.write(lemma + "\t" + json.dumps(names) + "\n")


def main():
    # python en_dataset_creation.py ../../datasets/WNs ../../datasets/en/ 2.0 3.0

    if len(sys.argv) < 3:
        raise Exception(
            "The following arguments are required:<WordNet path> <output_path> <old_version_float> <new_version_float>")

    path = sys.argv[1]
    out_path = sys.argv[2]
    old_version = sys.argv[3]

    if len(sys.argv) == 5:
        new_version = sys.argv[4]
    else:
        new_version = "3.0"

    wn2 = WordNetCorpusReader(os.path.join(path, 'WN' + old_version), None)
    wn3 = WordNetCorpusReader(os.path.join(path, 'WN' + new_version), None)

    for pos in ['nouns', 'verbs']:
        synsets_2n = set(wn2.all_synsets(pos[0]))
        synsets_3n = set(wn3.all_synsets(pos[0]))

        reference_nouns = synsets_3n.intersection(synsets_2n)
        new = extract_new_lemmas(synsets_3n.difference(synsets_2n), wn2, pos[0])
        hypernyms = generate_gold(new, wn3, reference_nouns, pos[0])

        print(f"Len {pos} {len(hypernyms)}")
        save(dict(hypernyms), out_path, f"{pos}_en.{old_version}-{new_version}.tsv")


if __name__ == '__main__':
    main()
