import argparse
import codecs
import json
import random
import os
from statistics import mean, stdev
from collections import defaultdict

# Consistent with Python 2


def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = read_fn(line_split[1])
            vocab[word].append(hypernyms)
    return vocab


def get_score_list(reference, submitted, submitted_path, k=10):
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

    write_to_file(os.path.splitext(submitted_path)[0]+"_map_scores.json", map_list)
    write_to_file(os.path.splitext(submitted_path)[0]+"_mrr_scores.json", mrr_list)

    return map_list, mrr_list


def get_score(reference, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in reference:
        reference_hypernyms = reference.get(neologism, [])
        predicted_hypernyms = predicted.get(neologism, [])

        ap_sum += compute_ap(reference_hypernyms, predicted_hypernyms, k)
        rr_sum += compute_rr([j for i in reference_hypernyms for j in i], predicted_hypernyms, k)
    return ap_sum / len(reference), rr_sum / len(reference)


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


def write_to_file(path, object):
    with open(path, 'w', encoding='utf-8') as w:
        json.dump(object, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('predicted')
    args = parser.parse_args()

    reference = read_dataset(args.reference, lambda x: json.loads(x))
    submitted = read_dataset(args.predicted)
    if len(set(reference).intersection(set(submitted))) == 0:
        raise Exception("Reference and Submitted files have no samples in common")
    elif set(reference) != set(submitted):
        print("Not all words are presented in your file", len(set(reference)), len(set(submitted)))

    mean_ap, mean_rr = get_score(reference, submitted, k=10)
    print("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))
    map_list, mrr_list = get_score_list(reference, submitted, args.predicted, k=10)
    print("{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}".format(mean(map_list), stdev(map_list), mean(mrr_list), stdev(mrr_list)))


if __name__ == '__main__':
    main()
