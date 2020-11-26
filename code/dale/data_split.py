import pandas as pd
import hashlib


def hash_float(text, salt='1', base=1000000):
    return int(hashlib.sha256((text+salt).encode('utf-8')).hexdigest(), 16) % base / base


def load_train_data(path='../data/training_data/training_nouns.tsv'):
    train_n = pd.read_csv(path, sep='\t', encoding='utf-8')
    train_n['parents_list'] = train_n.PARENTS.apply(lambda x: x.split(','))

    ttrain = train_n[train_n.synset_hash <= 0.8]
    ttest = train_n[train_n.synset_hash > 0.8]
    ttest_dev = ttest[ttest.synset_hash <= 0.82]
    ttest_test1 = ttest[(ttest.synset_hash > 0.82) & (ttest.synset_hash <= 0.84)]
    ttest_test2 = ttest[(ttest.synset_hash > 0.84) & (ttest.synset_hash <= 0.86)]
    ttest_hidden = ttest[(ttest.synset_hash > 0.86)]
    forbidden_id = set(ttest.SYNSET_ID)
    return ttrain, ttest_dev, ttest_test1, ttest_test2, ttest_hidden, forbidden_id


def split_dict(
        dataset,
        train_share = 0.8,
        dev_share = 0.02,
        test1_share = 0.02,
        test2_share = 0.02,
        hid_share = 0.14,
):
    assert train_share + dev_share + test1_share + test2_share + hid_share == 1
    train, dev, test1, test2, hid = {}, {}, {}, {}, {}

    for k, v in dataset.items():
        h = hash_float(k)
        if h <= train_share:
            train[k] = v
        elif h <= train_share + dev_share:
            dev[k] = v
        elif h <= train_share + dev_share + test1_share:
            test1[k] = v
        elif h <= train_share + dev_share + test1_share + test2_share:
            test2[k] = v
        else:
            hid[k] = v
    forbidden_words = {w for s in [dev, test1, test2, hid] for w in s.keys()}
    return train, dev, test1, test2, hid, forbidden_words
