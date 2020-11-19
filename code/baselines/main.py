import sys
import json
import codecs

from nltk.corpus import WordNetCorpusReader
from ruwordnet.ruwordnet_reader import RuWordnet
from predict_models import BaselineModel, HCHModel, RankedModel, LRModel


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def generate_taxonomy_fns(params, model):
    # for English WordNet
    if params['language'] == 'en':
        wn = WordNetCorpusReader(params["wordnet_path"], None)
        return lambda x: [hypernym.name() for hypernym in wn.synset(x).hypernyms()
                          if hypernym.name() in model.w2v_synsets.vocab], \
               lambda x: [hyponym.name() for hyponym in wn.synset(x).hyponyms() if hyponym.name()
                          in model.w2v_synsets.vocab], \
               lambda x: x.split(".")[0].replace("_", " ")
    # for RuWordNet
    elif params['language'] == 'ru':
        ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["wordnet_path"])
        return lambda x: ruwordnet.get_hypernyms_by_id(x), lambda x: ruwordnet.get_hyponyms_by_id(x), \
               lambda x: ruwordnet.get_name_by_id(x)
    else:
        raise Exception("task / language is not supported")


def save_to_file(words_with_hypernyms, output_path, _params):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            for hypernym in hypernyms:
                f.write(f"{word}\t{hypernym}\n")


def main():
    models = {"simple": BaselineModel, "baseline": HCHModel, "ranked": RankedModel, "lr": LRModel}
    params = load_config()

    with open(params['test_path'], 'r', encoding='utf-8') as f:
        test_data = f.read().split("\n")[:-1]

    model = models[params["model"]](params)
    print("Model loaded")

    topn = params["topn"] if "topn" in params else 10
    results = model.predict_hypernyms(list(test_data), *generate_taxonomy_fns(params, model), topn)
    save_to_file(results, params['output_path'], params)


if __name__ == '__main__':
    main()
