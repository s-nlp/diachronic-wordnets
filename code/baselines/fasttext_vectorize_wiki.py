from vectorizers.fasttext_vectorizer import FasttextVectorizer
import json
import re
import sys


def main():
    # python3 fasttext_vectorize_wiki.py en models/cc.en.300.bin ../../datasets/wiki/wiki_ru.jsonlines \
    # models/vectors/fasttext/ru/wiki.txt

    if len(sys.argv) < 5:
        raise Exception("Required arguments: <lang: ru/en> <fasttext-model> <input-wiki-file> <out-file>")
    language = sys.argv[1]
    ft = FasttextVectorizer(sys.argv[2])
    input_filename = sys.argv[3]
    out_filename = sys.argv[4]

    if language == 'ru':
        pattern = re.compile("[^А-я \-]")
    elif language == 'en':
        pattern = re.compile("[^A-z \-]")
    else:
        raise Exception(f"language '{language}' is not supported")

    hypernyms = set()
    counter = 0
    with open(input_filename, 'r') as f:
        for line in f:
            for hypernym in json.loads(line)['hypernyms']:
                hypernym = hypernym.replace("|", " ").replace('--', '')
                hypernym = pattern.sub("", hypernym)
                if not all([i == " " for i in hypernym]):
                    hypernyms.add(hypernym.replace(" ", "_"))
            counter += 1
    print(counter)
    ft.vectorize_multiword_data(hypernyms, out_filename, to_upper=False)


if __name__ == '__main__':
    main()
