import numpy as np
from gensim.models.fasttext import load_facebook_model
from string import punctuation


class FasttextVectorizer:
    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)
        print('Model loaded')

    # -------------------------------------------------------------
    # vectorize ruwordnet
    # -------------------------------------------------------------

    def vectorize_groups(self, synsets, output_path, to_upper=True):
        ids, vectors = self.__get_ruwordnet_vectors(synsets)
        self.save_as_w2v(ids, vectors, output_path, to_upper)

    def __get_ruwordnet_vectors(self, synsets):
        ids = []
        vectors = np.zeros((len(synsets), self.model.vector_size))
        for i, (_id, texts) in enumerate(synsets.items()):
            ids.append(_id)
            vectors[i, :] = self.__get_avg_vector(texts)
        return ids, vectors

    def __get_avg_vector(self, texts):
        sum_vector = np.zeros(self.model.vector_size)
        for text in texts:
            words = [i.strip(punctuation) for i in text.split()]
            sum_vector += np.sum(self.__get_data_vectors(words), axis=0)/len(words)
        return sum_vector/len(texts)

    # -------------------------------------------------------------
    # vectorize data
    # -------------------------------------------------------------

    def vectorize_words(self, data, output_path, to_upper=True):
        data_vectors = self.__get_data_vectors(data)
        self.save_as_w2v(data, data_vectors, output_path, to_upper)

    def __get_data_vectors(self, data):
        vectors = np.zeros((len(data), self.model.vector_size))
        for i, word in enumerate(data):
            vectors[i, :] = self.model[word]
        return vectors

    # -------------------------------------------------------------
    # vectorize multi-word data
    # -------------------------------------------------------------

    def vectorize_multiword_data(self, data, output_path, to_upper=True):
        data_vectors = self.get_multiword_vectors(data)
        self.save_as_w2v(data, data_vectors, output_path, to_upper)

    def get_multiword_vectors(self, data):
        vectors = np.zeros((len(data), self.model.vector_size))
        for i, multi_word in enumerate(data):
            words = multi_word.replace("_", " ").split()
            vectors[i, :] = np.sum(self.__get_data_vectors(words), axis=0)/len(words)
        return vectors

    # -------------------------------------------------------------
    # save
    # -------------------------------------------------------------

    @staticmethod
    def save_as_w2v(words: list, vectors: np.array, output_path: str, to_upper):
        assert len(words) == len(vectors)
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
            for word, vector in zip(words, vectors):
                vector_line = " ".join(map(str, vector))
                word = word.upper() if to_upper else word
                w.write(f"{word} {vector_line}\n")
