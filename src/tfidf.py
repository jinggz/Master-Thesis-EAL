import logging
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
from data_reader import DataReader
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

class Tfidf:

    def train(self, corpus, max_features=20000):
        '''
        use training corpus to train a tf-idf model and save
        :param corpus: type: list of string
        :return:
        '''
        tfidf_model = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True,
                                          sublinear_tf=True, analyzer='word', max_features=max_features,
                                          decode_error='ignore')
        idfs = tfidf_model.fit(corpus)
        logger.info('finished training tfidf model')
        # save model
        reader = DataReader()
        dir = reader.get_file_dir()
        with open(Path.joinpath(dir, 'output/wiki-idfs.pkl'), "wb") as f:
            pickle.dump(idfs, f)
        logger.info('tfidf model saved')

    def load_corpus(self):
        reader = DataReader()
        dir = reader.get_file_dir()
        with open(Path.joinpath(dir, 'output/wiki.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def build_corpus(self):
        corpus = []
        data = self.load_corpus()
        for entity in data.values():
            for section in entity:
                if section != 'lead_paragraphs':
                    corpus.append(entity[section]['content'])
        return corpus

    def load_tfidf_model(self):
        reader = DataReader()
        dir = reader.get_file_dir()
        with open(Path.joinpath(dir, 'output/wiki-idfs.pkl'), "rb") as f:
            tfidf_model = pickle.load(f)
        return tfidf_model

    def get_tfidf(self, doc):
        # Create new tfidfVectorizer with old vocabulary
        tfidf_new = self.load_tfidf_model
        # TODO cant transform directly
        X_tf1 = tfidf_new.transform(doc)
        return X_tf1

if __name__ == '__main__':
    # one time execution
    tfidf = Tfidf
    corpus = tfidf.build_corpus()
    tfidf.train(corpus)

    # TODO cosine similarity between vector of content and vector of sentence
