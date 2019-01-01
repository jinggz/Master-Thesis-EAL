import logging
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
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

    def __init__(self, logger):
        self.vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True,
                                          sublinear_tf=True, analyzer='word',
                                          decode_error='ignore')
        self.logger = logger

    def train(self, docs):
        '''
        use training corpus to train a tf-idf model and save
        :param docs: type: list of string
        :return:tfidf model after fit,
        :type: TfidfVectorizer object
        '''
        self.vectorizer.fit(docs)
        logger.info('finished training tfidf model')

    def save_tfidf_model(self, model_path):
        # save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

def retrieve_docs(data_path):
    # each aspect is regarded as a doc
    logger.info("Starting docs retrieval")
    docs = list()
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entity in data.values():
        for section in entity:
            if section != 'lead_paragraphs':
                doc = entity[section]['content'].lower()
                docs.append(doc)
    return docs

if __name__ == '__main__':

    dir = Path(__file__).parent.parent

    docs = retrieve_docs(Path.joinpath(dir, 'trained', 'wiki_subj.json'))

    logger.info('Starting tfidf training for wikipedia corpus')
    tfidf_trainer = Tfidf(logger)
    tfidf_trainer.train(docs)
    # save to file
    tfidf_trainer.save_tfidf_model(Path.joinpath(dir, 'model', 'tfidf_subj.pkl'))
    logger.info('Saved tf-idf trained model')


