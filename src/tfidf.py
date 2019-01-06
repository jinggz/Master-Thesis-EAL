import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
import json

import nlp_preprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class Tfidf:

    def __init__(self, logger):
        self.vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=True,
                                          sublinear_tf=True, decode_error='ignore')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def train(self, docs):
        '''
        use training corpus to train a tf-idf model and save
        :param docs: type: list of str
        :return:tfidf model after fit,
        :type: TfidfVectorizer object
        '''
        nlp_preprocessing.nlp_pipeline(docs)
        self.vectorizer.fit(docs)
        self.logger.info('finished training tfidf model')

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
            doc = entity[section]['content'].lower()
            docs.append(doc)
    return docs

if __name__ == '__main__':

    os.environ['customer'] = 'subj'
    if os.getenv('customer') in ['subj', 'obj', 'both']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both'.\n""")

    dir = Path(__file__).parent.parent

    docs = retrieve_docs(Path.joinpath(dir, 'trained', 'wiki_'+os.environ['customer']+'.json'))

    logger.info('Starting tfidf training for wikipedia corpus')
    tfidf_trainer = Tfidf(logger)
    tfidf_trainer.train(docs)
    # save to file
    tfidf_trainer.save_tfidf_model(Path.joinpath(dir, 'model', 'tfidf_'+os.environ['customer']+'.pkl'))
    logger.info('Saved tf-idf trained model')


