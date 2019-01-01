import pandas as pd
import json
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import logging
import filter_sentences

import nlp_preprocessing
import set_env

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

class TfidfRanking:
    def __init__(self, model_file):
        self.load_model(model_file)

    def load_model(self, model_file):
        #load tfidf model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        logger.info('started TF-IDF ranking with model file: %s' % model_file)

    def load_train(self, sentence_file, wiki_file):
        with open(wiki_file, 'r', encoding='utf-8') as f:
            self.wiki_dict = json.load(f)

        self.sentences = pd.read_json(sentence_file, orient='records')
        logger.info('Training files loaded')
        # return dataframe of cleaned sentence samples
        logger.info('Filtering sentence samples')
        self.sentences = filter_sentences.filter_samples(self.wiki_dict, self.sentences)
        logger.info('The total number of trained sentences is %s' % len(self.sentences))

    def training(self):
        p_list = self.sentences.apply(self.row_iter, axis=1).tolist()
        logger.info('calculating the average precision @1')
        avg_precision(p_list)

    def row_iter(self, row):
        aspect_pred = self.get_prediction(row['sentence'], row['entity'])
        if aspect_pred == row['aspect']:
            p = 1
        else:
            p = 0
        return p

    def get_aspects_dict(self, entity):
        s = entity.replace('_', ' ').lower()
        return self.wiki_dict.get(s)

    def get_tfidf(self, text):
        '''
        :param text:
        :type: list of str
        :return: sparse matrix
        '''
        nlp_preprocessing.nlp_pipeline(text)
        return self.model.transform(text)

    def get_aspects_vect(self, entity):
        y_aspects = []
        y_content = []
        for k, v in self.get_aspects_dict(entity).items():
            if k != 'lead_paragraphs':
                y_aspects.append(k)
                y_content.append(v['content'])
        y_feature = self.get_tfidf(y_content)
        return y_aspects, y_feature

    def cos_sim(self, a, b):
        return cosine_similarity(a, b).flatten()

    def get_prediction(self, sentence, entity):
        '''
        return the closest aspect of a given entity appeared in a given sentence
        :param sentence: a given sentence containing a representation of an entity
        :type: str
        :param entity: a given entity identified in the sentence
        :type: str
        :return: the closet aspect found in the Wikipedia page of the given entity
        :type: str
        '''

        # sentence vector
        x_feature = self.get_tfidf([sentence])
        # get aspect dict and aspects vector
        y_aspects, y_feature = self.get_aspects_vect(entity)
        cos_ranking = self.cos_sim(x_feature, y_feature)
        y_pred = y_aspects[np.argmax(cos_ranking)]
        return y_pred


def avg_precision(p_list, rel_tol=1e-03):
    '''
    return the moving average p@1, stop when the different of last two p@1 smaller than rel_tol
    :param: p: the list of p@1
    :param rel_tol: the relelvant tolerance between 2 precisions,(0,1)
    :type: double
    :return: average p@1
    :return: indicator of convergence
    '''
    for i in range(100,len(p_list)):
        ap_next = sum(p_list[:i+1]) / (i+1)
        ap_current = sum(p_list[:i])/i
        ap_last = sum(p_list[:i-1]) / (i-1)
        if abs(ap_last/ap_current-1)<=rel_tol and abs(ap_next/ap_current-1)<=rel_tol:
            map = ap_current
            logger.info('The AP at 1 converged at %s th samples' % i)
            logger.info('The AP at 1 is %s.' % map)
            break
    else:
        ap_end = sum(p_list) / len(p_list)
        logger.info('The AP does not converge.')
        logger.info('The AP at 1 is %s.' % ap_end)



if __name__ == '__main__':
    os.environ['customer'] = 'subj'
    if os.getenv('customer') in ['subj', 'obj', 'both']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both'.\n""")
    dir = Path(__file__).parent.parent
    model_file = Path.joinpath(dir, 'model', 'tfidf_'+os.environ['customer']+'.pkl')
    wiki_file = Path.joinpath(dir, 'trained', 'wiki_'+os.environ['customer']+'.json')
    sentence_file = Path.joinpath(dir, 'trained', 'sentences_'+os.environ['customer']+'.json')

    AR = TfidfRanking(model_file) # 'model_file' should be set as an env in docker
    AR.load_train(sentence_file, wiki_file)    # this function for my own training #sentence will be clean
    logger.info('start training...')
    AR.training()
    logger.info('end training.')

