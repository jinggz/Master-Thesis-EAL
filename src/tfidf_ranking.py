import pandas as pd
import json
import os
from pathlib import Path
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import logging
import filter_sentences
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

    def get_aspects_dict(self, entity):
        return self.wiki_dict.get(entity)


def avg_precision(p, rel_tol=1e-09):
    '''
    return the moving average p@1, stop when the different of last two p@1 smaller than rel_tol
    :param: p: the list of p@1
    :param rel_tol: the relelvant tolerance between 2 precisions,(0,1)
    :type: double
    :return: average p@1
    :return: indicator of convergence
    '''
    converged = False
    p_current = sum(p)/len(p)
    if len(p)==1:
        p_last=p_current
    else:
        p_last = sum(p[:-1])/(len(p)-1)
    if len(p)>100 and abs(p_last/p_current-1)<=rel_tol: # at least 100 times training
        converged=True
    return converged, p_current

def get_tfidf(text, vectorizer):
    return vectorizer.transform(text)

def cos_sim(a, b):
    return cosine_similarity(a, b).flatten()

def get_max(aspect_list, score_list):
    '''
    :param aspect_list:
    :param score_list:
    :return: the aspect with highest cosine similarity
    '''
    return aspect_list[np.argmax(score_list)]

def get_entity_tfidf(entity,tfidf_model, wiki_dict):
    aspect_list = []
    content_v = []
    for k in wiki_dict[entity]:
        if k != 'lead_paragraphs':
            aspect_list.append(k)
            content_v.append(get_tfidf(wiki_dict[entity][k]['content'], tfidf_model))
    return aspect_list, content_v

def get_prediction(sentence, entity): #  most outside func, should return predictions
    # ranking algo should inside it.
    tfidf_file = os.environ['tfidf_file']
    vectorizer = TfIdf(tfidf_file)

    a= get_tfidf(sentence,vectorizer)
    aspects, b=get_entity_tfidf(entity,vectorizer,wiki_dict)
    cs_sim_scores = cos_sim(a,b)
    # make  seperates func for precision

    return get_max(aspects, cs_sim_scores)


def tfidf_ranking(x_train): # only for calculating precision for own needs, no need for external use
    # get clean x_train trough nlp process
    # suppose sentence is an arrary like [text, entity, aspect]
    p = []
    for idx, sentence in enumerate(x_train):
        aspect_pred = get_prediction(sentence[0], sentence[1]) # todo

        if sentence[2] == aspect_pred:
            p.append(1)
        else:
            p.append(0)
        isconverged, p_current = avg_precision(p)
        if isconverged is True:
            p_final = p_current
            logger.info('Trained {0:d} sentences.'.format(idx + 1))
            break
    else:
        _, p_final = avg_precision(p)
        logger.info('Trained all sentences.')
    return p_final



if __name__ == '__main__':
    dir = Path(__file__).parent.parent
    model_file = Path.joinpath(dir, os.getenv('tfidf_file'))
    wiki_file = Path.joinpath(dir, os.getenv('wiki_file'))
    sentence_file = Path.joinpath(dir, os.getenv('sentence_file'))

    AR = TfidfRanking(model_file) # 'model_file' should be set as an env in docker
    AR.load_train(sentence_file, wiki_file)    # this function for my own training #sentence will be clean

