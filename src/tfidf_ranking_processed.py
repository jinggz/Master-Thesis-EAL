# this file is to run tfidf ranking using preprocessed data to save time (cleaned tsv)
# input file: cleaned_[customer].tsv

import pandas as pd
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import logging


logger = logging.getLogger('main')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/tfidf_ranking_header_'+os.environ['customer']+'.log',
    filemode='w')

class TfidfRanking:
    def __init__(self, model_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.load_model(model_file)

    def load_model(self, model_file):
        #load tfidf model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.logger.info('started TF-IDF ranking with model file: %s' % model_file)

    def load_train(self, file):
        self.train_df= pd.read_csv(file, delimiter='\t', encoding='utf-8')
        # exclude lead paragraph section from aspect candidates
        logger.info("File loaded.")

    def predication_pipeline(self):
        # p_list = self.sentences.apply(self.row_iter, axis=1).tolist()
        # self.logger.info('calculating the average precision @1')
        # avg_precision(p_list)

        # create a sequence of sentences for getting tfidf vector
        # use sentence and aspect header

        sentences_list = self.train_df.sentence.tolist()
        aspects_list = self.train_df.aspect.tolist()
        cos_ranking=[]
        for sentence, aspect in zip(sentences_list, aspects_list):
            vec_sentence = self.get_tfidf([sentence])
            vec_aspect = self.get_tfidf(aspect)
            if vec_sentence is not None and vec_aspect is not None:
                # cos_ranking = self.matrix_cosine(vec_sentences, vec_aspects)
                cos_ranking.extend(self.cos_sim(vec_sentence, vec_aspect).tolist())

        self.logger.info('Start transform text to tdidf vector....')
        # vec_sentences = self.get_tfidf(sentences_list)
        # vec_aspects = self.get_tfidf(aspects_list)

        # calculate cosine similarity of pairs of sentence and aspect
        self.logger.info('Start calculating the cosine similarities...')
        # if vec_sentences is not None and vec_aspects is not None:
        #     #cos_ranking = self.matrix_cosine(vec_sentences, vec_aspects)
        #     cos_ranking = self.cos_sim(vec_sentences, vec_aspects).tolist()
        #     cos_ranking.add(cos_ranking)

        # else:
        #     raise Exception("Either vector of sentences or that of aspects is None")

        self.train_df["cos_sim"] = cos_ranking

        # calculate the precision @ 1 by comparing the label with predication
        logger.info('Evaluating using precision @ 1...')
        precision_list = self.get_precision()
        self.logger.info('calculating the average precision @1')
        avg_precision(precision_list)

    def get_precision(self):
        # get the max(cos sim) by grouping by sentence_idx, then by comparing with label, get the p@1
        return self.train_df.groupby(self.train_df.idx_sentence).apply(self.__precision)

    def matrix_cosine(self, x, y):
        # calculate cos_sim between matching rows
        # 2 ndarray must have same #samples
        # return: (#samples,)
        assert x.shape == y.shape, "The shape of x, y must be equal"
        return np.einsum('ij,ij->i', x, y) / (
                np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        )

    def __precision(self, x):
        '''
        for each sentence group, get the index of predication then return the precision at 1
        '''

        if x.loc[x["cos_sim"].idxmax(), "label"] == 1:
            y_pred = 1
        else:
            y_pred = 0
        return y_pred

    def get_aspects_dict(self, entity):
        s = entity.replace('_', ' ').lower()
        return self.wiki_dict.get(s)

    def get_tfidf(self, text):
        '''
        :param text:
        :type: list of str
        :return: sparse matrix
        '''
        #text = nlp_preprocessing.nlp_pipeline(text)
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

        # get aspect dict and aspects vector
        try:
            y_aspects, y_feature = self.get_aspects_vect(entity)
        except ValueError as error:
            self.logger.error(error)
            self.logger.error("The page of the entity contains no proper aspect other than lead section.")
            return "summary"
        # sentence vector
        x_feature = self.get_tfidf([sentence])
        self.logger.debug("calculating the most relevant aspect...")
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
    ap_end = sum(p_list) / len(p_list)
    for i in range(100,len(p_list)):
        ap_next = sum(p_list[:i+1]) / (i+1)
        ap_current = sum(p_list[:i])/i
        ap_last = sum(p_list[:i-1]) / (i-1)
        if abs(ap_last/ap_current-1)<=rel_tol and abs(ap_next/ap_current-1)<=rel_tol:
            map = ap_current
            logger.info('The AP at 1 converged at %s th samples' % i)
            logger.info('The AP at 1 is %.4f.' % map)
            break
    else:
        logger.info('The AP does not converge.')
        logger.info('The AP at 1 is %.4f.' % ap_end)
    logger.info('The final AP at 1 is %.4f.' % ap_end)


if __name__ == '__main__':
    if os.getenv('customer') in ['subj', 'obj', 'both_subj', 'both_obj']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both'.\n""")
    dir = Path(__file__).parent.parent
    model_file = Path.joinpath(dir, 'model', 'tfidf_'+os.environ['customer']+'.pkl')

    train_file = Path.joinpath(dir, 'trained', 'cleaned_'+os.environ['customer']+'.tsv')

    AR = TfidfRanking(model_file) # 'model_file' should be set as an env in docker
    AR.load_train(train_file)    # this function for my own training #sentence will be clean
    logger.info('start ranking...')
    AR.predication_pipeline()
    logger.info('end ranking.')

