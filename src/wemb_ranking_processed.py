import pandas as pd
import json
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

logger = logging.getLogger('main')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/wemb_ranking_header_'+os.environ['customer']+'.log',
    filemode='w')

class WembRanking:
    def __init__(self, model_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.load_model(model_file)

    def load_model(self, model_file):
        # load glove300d pretrained vector
        tmp_file = get_tmpfile("temp_word2vec.txt")
        glove2word2vec(model_file, tmp_file)
        self.model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)

        self.logger.info('Glove pretrained vector model loaded')

    def load_train(self, file):
        self.train_df= pd.read_csv(file, delimiter='\t', encoding='utf-8')
        logger.info("File loaded.")

    def predication_pipeline(self):

        # create a sequence of sentences for getting wemb vector
        # use sentence and aspect header

        sentences_list = self.train_df.sentence.tolist()
        aspects_list = self.train_df.aspect.tolist()
        cos_ranking=[]
        self.logger.info('Start transform text to word embedding vector and calculating the cs....')
        for sentence, aspect in zip(sentences_list, aspects_list):
            vec_sentence = self.sentemb(sentence)
            vec_aspect = self.sentemb(aspect)
            if vec_sentence is not None and vec_aspect is not None:
                # cos_ranking = self.matrix_cosine(vec_sentences, vec_aspects)
                vec_sentence = np.reshape(vec_sentence, (1, -1))
                vec_aspect = np.reshape(vec_aspect, (1, -1))
                cos_ranking.extend(self.cos_sim(vec_sentence, vec_aspect).tolist())

            else:
                raise Exception("Either vector of sentences or that of aspects is None")
        print(len(cos_ranking))
        self.train_df["cos_sim"] = cos_ranking

        # calculate the precision @ 1 by comparing the label with predication
        logger.info('Evaluating using precision @ 1...')
        precision_list = self.get_precision()
        self.logger.info('calculating the average precision @1')
        avg_precision(precision_list)

    def sentemb(self, sentence):
        '''
        return aggregate average embedding vector for a sentence
        :param sentence: a long string
        :type:str
        :return:
        '''
        words = sentence.split()
        article_embedd = []
        for word in words:
            try:
                embed_word = self.model[word]
                article_embedd.append(embed_word)
            except KeyError:
                continue
        if article_embedd == []:
            return np.zeros(300)
        else:
            article_embedd = np.asarray(article_embedd)
            avg = np.average(article_embedd, axis=0)
        return avg

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

    def cos_sim(self, a, b):
        return cosine_similarity(a, b).flatten()

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
        Your options are: customer='subj' or 'obj' or 'both_subj' or 'both_obj.\n""")
    dir = Path(__file__).parent.parent
    model_file = Path.joinpath(dir, 'model', 'glove.6B.300d.txt')

    train_file = Path.joinpath(dir, 'trained', 'cleaned_'+os.environ['customer']+'.tsv')

    AR = WembRanking(model_file) # 'model_file' should be set as an env in docker
    AR.load_train(train_file)    # this function for my own training #sentence will be clean
    logger.info('start ranking...')
    AR.predication_pipeline()
    logger.info('end ranking.')

