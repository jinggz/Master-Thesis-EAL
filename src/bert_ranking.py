# do ranking using bert model and can use either content or header
# input file: cleaned_[customer].tsv (preprocessed) or labeled_[customer].tsv (not preprocessed)

import time
import datetime
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import nlp_preprocessing
from bert_embedding import BertEmbedd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/bert_ranking_hd_50_'+os.environ['customer']+'.log',
    filemode='w')
dir = Path(__file__).parent.parent
class BertRanking:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def load_train(self, file):
        self.train_df= pd.read_csv(file, delimiter='\t', encoding='utf-8')
        # exclude lead paragraph section from aspect candidates
        logger.info("File loaded.")

    def __prepare(self):
        # exclude lead paragraph section from aspect candidates
        self.train_df.drop(self.train_df[(self.train_df.aspect == 'lead_paragraphs')].index, inplace=True, axis=0)
        self.train_df.dropna(inplace=True)
        # preprocessing the text
        # option 1: use content
        #cols = ["sentence", "aspect_content"]
        # option 2: use title
        cols = ["sentence", "aspect"]
        self.train_df[cols] = self.train_df[cols].apply(nlp_preprocessing.nlp_pipeline)
        #self.train_df.drop(self.train_df[(self.train_df.aspect_content == '')].index, inplace=True, axis=0)
        self.train_df.drop(self.train_df[(self.train_df.aspect == '')].index, inplace=True, axis=0)
        self.train_df[cols] = self.train_df[cols].apply(nlp_preprocessing.whitespace_tokenize)
        self.train_df.index = range(len(self.train_df.index))
        self.logger.info('preprocessed text successfully.')


    def predication_pipeline(self):
        ##uncomment line 51 when using unpreprocessed data
        #self.__prepare()

        # create a sequence of sentences for inputs of BertServer
        # use sentence and aspect content
        sentences_list = self.train_df.sentence.tolist()

        # 2 options
        #aspects_list = self.train_df.aspect_content.tolist()
        aspects_list = self.train_df.aspect.tolist()

        logger.info('Start embedding texts from BertServer....')
        emb_start = time.time()
        # get embeddings from BertServer
        bc = BertEmbedd()
        bc.get_connection()
        vec_sentences = bc.get_encode(sentences_list, istokenized=False)
        self.__save_vec_to_file(vec_sentences)
        #vec_sentences = self.__load_vec_from_file()
        logger.info("The shape of loaded vector: ", vec_sentences.shape)
        vec_aspects = bc.get_encode(aspects_list, istokenized=False)
        #self.__save_vec_to_file(vec_aspects)
        bc.close_connection()
        logger.info("The total embedding time used:  {}".format(datetime.timedelta(seconds=time.time()-emb_start)))

        logger.info('Fetched all')

        # calculate cosine similarity of pairs of sentence and aspect
        logger.info('Start calculating the cosine similarities...')
        rank_start=time.time()
        if vec_sentences is not None and vec_aspects is not None:
            cos_ranking = self.matrix_cosine(vec_sentences, vec_aspects)
        else:
            raise Exception("Either vector of sentences or that of aspects is None")

        self.train_df["cos_sim"] = cos_ranking.tolist()

        # calculate the precision @ 1 by comparing the label with predication
        logger.info('Evaluating using precision @ 1...')
        precision_list = self.get_precision()
        logger.info('calculating the average precision @1')
        avg_precision(precision_list)
        logger.info("The ranking time used: {}".format(datetime.timedelta(seconds=time.time()-rank_start)) )

    def get_precision(self):
        # get the max(cos sim) by grouping by sentence_idx, then by comparing with label, get the p@1
        return self.train_df.groupby(self.train_df.idx_sentence).apply(self.__precision)

    def cos_sim(self, a, b):
        return cosine_similarity(a, b).flatten()

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

    def __save_vec_to_file(self, vec):
        dir = Path(__file__).parent.parent
        file = Path.joinpath(dir, 'model', 'bertsent_' + os.environ['customer'])
        np.savez_compressed(file, a=vec)

    def __load_vec_from_file(self):
        dir = Path(__file__).parent.parent
        file = Path.joinpath(dir, 'model', 'bertsent_' + os.environ['customer']+'.npz')
        loaded =  np.load(file)
        return loaded['a']

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
    # for i in range(100,len(p_list)):
    #     ap_next = sum(p_list[:i+1]) / (i+1)
    #     ap_current = sum(p_list[:i])/i
    #     ap_last = sum(p_list[:i-1]) / (i-1)
    #     if abs(ap_last/ap_current-1)<=rel_tol and abs(ap_next/ap_current-1)<=rel_tol:
    #         map = ap_current
    #         logger.info('The AP at 1 converged at %s th samples' % i)
    #         logger.info('The AP at 1 is %.4f.' % map)
    #         break
    # else:
    #     logger.info('The AP does not converge.')
    #     logger.info('The AP at 1 is %.4f.' % ap_end)
    logger.info('The final AP at 1 is %.4f, with %d samples' % (ap_end, len(p_list)))


if __name__ == '__main__':
    if os.getenv('customer') in ['subj', 'obj', 'both_subj', 'both_obj']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both_subj' or 'both_obj'.\n""")
    dir = Path(__file__).parent.parent
    file = Path.joinpath(dir, 'trained', 'cleaned_'+os.environ['customer']+'.tsv')

    BR = BertRanking() # 'model_file' should be set as an env in docker
    BR.load_train(file)
    logger.info('start training...')
    BR.predication_pipeline()
    logger.info('end training.')

