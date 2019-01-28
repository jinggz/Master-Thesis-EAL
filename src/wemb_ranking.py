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

import filter_sentences
import nlp_preprocessing

logger = logging.getLogger('main')
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/wemb_ranking_' + os.environ['customer'] + '.log',
    filemode='w'
)

class WembRanking:
    def __init__(self, model_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.load_model(model_file)

    def load_model(self, model_file):
        #load glove300d pretrained vector
        tmp_file = get_tmpfile("temp_word2vec.txt")
        glove2word2vec(model_file, tmp_file)
        self.model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)

        self.logger.info('Glove pretrained vector model loaded')

    def load_train(self, sentence_file, wiki_file):
        with open(wiki_file, 'r', encoding='utf-8') as f:
            self.wiki_dict = json.load(f)

        self.sentences = pd.read_json(sentence_file, orient='records')
        self.logger.info('Training files loaded: %s' % sentence_file)
        # return dataframe of cleaned sentence samples
        self.logger.info('Filtering sentence samples')
        self.sentences = filter_sentences.filter_samples(self.wiki_dict, self.sentences)
        self.logger.info('The total number of trained sentences is %s' % len(self.sentences))

    def predication_pipeline(self):
        p_list = self.sentences.apply(self.row_iter, axis=1).tolist()
        self.logger.info('calculating the average precision @1')
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

    def get_textembedding(self, text_list):
        '''
        :param text:
        :type: list of str
        :return: sparse matrix
        '''
        text_list = nlp_preprocessing.nlp_pipeline(text_list)
        res = []
        for text in text_list:
            res.append(self.sentemb(text))
        return res

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

    def get_aspects_vect(self, entity):
        y_aspects = []
        y_content = []
        for k, v in self.get_aspects_dict(entity).items():
            if k != 'lead_paragraphs':
                y_aspects.append(k)
                y_content.append(v['content'])
        y_feature = self.get_textembedding(y_content)
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
        x_feature = self.get_textembedding([sentence])
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
    if os.getenv('customer') in ['subj', 'obj', 'both']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both'.\n""")
    dir = Path(__file__).parent.parent
    model_file = Path.joinpath(dir, 'model', 'glove.6B.300d.txt')
    wiki_file = Path.joinpath(dir, 'trained', 'wiki_'+os.environ['customer']+'.json')
    sentence_file = Path.joinpath(dir, 'trained', 'sentences_'+os.environ['customer']+'.json')

    AR = WembRanking(model_file) # 'model_file' should be set as an env in docker
    AR.load_train(sentence_file, wiki_file)
    logger.info('start training...')
    AR.predication_pipeline()
    logger.info('end training.')

