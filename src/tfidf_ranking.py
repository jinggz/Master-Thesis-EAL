from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import logging

import nlp_preprocessing
from wiki_crawler import EntityPage

main_logger = logging.getLogger('main')

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

    def get_aspects_dict(self, entity):
        return None

    def get_tfidf(self, text):
        '''
        :param text:
        :type: list of str
        :return: sparse matrix
        '''
        text = nlp_preprocessing.nlp_pipeline(text)
        return self.model.transform(text)

    def get_aspects_vect(self, entity):
        y_aspects = []
        y_content = []
        for k, v in self.get_aspects_dict(entity).items():
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
            self.logger.error("The page of the entity contains no proper aspect.")
            return "", 0
        # sentence vector
        x_feature = self.get_tfidf([sentence])
        self.logger.info("calculating the most relevant aspect...")
        cos_ranking = self.cos_sim(x_feature, y_feature)
        if len(cos_ranking) == 1: # only contain 1 aspect (summary)
            y_pred = "summary"
            cos_sim = cos_ranking[0]
        else:
            ind = np.argpartition(cos_ranking, -2)[-2:]  # get indexes of 2 biggest values
            if ind[1] != 0: # if largest score is not 'summary'
                y_pred = y_aspects[ind[1]]
                cos_sim = cos_ranking[ind[1]]
            else:
                y_pred = y_aspects[ind[0]]
                cos_sim = cos_ranking[ind[0]]
        return y_pred, cos_sim

class EAL(TfidfRanking):
    def get_aspects_dict(self, entity):
        # adapt to external use
        entity = entity.strip()
        EP = EntityPage(entity)
        self.logger.info('Connected to Wikipedia')
        if EP.soup:
            EP.build_page_dict()
        self.logger.info('Built dictionary of entity aspects...')
        return EP.page_dict

if __name__ == '__main__':

    dir = Path(__file__).parent.parent
    # set 'model_file' to your own path
    model_file = Path.joinpath(dir, 'model', 'tfidf_model.pkl')
    while True:
        sentence = input('enter sentence:')
        entity = input('enter entity: ')
        logging.info('start training...')
        ranking = EAL(model_file)
        aspect_predicted, score = ranking.get_prediction(sentence, entity)
        logging.info('end training.')
        print('Predicted most relevant aspect is: %s(score: %.4f)' % (aspect_predicted,score))


