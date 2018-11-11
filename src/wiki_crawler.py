import pickle
from pathlib import Path
import bs4 as bs
import urllib.request
import time
import pprint as pp
import logging
import json

from data_reader import DataReader

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

class EntityPage:
    def __init__(self):
        self.page_dict = dict()
        self.soup = None

    def retrieve_wiki_page(self, entity):
        # To connect and extract the page content of the Wiki Page of the entity
        err_log = ''
        try:
            source = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + entity).read()
            soup = bs.BeautifulSoup(source, 'lxml')
        except Exception as exception:
            err_log = "No page of [%s]: [%s] %s" % (entity, exception, exception.url)
            soup = []

        self.soup = soup
        return err_log

    def get_page_dict_4_entity(self):
        h2_list = self.get_h2_list()
        for i in range(0, len(h2_list) - 1):
            try:
                h2_id = h2_list[i].contents[0]['id']
                h2_str = str(h2_list[i].contents[0].text)
                mark = self.soup.find(id=h2_id)
                content = self.get_text_bw_2h2(mark)
                self.page_dict[h2_str] = content
            except Exception as exception:
                logger.info(exception)
                pass

    #  TODO collect contents b4 the first h2 -- summary
    def get_h2_list(self):
        p_set = []
        for paragraph in self.soup.find_all('h2'):
            p_set.append(paragraph)
        return p_set

    def get_text_bw_2h2(self, mark):
        '''
        get the content between 2 sections
        :param: mark: the start section name. typ: str
        :return: the whole text between 2 sections as a string
        '''
        result = []
        for elt in mark.parent.next_siblings:
            if elt.name == "h2":
                break
            try:
                for chi in elt.children:
                    if chi.name == 'sup':
                        continue
                    else:
                        result.append(str(chi))
            except (AttributeError, KeyError):
                pass
        result = " ".join(result)
        return result

if __name__ == '__main__':
    dir = Path(__file__).parent.parent

    input_file = Path.joinpath(dir, 'data/data_eal.json')
    output_file = 'data/wiki.pickle'

    # load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info('Sentence data loaded')

    entities = set(row['entity'] for row in data)
    entities.discard('None')
    # 19985 entities, among them 10410 unique entities
    logger.info('Remove empty entity')
    # TODO remove entity_sect_text_dict
    entity_sect_text_dict = dict()
    logger.info('Start to create dictionary for Wiki page... ')
    for entity in entities:
        instance = EntityPage()
        err = instance.retrieve_wiki_page(entity)
        if err:
            logger.info(err)
        else:
            instance.get_page_dict_4_entity()
            entity_sect_text_dict[entity] = instance.page_dict
            # TODO save to seperate json file
            # logger.info('Save the dictionary of %s to %s.' % (entity, output_file) )

    # pp.pprint(entity_sect_text_dict['Cello Suites (Bach)'])
    # saver = DataReader()
    # saver.save2pickle(output_file)
    # TODO save each entity page in a seperate file

