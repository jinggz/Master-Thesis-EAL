import pickle
from pathlib import Path
import bs4 as bs
import urllib.request
from urllib.parse import quote
import time
import pprint as pp
import logging
import json
import re

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

    def get_connection(self, entity):
        try:
            # example: https://en.wikipedia.org/wiki/Segal%E2%80%93Bargmann_space
            connection = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + quote(entity))
        except Exception as e:
            logger.exception(e, exc_info=False)
            return []
        return connection

    def retrieve_wiki_page(self, entity):
        # To connect and extract the page content of the Wiki Page of the entity
        connection = self.get_connection(entity)
        if connection:
            source = connection.read()
            soup = bs.BeautifulSoup(source, 'lxml')
        # remove unused tags
        [x.extract() for x in soup.find_all('sup')]
        [x.extract() for x in soup.find_all(class_='mw-editsection')]
        [x.extract() for x in soup.find_all(class_='image')]
        self.soup = soup

    def get_page_dict_4_entity(self):
        h2_list = self.get_h2_list()
        for i in range(0, len(h2_list) - 1):
            try:
                h2_id = h2_list[i].contents[0]['id']
                h2_str = str(h2_list[i].contents[0].text)
                mark = self.soup.find(id=h2_id)
                content = self.get_text_bw_2h2(mark)
                self.page_dict[h2_str] = content
            except Exception as e:
                logger.info(e)

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
        contents = []
        links = []
        for elt in mark.parent.next_siblings:
            if elt.name == "h2":
                break
            try:
                for link in elt.find_all('a', href=self.__not_file):
                    links.append(urllib.parse.unquote((link.get('href'))))
                contents.append(elt.get_text(' ', strip=True))
            except Exception as e:
                logger.exception(e, exc_info=False)
        links = " ".join(links)
        contents = " ".join(contents)
        return contents, links

    def __not_file(self, href):
        return href and not re.compile("File").search(href)


if __name__ == '__main__':

    dir = Path(__file__).parent.parent
    input_file = Path.joinpath(dir, 'output/sentences_eal_subj.json')
    #output_file = Path.joinpath(dir, 'output/entity_dict/')

    # load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f, encoding='utf-8')
    logger.info('Sentence data loaded')

    # load list of entities
    entities = set(row['entity'] for row in data)
    entities.discard('None')
    # 19985 entities, among them 10410 unique entities
    logger.info('Remove empty entity')

    # begin crawling
    entity_sect_text_dict = dict()
    logger.info('Start to create dictionary for Wiki page... ')
    for entity in entities:
        instance = EntityPage()
        instance.retrieve_wiki_page(entity)
        instance.get_page_dict_4_entity()
        #TODO save to seperate json file
        #         # #saver.save2json(file, instance.page_dict)
        #entity_sect_text_dict[entity] = instance.page_dict

    #         #logger.info('Saved the dictionary of %s to %s.' % (entity, output_file) )
    logger.info('Finished dictionary creation.')

    # pp.pprint(entity_sect_text_dict['Cello Suites (Bach)'])
    # saver = DataReader()
    # saver.save2json(Path.joinpath(dir, 'output/entity_dicts.json'), entity_sect_text_dict)
    # logger.info('Saved the dictionary of entities to file.')


