import os
from pathlib import Path
import bs4 as bs
import urllib.request
from urllib.parse import quote
import logging
import json
import re
import html
from pprint import pprint
import unicodedata


class EntityPage:
    def __init__(self, entity):
        '''
        :param page_dict: the dictionary of aspects crawled from wikipedia
        :param soup: a Beautifulsoup object
        :param entity: a string represented an entity
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.page_dict = dict()
        self.soup = None
        self.retrieve_wiki_page(entity)

    def __remove_special(self, s):
        s = html.unescape(s)
        s = unicodedata.normalize("NFKD", s)
        return s

    def __get_connection(self, entity):
        entity = entity.replace(' ', '_')
        try:
            # example: https://en.wikipedia.org/wiki/Segal%E2%80%93Bargmann_space
            connection = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + quote(entity))
        except Exception as e:
            self.logger.exception(e, exc_info=False)
            return []
        return connection

    def retrieve_wiki_page(self, entity):
        # To connect and extract the page content of the Wiki Page of the entity
        connection = self.__get_connection(entity)
        if connection:
            source = connection.read()
            soup = bs.BeautifulSoup(source, 'lxml')
            # remove unused tags
            [x.extract() for x in soup.find_all('sup')]
            [x.extract() for x in soup.find_all(class_='mw-editsection')]
            [x.extract() for x in soup.find_all(class_='image')]
            self.soup = soup

    def build_page_dict(self):
        '''
        take an entity, crawl the corresponding page from Wikipedia, and build an dictionary of aspects
        :return: page_dict: the dictionary of aspects crawled from wikipedia
        '''
        self.__get_lead_pa()
        # extract h2 headings and its content, entities and sub headings
        h2_list = self.__get_h2_list()
        for h2 in h2_list:
            nextNode = h2.next_sibling
            p = []
            links = []
            heads = []
            while nextNode and nextNode.name != 'h2':
                if not isinstance(nextNode, bs.element.NavigableString):
                    if nextNode.name == 'p':
                        p.append(self.__remove_special(nextNode.text.strip()))
                    for link in nextNode.find_all('a', href=self.__not_file):
                        links.append(urllib.parse.unquote((link.get('href'))))
                    # get sub heads within a h2 head
                    if nextNode.name in ['h3', 'h4', 'h5', 'h6']:
                        p.append(self.__remove_special(nextNode.text.strip()))
                        heads.append(self.__remove_special(nextNode.text.strip().lower()))
                nextNode = nextNode.next_sibling
            links = " ".join(links)
            contents = " ".join(p)
            self.page_dict[h2.text.strip().lower()] = {'content': contents, 'links': links, 'heads': heads}

    def __get_lead_pa(self):
        # extract the lead paragraphs before the first heading
        body = self.soup.find(id='bodyContent')
        start = body.find(class_='mw-parser-output')
        leading = []
        for chi in start.children:
            if chi.name == 'p':
                leading.append(self.__remove_special(chi.text.strip()))
            if chi.name == 'h2':
                break
        leading = " ".join(leading)
        self.page_dict['lead_paragraphs'] = {'content': leading}

    def __get_h2_list(self):
        # http://trec-car.cs.unh.edu/process/dataselection.html
        # extract all h2 headings of a Wiki page and remove frequent headings like 'see also' etc.
        h_remove = ["see also", "contents", "notes and references",
                    "references",
                    "external links",
                    "notes",
                    "bibliography",
                    "gallery",
                    "publications",
                    "further reading",
                    "track listing",
                    "sources",
                    "cast",
                    "discography",
                    "awards",
                    "other"]
        body = self.soup.find(id='bodyContent')
        h2_set = []
        for h2 in body.find_all('h2'):
            if not h2.text.strip().lower() in h_remove:
                h2_set.append(h2)
        return h2_set

    def __not_file(self, href):
        return href and not re.compile("File").search(href) and re.compile("wiki").search(href)

def build_dict_training():
    logger = logging.getLogger('build log')
    dir = Path(__file__).parent.parent
    input_file = Path.joinpath(dir, 'trained', 'sentences_'+os.environ['customer']+'.json')
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
    wiki_dict = dict()
    logger.info('Start to create dictionary for Wiki pages... ')

    for entity in entities:
        instance = EntityPage(entity)
        if instance.soup:
            instance.build_page_dict()
        wiki_dict[entity.replace('_', ' ').lower()] = instance.page_dict
    logger.info('Finished the creation for dictionary of Wikipedia pages')

    with open(Path.joinpath(dir, 'trained', 'wiki_'+os.environ['customer']+'.json'), 'w', encoding='utf-8') as outfile:
        json.dump(wiki_dict, outfile)
    logger.info('Saved the dictionary of entities to %s' % outfile)

if __name__ == '__main__':

    while True:
        term = input('enter entity: ')
        EP = EntityPage(term)
        if EP.soup:
            EP.build_page_dict()
        pprint(EP.page_dict)


