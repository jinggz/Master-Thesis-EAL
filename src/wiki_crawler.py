import pickle
from pathlib import Path
import bs4 as bs
import urllib.request
import time
from data_reader import DataReader

class EntityPage:
    def __init__(self, entity):
        self.soup = self.retrieve_wiki_page(entity)
        self.page_dict = dict()

    def retrieve_wiki_page(self, entity):
        try:
            source = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + entity).read()
            soup = bs.BeautifulSoup(source, 'lxml')
        except:
            soup = []

        return soup

    def get_page_dict_4_entity(self):
        h2_list = self.get_h2_list()
        for i in range(0, len(h2_list) - 1):
            try:
                h2_id = h2_list[i].contents[0]['id']
                h2_str = str(h2_list[i].contents[0].text)
                mark = self.soup.find(id=h2_id)
                content = self.get_text_bw_2h2(mark)
                self.page_dict[h2_str] = content
            except:
                pass

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
                        result.append(str(chi.string))
            except (AttributeError, KeyError):
                pass
        result = " ".join(result)
        return result

if __name__ == '__main__':
    dir = Path(__file__).parent.parent
    input_file = 'data/data_eal.pickle'
    input_dir = Path.joinpath(dir, input_file)
    output_file = 'data/wiki.pickle'
    # load the data
    with open(Path.joinpath(dir, input_file), 'rb') as handle:
        data = pickle.load(handle, encoding='utf-8')

    entities = set(row['entity'] for row in data)
    entities.discard('None')
    # 19985 entities, among them 10410 unique entities

    entity_sect_text_dict = dict()
    for entity in entities:
        instance = EntityPage(entity)
        if instance.soup != []:
            instance.get_page_dict_4_entity()
            entity_sect_text_dict[entity] = instance.page_dict

    saver = DataReader()
    saver.save2pickle(output_file)
    # TODO save each entity page in a seperate file?