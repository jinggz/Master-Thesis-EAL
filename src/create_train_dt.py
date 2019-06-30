#this code is to create eal data
#input: sentence data and wiki data
#output: context_[customer].json

import os
import re
from pathlib import Path
import json
import logging
import pandas as pd
import ahocorasick
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

def load_file(wiki_file, sentence_file, locale):
    with open(wiki_file, 'r', encoding='utf-8') as f:
        wiki_dict = json.load(f)

    sentences = pd.read_json(sentence_file, orient='records')
    # return dataframe of cleaned sentence samples
    logger.info('Filtering sentence samples')
    # do filtering for sentences dataset
    sentences = filter_samples(wiki_dict, sentences, ["aspect", "entity", "sentence"])
    return wiki_dict, sentences

def form_dict(wiki_dict, sentences, locale):
    train_dt = dict()
    context_sent = sentences.apply(row_iter, axis=1, locale=locale)
    for i, sent in tqdm(enumerate(context_sent)):
        idx = str(i)+"_"+locale
        entity = sent["entity_mention"]["entity"]
        aspect_candidates = extract_wiki(entity, wiki_dict)
        sent["aspect_candidates"] = aspect_candidates
        train_dt[idx] = sent

    return train_dt

def row_iter(row, locale):
    sent_context = row['sentence']
    true = row['aspect']
    if locale in ['subj', 'both_subj']:
        mention = row["triple"][0]
    elif locale in ['obj', 'both_obj']:
        mention = row["triple"][2]
    entity_mention = {"entity": row['entity'], "mention": mention}

    return {"entity_mention": entity_mention, "true": true, "sent_context": sent_context}

def extract_wiki(entity, wiki_dict):
    candidates = []
    for k, v in get_aspects_dict(entity, wiki_dict).items():
        id_aspect = k
        content = v['content']
        header = id_aspect
        entities = extract_link(v.get('links'))
        candidates.append({"id_aspect": id_aspect, "content": content, "header":header, "entities":entities})
    return candidates

def extract_link(links):
    if links is None or links == '':
        return []
    e_list=[]
    lst = ['none', 'list of', 'lists of', 'disambiguation', 'category', 'categories', 'gaa', 'cup', 'season',
           'champions', 'league']
    pattern = r'\/wiki\/'
    for link in links.split(" "):
        if re.match(pattern, link) is None:
            continue
        link = link.split('/')[2].lower().replace("_", " ")
        if link.startswith('help') == -1 or link.startswith('wikipedia') == -1:
            continue
        for sub in lst:
            if link.find(sub) != -1:
                break
        else:
            e_list.append(link)
    return e_list

def get_aspects_dict(entity, wiki_dict):
    s = entity.replace('_', ' ').lower()
    return wiki_dict.get(s)

def filter_samples(wiki_dict, df, subcol):
    '''
    :param
    wiki_dict: the dictionary of entity-aspects pages
    :type: dict
    sentences: the raw sentence tuples extracted from Kiril's data
    :type: dataframe
    :return: sentences: the filtered sentences samples for training
    :type: dataframe
    '''

    df.drop_duplicates(subset=subcol, inplace=True)
    df.drop(df[(df.entity == 'None') | (df.aspect == 'None')].index, inplace=True)
    df.index = range(len(df.index))
    df.aspect = df.aspect.str.lower()
    df.entity = df.entity.str.lower()
    df.aspect = df.aspect.str.replace("_", " ")
    df.entity = df.entity.str.replace("_", " ")
    lst = ['none', 'list of', 'lists of', 'disambiguation', 'category', 'categories', 'gaa', 'cup', 'season',
           'champions', 'league']
    df = df[using_ahocorasick(df.entity, lst) == False] # filter out entity contain substr in lst
    df.index = range(len(df.index))

    # TODO future: refactor the for loop to df[df.apply(row_iter, axis=1)!=0]
    # issue: line 55 cant be used in df.apply
    for row in df.itertuples():
        aspects_dict = wiki_dict.get(row.entity)
        if aspects_dict is None:
            df.drop(row.Index, inplace=True)
        elif row.aspect not in aspects_dict:
            for k in aspects_dict:
                if aspects_dict[k].get('heads') and row.aspect in aspects_dict[k].get('heads'):
                    df.at[row.Index, 'aspect'] = k
                    break
            else:
                df.drop(row.Index, inplace=True)

    df.index = range(len(df.index))

    return df

def using_ahocorasick(col, lst):
    A = ahocorasick.Automaton(ahocorasick.STORE_INTS)
    for word in lst:
        A.add_word(word.lower())
    A.make_automaton()
    col = col.str.lower()
    mask = col.apply(lambda x: bool(list(A.iter(x))))
    return mask


if __name__ == '__main__':

    if os.getenv('customer') in ['subj', 'obj', 'both_obj', 'both_subj']:
        customer = os.environ['customer']
        logging.info('source data set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both_obj' 'both_subj'.\n""")

    dir = Path(__file__).parent.parent
    wiki_file = Path.joinpath(dir, 'trained', 'wiki_'+customer+'.json')
    sentence_file = Path.joinpath(dir, 'trained', 'sentences_'+customer+'.json')
    wiki_dict, sentences = load_file(wiki_file, sentence_file, locale=customer)
    logger.info("start forming...")
    contexts_dt = form_dict(wiki_dict, sentences, locale=customer)
    with open(Path.joinpath(dir, 'trained', 'context_'+os.environ['customer']+'.json'), 'w', encoding='utf-8') as outfile:
        json.dump(contexts_dt, outfile)
    logger.info('Saved the file of context.')