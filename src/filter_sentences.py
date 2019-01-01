# This file is used to filter out unwanted sentences samples,
# and store wanted samples into sentences_clean_xxx.json for furture training
# The following samples will be excluded or modified:
    # no corresponding entity page found in our wiki corpus
    # entity name starting with “List of” or “Lists of", or containing “(disambiguation)”
    # no corresponding aspect found in our wiki corpus
    # for aspects found as lower level of headings, replace with h2 headings
    # remove duplicates
# reference: http://trec-car.cs.unh.edu/process/dataselection.html
import os
import json
import logging
from pathlib import Path
import pandas as pd
import ahocorasick

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

def filter_samples(wiki_dict, df):
    '''
    :param
    wiki_dict: the dictionary of entity-aspects pages
    :type: dict
    sentences: the raw sentence tuples extracted from Kiril's data
    :type: dataframe
    :return: sentences: the filtered sentences samples for training
    :type: dataframe
    '''

    df.drop(columns=['factuality', 'triple'], inplace=True) # only use column 'sentence', 'entity', 'aspect'
    df.drop_duplicates(inplace=True)
    df.drop(df[(df.entity == 'None') | (df.aspect == 'None')].index, inplace=True)
    df.index = range(len(df.index))
    df = df.apply(lambda x: x.str.lower())
    df.entity = df.entity.replace('_', ' ')
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

    os.environ['customer'] = 'subj'
    if os.getenv('customer') in ['subj', 'obj', 'both']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both'.\n""")
    dir = Path(__file__).parent.parent
    wiki_file = Path.joinpath(dir, 'trained', 'wiki_'+os.environ['customer']+'.json')
    sentence_file = Path.joinpath(dir, 'trained', 'sentences_'+os.environ['customer']+'.json')
    out_file = Path.joinpath(dir, 'trained', 'sentences_clean_'+os.environ['customer']+'.json')
    with open(wiki_file, 'r', encoding='utf-8') as f:
        wiki_dict = json.load(f)
    sentences = pd.read_json(path_or_buf=sentence_file,
                      orient='records')
    logger.info('Original files loaded')

    clean_sentences = filter_samples(wiki_dict, sentences)
    # no need to save, can be used as the next input of tfidf_ranking.py
    # with open(out_file, 'w', encoding='utf-8') as f:
    #     json.dump(clean_sentences, f)
    # logger.info('Stored the cleaned sentences file to %s' % out_file)