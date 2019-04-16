import os
import re
from pathlib import Path
import json
import logging
import csv
import pandas as pd
import ahocorasick
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

def load_file(context_file, locale):
    with open(context_file, 'r', encoding='utf-8') as f:
        context_dict = json.load(f)
    logger.info("File loaded.")
    return context_dict

def create_train(contexts, locale):
    trainset_1=[]
    trainset_2=[]

    for k, sent in tqdm(contexts.items()):
        A = sent["sent_context"]
        for asp in sent["aspect_candidates"]:
            B_1 = asp["id_aspect"]
            B_2 = asp["content"]
            if B_1 == sent["true"]:
                label = 1
            else:
                label=0
            trainset_1.append([A, B_1, label])
            trainset_2.append([A, B_2, label])
    return trainset_1, trainset_2

if __name__ == '__main__':

    if os.getenv('customer') in ['subj', 'obj', 'both_obj', 'both_subj']:
        customer = os.environ['customer']
        logging.info('source data set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both_obj' 'both_subj'.\n""")

    dir = Path(__file__).parent.parent
    context_file = Path.joinpath(dir, 'trained', 'context_'+customer+'.json')
    contexts = load_file(context_file, locale=customer)
    #logger.info("start forming...")
    # dt1: aspect title, dt2: aspect content
    # dt1&dt2 [0]->sentence, [1]->aspect, [2]->label
    dt1, dt2 = create_train(contexts, locale=customer)

    with open(Path.joinpath(dir, 'trained', 'labeled_title_'+os.environ['customer']+'.tsv'), 'w', encoding='utf-8') as outfile:
        tsv_writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')
        for i in dt1:
            tsv_writer.writerow(i)
    with open(Path.joinpath(dir, 'trained', 'labeled_content_'+os.environ['customer']+'.tsv'), 'w', encoding='utf-8') as outfile:
        tsv_writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')
        for i in dt2:
            tsv_writer.writerow(i)
    logger.info('Labeled datasets created.')