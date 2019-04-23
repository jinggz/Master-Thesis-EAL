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

def load_file(context_file):
    with open(context_file, 'r', encoding='utf-8') as f:
        context_dict = json.load(f)
    logger.info("File loaded.")
    return context_dict

def create_train(contexts):
    trainset=[]

    i = 0 # index rows with same sentence
    for k, sent in tqdm(contexts.items()):
        A = sent["sent_context"]
        for asp in sent["aspect_candidates"]:
            B_1 = asp["id_aspect"]
            B_2 = asp["content"]
            if B_1 == sent["true"]:
                label = 1
            else:
                label = 0
            trainset.append([A, B_1, B_2, label, i])
        i += 1
    return trainset

if __name__ == '__main__':

    if os.getenv('customer') in ['subj', 'obj', 'both_obj', 'both_subj']:
        customer = os.environ['customer']
        logging.info('source data set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both_obj' 'both_subj'.\n""")

    dir = Path(__file__).parent.parent
    context_file = Path.joinpath(dir, 'trained', 'context_'+customer+'.json')
    contexts = load_file(context_file)
    logger.info("start forming...")

    # data [0]->sentence, [1]->aspect, [2]->aspect_content, [3]->label, [4]->sentence_index
    data = create_train(contexts)

    with open(Path.joinpath(dir, 'trained', 'labeled_'+os.environ['customer']+'.tsv'), 'w', encoding='utf-8') as outfile:
        tsv_writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["sentence", "aspect", "aspect_content", "label", "idx_sentence"])
        for i in data:
            tsv_writer.writerow(i)

    logger.info('Labeled datasets created.')