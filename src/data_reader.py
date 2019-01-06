import os
import re
from pathlib import Path
import json
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

class DataReader:

    def __init__(self):
        self.placeholder = ('')

    def read_raw_data(self, file):
        # example line:
        # Sentence: Filming took place at Cinevillage Studios in Toronto , Canada and on locations in Toronto including Eaton Centre to Ontario Place The four movies were first broadcast in 1994 on CTV in Canada and in syndication in the United States as part of Universal Television 's Action Pack .
        # Triple: "cinevillage studios"	"is in"	"toronto"	factuality: (POSITIVE,CERTAINTY)	quantities: ()	attribution: (NO ATTRIBUTION DETECTED)	time: ()	space: (pred=in, Toronto , premods: , postmods: ; )
        # Links: (SubjLink:[Leslieville#History]	ObjLink:[Toronto, Canada])
        res = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = self.extract_sample(line)
                res.append(sample)
        logger.info('The data has been extracted.')
        return res

    def extract_sample(self, line):

        line_splited = line.strip().split("\t")

        # extract sentence
        sentence = line_splited[0].split(":", 1)[1].strip()
        # extract triple
        triple = []
        tmp = re.split(r'[\:]+', line_splited[1])
        triple.append(tmp[1].lstrip().strip('"'))
        for i in range(2, len(line_splited)):
            if not line_splited[i].startswith('factuality'):
                triple.append(line_splited[i].strip('"'))
            else:
                break
        # extract factuality & link
        for idx, term in enumerate(line_splited):
            if term.startswith('factuality'):
                fact = term[term.find('(')+1:term.find(')')]
                fact = fact.split(',')
            elif term.startswith(self.placeholder):
                entity = term[term.find('[') + 1:term.find('#')]
                aspect = term[term.find('#') + 1:term.find(']')]
        if not entity:
            entity = 'None'
        if not aspect:
            aspect = 'None'
        res = {'sentence': sentence, 'entity': entity, 'aspect': aspect, 'triple': triple, 'factuality': fact}
        return res

    def save2pickle(self, file,result):
        with open(file, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def save2json(self, file, result):
        with open(file, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile)

    def get_file_dir(self):
        return Path(__file__).parent.parent

class SubjReader(DataReader):
    def __init__(self):
        self.placeholder = ('Links')

class ObjReader(DataReader):
    def __init__(self):
        self.placeholder = ('ObjLink')

class BothJReader(DataReader):
    # todo. have to differentiate 2 types of link
    def __init__(self):
        self.placeholder = ('Links', 'ObjLink')


if __name__ == '__main__':
    os.environ['customer'] = 'obj'
    if os.getenv('customer') in ['subj', 'obj', 'both']:
        logger.info('source data  set to ' + os.environ['customer'])
    else:
        raise NameError("""Please set an environment variable to indicate which source to use.\n
        Your options are: customer='subj' or 'obj' or 'both'.\n""")

    dir = Path(__file__).parent.parent

    input_file = Path.joinpath(dir, "data", os.environ['customer']+"link.txt")
    output_folder = Path.joinpath(dir, 'trained')

    if not Path(output_folder).is_dir():
        Path(output_folder).mkdir()

    if os.environ['customer'] == 'subj':
        reader = SubjReader()
    elif os.environ['customer'] == 'obj':
        reader = ObjReader()
    else:
        reader = BothJReader()
    result = reader.read_raw_data(input_file)
    reader.save2json(Path.joinpath(dir, 'trained', 'sentences_'+os.environ['customer']+'.json'), result)
    logger.info('Saved the data.')


