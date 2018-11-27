import re
from pathlib import Path
import pprint as pp
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

        # exact sentence
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
            elif term.startswith('Links'):
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


if __name__ == '__main__':

    dir = Path(__file__).parent.parent

    input_file = Path.joinpath(dir, "data/subjlink.txt")
    output_folder = Path.joinpath(dir, 'output')

    if not Path(output_folder).is_dir():
        Path(output_folder).mkdir()

    reader = DataReader()
    result = reader.read_raw_data(input_file)
    reader.save2json(Path.joinpath(dir, 'output/sentences_eal_subj.json'), result)
    logger.info('Saved the data.')
    #reader.save2pickle(output_file_1)

    # load
    with open(Path.joinpath(dir, 'output/sentences_eal_subj.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(data[0])

    # load
    # with open(Path.joinpath(dir, output_file_1), 'rb') as handle:
    #     data = pickle.load(handle)

