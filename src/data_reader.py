import re
from pathlib import Path

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
        res = {'sentence': sentence, 'triple': triple, 'factuality': fact, 'entity': entity, 'aspect': aspect}
        return res


if __name__ == '__main__':

    input_file = "data/subjlink.txt"
    input_dir = Path.joinpath(Path(__file__).parent.parent, input_file)
    reader = DataReader()
    result = reader.read_raw_data(input_dir)
    # TODO maybe save dict? then how to read directly?