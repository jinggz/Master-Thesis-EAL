
from pathlib import Path


def read_raw_data(file):
    # read the subjlink.txt file and store the needed tuples into dict
    res ={}

    with open(file, 'r', encoding='utf-8') as f:
         for line in f:
             # TODO split line
             print(line)


if __name__ == '__main__':

    input_file = "data/subjlink.txt"
    input_dir = Path.joinpath(Path(__file__).parent.parent , input_file)
    read_raw_data(input_dir)