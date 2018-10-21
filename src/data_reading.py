
def read_raw_data(file):
    res ={}
    # TODO use proper file path pacakge
    with open(file, 'r', encoding='utf-8') as f:
         for line in f:
             # TODO split line
             print(line)


if __name__ == '__main__':

    input_file = "data/subjlink.txt"
    read_raw_data(input_file)