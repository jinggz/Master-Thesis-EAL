import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
punct = set(string.punctuation)
stop_word_list = set(stopwords.words('english'))

def nlp_pipeline(text_list):
    '''
    this pipeline aims to do a series of preprocessing steps:
    tokenizing, punctuations and numbers removal, lemmatization
    :param:text_list
    :type: a list of str
    :return: a list of preprocessed str
    '''
    string_list = []
    for text in text_list:
        text = text.lower()
        text = word_tokenize(text)
        text = [token for token in text if token not in string.punctuation and token.isalpha()]
        text = [token for token in text if token not in stop_word_list]
        text = [lemmatizer.lemmatize(token) for token in text]
        wordstring = ' '.join(text)
        string_list.append(wordstring)
    return string_list

def whitespace_tokenize(text_list):
    string_list = []
    for text in text_list:
        text = text.split()
        string_list.append(text)
    return string_list

if __name__ == '__main__':
    data=["The son of James H. Ganong and Susan E. Brittain , he is the brother of Susie , Kit -LRB- Whidden -RRB- , Arthur , and William .	"]
    result = whitespace_tokenize(nlp_pipeline(data))
    print(result)