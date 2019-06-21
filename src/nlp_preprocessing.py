import re
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

def nlp_pipeline(text):
    '''
    this pipeline aims to do a series of preprocessing steps:
    tokenizing, punctuations and numbers removal, lemmatization
    :param:text_list
    :type: a list of str
    :return: a list of preprocessed str
    '''
    text = str(text)
    text = text.lower()
    text = word_tokenize(text)
    text = [token for token in text if token not in string.punctuation and token.isalpha()]
    text = [token for token in text if token not in stop_word_list]
    text = [lemmatizer.lemmatize(token) for token in text]
    wordstring = ' '.join(text)
    return wordstring

def whitespace_tokenize(text):

    text = str(text)
    text = text.split()
    return text

def extra_cleaning_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"e-mail", "email", text)

    return text

if __name__ == '__main__':
    data=["I'm e-mail we'll what's The son of James H. Ganong and Susan E. Brittain , he is the brother of Susie , Kit -LRB- Whidden -RRB- , Arthur , and William .	"]
    result = nlp_pipeline(data)
    print(result)