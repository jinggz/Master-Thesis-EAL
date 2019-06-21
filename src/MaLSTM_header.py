
# coding: utf-8

# In[4]:

import time
import pandas as pd
import os
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm

from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
# from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


# In[18]:


from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# In[38]:

logger = logging.getLogger('main')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/lstm_ranking_content_'+os.environ['customer']+'.log',
    filemode='w')

dir = Path(__file__).parent.parent

def load_file(train, test):
    train_df = pd.read_csv(train, delimiter='\t', encoding='utf-8')
    test_df = pd.read_csv(test, delimiter='\t', encoding='utf-8')
    #logger.info("File loaded.")
    return train_df, test_df


# In[39]:

train_file = Path.joinpath(dir, 'trained', 'cleaned_subj.tsv')
test_file = Path.joinpath(dir, 'trained', 'cleaned_both_subj.tsv')

train_df_unsampled, test_df = load_file(train_file, test_file)

# In[41]:

#train_df_unsampled, test_df = train_test_split(data, test_size=0.1, random_state=42)
# upsampling positive samples
max_size = train_df_unsampled['label'].value_counts().max()
lst = [train_df_unsampled]
for class_index, group in train_df_unsampled.groupby('label'):
    lst.append(group.sample(max_size-len(group), replace=True))
train_df = pd.concat(lst)
train_df.index = range(len(train_df.index))

# In[22]:


# Prepare embedding
EMBEDDING_FILE = Path.joinpath(dir, 'model', 'glove.6B.300d.txt')
tmp_file = get_tmpfile("temp_word2vec.txt")
glove2word2vec(EMBEDDING_FILE, tmp_file)
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
word2vec = KeyedVectors.load_word2vec_format(tmp_file, binary=False)


# In[47]:


questions_cols = ['sentence', 'aspect']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for question in questions_cols:

            q2n = []  # q2n -> question numbers representation
            
            words = row[question].split() # do whitespace token

            # truncate words after 80 tokens to speed up training
            # if len(words) >= 174:
            #     words = words[:174]
            if len(words) >= 50:  #for header
                words = words[:50]

            for word in words:

                # Check for unwanted words
                if word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.at[index, question] = q2n


# In[48]:


embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for word, index in tqdm(vocabulary.items()):
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec


# In[52]:


# Prepare training and validation data
max_seq_length = max(train_df.sentence.map(lambda x: len(x)).max(),
                     train_df.aspect.map(lambda x: len(x)).max(),
                     test_df.sentence.map(lambda x: len(x)).max(),
                     test_df.aspect.map(lambda x: len(x)).max())

# Split to train validation
validation_size = 0.1
training_size = len(train_df) - validation_size

X_train = train_df[questions_cols]
Y_train = train_df['label']

#X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.sentence, 'right': X_train.aspect}
#X_validation = {'left': X_validation.sentence, 'right': X_validation.aspect_content}
X_test = {'left': test_df.sentence, 'right': test_df.aspect}

# Convert labels to their numpy representations
Y_train = Y_train.values
#Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# In[57]:


# Build the model
# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 30 #25,30
class_weight = {1: 0.9, 0: 0.1} # for balancing data

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings),
                            embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)
# here if add softmax?
left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir=LOG_DIR.format(datetime.time()))


# In[58]:


# Start training

training_start_time = time.time()


es = EarlyStopping(monitor='val_loss', min_delta=0, patience=-2, verbose=0, mode='auto', restore_best_weights=True)

#mc = ModelCheckpoint('model\\weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=1)

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_split=0.1, class_weight=None, callbacks=[es])

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time.time()-training_start_time)))

# Predict
y_true = test_df['label']
y_true = y_true.values
y_pred = malstm.predict([X_test['left'], X_test['right']])

# Evaluation using p@1
def get_precision(test_df):
    # get the max(cos sim) by grouping by sentence_idx, then by comparing with label, get the p@1
    return test_df.groupby(test_df.idx_sentence).apply(__precision)

def __precision( x):
    '''
    for each sentence group, get the index of predication then return the precision at 1
    '''

    if x.loc[x["similarity"].idxmax(), "label"] == 1:
        y_pred = 1
    else:
        y_pred = 0
    return y_pred

def avg_precision(p_list, rel_tol=1e-03):
    '''
    return the moving average p@1, stop when the different of last two p@1 smaller than rel_tol
    :param: p: the list of p@1
    :param rel_tol: the relelvant tolerance between 2 precisions,(0,1)
    :type: double
    :return: average p@1
    :return: indicator of convergence
    '''
    ap_end = sum(p_list) / len(p_list)
    for i in range(100,len(p_list)):
        ap_next = sum(p_list[:i+1]) / (i+1)
        ap_current = sum(p_list[:i])/i
        ap_last = sum(p_list[:i-1]) / (i-1)
        if abs(ap_last/ap_current-1)<=rel_tol and abs(ap_next/ap_current-1)<=rel_tol:
            map = ap_current
            logger.info('The AP at 1 converged at %s th samples' % i)
            logger.info('The AP at 1 is %.4f.' % map)
            break
    else:
        logger.info('The AP does not converge.')
        logger.info('The AP at 1 is %.4f.' % ap_end)
    logger.info('The final AP at 1 is %.4f, with %d samples' % (ap_end, len(p_list)))

test_df["similarity"] = y_pred
precision_list = get_precision(test_df)
logger.info('calculating the average precision @1')
avg_precision(precision_list)


# Evaluation using F1-score

y_pred = y_pred > 0.5
y_pred = y_pred.flatten().astype(int)
acc = accuracy_score(y_true, y_pred)
f_score = f1_score(y_true, y_pred, average='binary', pos_label=1)

print("The accuracy evaluated on test dataset is %.4f" % acc)
print('The F1 score evaluated on test dataset is %.4f' % f_score)
print(classification_report(y_true, y_pred))

# Save model
MODEL_FILE = Path.joinpath(dir, 'model', 'malstm_subj_header_0621.h5')
malstm.save_weights(MODEL_FILE)
