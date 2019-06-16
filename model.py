from __future__ import print_function

import os
import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Subtract, Activation
import pandas as pd

BASE_DIR = '.'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')


train_df = pd.read_csv('./datasets/Fodors_Zagats/Fodors_Zagats_train.csv')
validation_df = pd.read_csv('./datasets/Fodors_Zagats/Fodors_Zagats_valid.csv')
test_df = pd.read_csv('./datasets/Fodors_Zagats/Fodors_Zagats_test.csv')

train_labels = train_df['label']
valid_labels = validation_df['label']
test_labels = test_df['label']

left_train_text = train_df['attributi_x']
right_train_text = train_df['attributi_y']

left_valid_text = validation_df['attributi_x']
right_valid_text = validation_df['attributi_y']

left_test_text = test_df['attributi_x']
right_test_text = test_df['attributi_y']

frames1 = [left_train_text, left_valid_text, left_test_text]
frames2 = [right_train_text, right_valid_text, right_test_text]
label_frames = [train_labels, valid_labels, test_labels]
text1 = pd.concat(frames1)
text2 = pd.concat(frames2)
labels = pd.concat(label_frames)

print('Found %s texts.' % len(text1))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer1 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer1.fit_on_texts(text1)
sequences1 = tokenizer1.texts_to_sequences(text1)
word_index1 = tokenizer1.word_index

tokenizer2 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer2.fit_on_texts(text2)
sequences2 = tokenizer2.texts_to_sequences(text2)
word_index2 = tokenizer2.word_index

print('Found %s unique tokens.' % len(word_index1))

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)

# split the data into a training set and a validation set
num_training_samples = left_train_text.shape[0]
num_validation_samples = left_valid_text.shape[0]
num_test_samples = left_test_text.shape[0]

x_train1 = data1[:num_training_samples]
x_train2 = data2[:num_training_samples]
y_train = labels[:num_training_samples]
x_val1 = data1[num_training_samples:(
    num_training_samples + num_validation_samples)]
x_val2 = data2[num_training_samples:(
    num_training_samples + num_validation_samples)]
y_val = labels[num_training_samples:(
    num_training_samples + num_validation_samples)]
x_test1 = data1[-num_test_samples:]
x_test2 = data2[-num_test_samples:]
y_test = labels[-num_test_samples:]


print('Shape of training data:', x_train1.shape)
print('Shape of validation data:', x_val1.shape)
print('Shape of test data:', x_test1.shape)
print('Preparing embedding matrix.')

# prepare embedding matrix
num_words1 = min(MAX_NUM_WORDS, len(word_index1)) + 1
embedding_matrix1 = np.zeros((num_words1, EMBEDDING_DIM))
for word, i in word_index1.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix1[i] = embedding_vector

num_words2 = min(MAX_NUM_WORDS, len(word_index2)) + 1
embedding_matrix2 = np.zeros((num_words2, EMBEDDING_DIM))
for word, i in word_index2.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix2[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
inputA = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
inputB = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

x1 = Embedding(num_words1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,
               weights=[embedding_matrix1], trainable=True)(inputA)
x1 = LSTM(150)(x1)

x2 = Embedding(num_words2, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,
               weights=[embedding_matrix2], trainable=True)(inputB)
x2 = LSTM(150)(x2)

subtracted = Subtract()([x1, x2])
dense = Dense(256, activation='relu')(subtracted)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[inputA, inputB], outputs=[output])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
print(model.summary())

model.fit([x_train1, x_train2], y_train, batch_size=32, epochs=20, validation_data=(
    [x_val1, x_val2], y_val), callbacks=[EarlyStopping(patience=4)])

test_result = model.evaluate(x=[x_test1, x_test2], y=y_test)
print(test_result)
