from __future__ import print_function

import os
import numpy as np
import pandas as pd
from keras import Input
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Subtract, Activation
from keras.optimizers import Adam

BASE_DIR = '.'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# first, build index mapping words in the embeddings set
# to their embedding vector
wordToEmbeddingMap = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        wordToEmbeddingMap[word] = coefs

print('Found %s word vectors.' % len(wordToEmbeddingMap))

# read train, test and validation datasets
trainDf = pd.read_csv('./datasets/Fodors_Zagats/Fodors_Zagats_train.csv')
valDf = pd.read_csv('./datasets/Fodors_Zagats/Fodors_Zagats_valid.csv')
testDf = pd.read_csv('./datasets/Fodors_Zagats/Fodors_Zagats_test.csv')


# extract labels from each dataset
trainLabels = trainDf['label']
valLabels = valDf['label']
testLabels = testDf['label']


# extract data from each dataset
leftTableTrainRecords = trainDf['attributi_x']
rightTableTrainRecords = trainDf['attributi_y']

leftTableValRecords = valDf['attributi_x']
rightTableValRecords = valDf['attributi_y']

leftTableTestRecords = testDf['attributi_x']
rightTableTestRecords = testDf['attributi_y']


# put train, test and validation records into a list
leftTableRecordsList = [
    leftTableTrainRecords,
    leftTableValRecords,
    leftTableTestRecords]
rightTableRecordsList = [
    rightTableTrainRecords,
    rightTableValRecords,
    rightTableTestRecords]


# concat previously defined lists
leftTableRecords = pd.concat(leftTableRecordsList)
rightTableRecords = pd.concat(rightTableRecordsList)


print('Found %s texts.' % len(leftTableRecords))


# finally, vectorize the text samples into a 2D integer tensor
leftTableTokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
leftTableTokenizer.fit_on_texts(leftTableRecords)
leftTableVectors = leftTableTokenizer.texts_to_sequences(leftTableRecords)
leftTableWordToIndexMap = leftTableTokenizer.word_index

rightTableTokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
rightTableTokenizer.fit_on_texts(rightTableRecords)
rightTableVectors = rightTableTokenizer.texts_to_sequences(rightTableRecords)
rightTableWordToIndexMap = rightTableTokenizer.word_index


print('Found %s unique tokens.' % len(leftTableWordToIndexMap))


# pad with zeros each integer sequence in each table up to MAX_SEQUENCE_LENGTH
leftTablePaddedVectors = pad_sequences(
    leftTableVectors, maxlen=MAX_SEQUENCE_LENGTH)
rightTablePaddedVectors = pad_sequences(
    rightTableVectors, maxlen=MAX_SEQUENCE_LENGTH)


# compute training, test and validation set sizes
trainingSetSize = leftTableTrainRecords.shape[0]
validationSetSize = leftTableValRecords.shape[0]
testSetSize = leftTableTestRecords.shape[0]


# split the data into training, test and validation set
leftTableTrainData = leftTablePaddedVectors[:trainingSetSize]
rightTableTrainData = rightTablePaddedVectors[:trainingSetSize]

leftTableValData = leftTablePaddedVectors[trainingSetSize:(
    trainingSetSize + validationSetSize)]
rightTableValData = rightTablePaddedVectors[trainingSetSize:(
    trainingSetSize + validationSetSize)]

leftTableTestData = leftTablePaddedVectors[-testSetSize:]
rightTableTestData = rightTablePaddedVectors[-testSetSize:]


print('Shape of training data:', leftTableTrainData.shape)
print('Shape of validation data:', leftTableValData.shape)
print('Shape of test data:', leftTableTestData.shape)
print('Preparing embedding matrix.')


# prepare embedding matrix
leftTableVocab = min(MAX_NUM_WORDS, len(leftTableWordToIndexMap)) + 1
leftTableEmbeddingMatrix = np.zeros((leftTableVocab, EMBEDDING_DIM))
for word, i in leftTableWordToIndexMap.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = wordToEmbeddingMap.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        leftTableEmbeddingMatrix[i] = embedding_vector

rightTableVocab = min(MAX_NUM_WORDS, len(rightTableWordToIndexMap)) + 1
rightTableEmbeddingMatrix = np.zeros((rightTableVocab, EMBEDDING_DIM))
for word, i in rightTableWordToIndexMap.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = wordToEmbeddingMap.get(word)
    if embedding_vector is not None:
        rightTableEmbeddingMatrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
inputA = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
inputB = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

x1 = Embedding(
    leftTableVocab,
    EMBEDDING_DIM,
    input_length=MAX_SEQUENCE_LENGTH,
    weights=[leftTableEmbeddingMatrix],
    trainable=True,
    mask_zero=True)(inputA)
x1 = LSTM(150)(x1)

x2 = Embedding(
    rightTableVocab,
    EMBEDDING_DIM,
    input_length=MAX_SEQUENCE_LENGTH,
    weights=[rightTableEmbeddingMatrix],
    trainable=True,
    mask_zero=True)(inputB)
x2 = LSTM(150)(x2)

subtracted = Subtract()([x1, x2])
dense = Dense(256, activation='relu')(subtracted)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[inputA, inputB], outputs=[output])
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'])
print(model.summary())

model.fit([leftTableTrainData,
           rightTableTrainData],
          trainLabels,
          batch_size=16,
          epochs=20,
          validation_data=([leftTableValData,
                            rightTableValData],
                           valLabels),
          callbacks=[EarlyStopping(patience=4)])

test_result = model.evaluate(
    x=[leftTableTestData, rightTableTestData], y=testLabels)
print(test_result)
