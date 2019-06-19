import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def preprocess_data(
        datasetName,
        baseDir='.',
        gloveDir='glove',
        datasetDir='datasets',
        maxSequenceLength=1000,
        maxNumWords=20000):
    
    EMBEDDING_DIM = 300
    GLOVE_DIR = os.path.join(baseDir, gloveDir)
    DATASET_DIR = os.path.join(baseDir, datasetDir, datasetName)
    GLOVE_FILENAME = 'glove.840B.300d.txt'
    DATASET_FILENAME_FMT = datasetName + '_{}.csv'
    DATASET_FILEPATH_FMT = os.path.join(DATASET_DIR, DATASET_FILENAME_FMT)

    # first, build a dictionary mapping words in the embeddings set
    # to their embedding vector
    wordToEmbeddingMap = {}
    with open(os.path.join(GLOVE_DIR, GLOVE_FILENAME)) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            # we found out that glove.840B.300d.txt is supposed to contain 301-item lines (the token and its 300 weights) but some lines do contain 
            # more than one str tokens, so we ignore those malformed lines (whose corresponding coefs variable is an empty array)
            if len(coefs) != 0:
                wordToEmbeddingMap[word] = coefs

    # read train, test and validation datasets
    trainDf = pd.read_csv(DATASET_FILEPATH_FMT.format('train'))
    valDf = pd.read_csv(DATASET_FILEPATH_FMT.format('valid'))
    testDf = pd.read_csv(DATASET_FILEPATH_FMT.format('test'))

    # extract labels from each dataset
    trainLabels = to_categorical(np.asarray(trainDf['label']))
    valLabels = to_categorical(np.asarray(valDf['label']))
    testLabels = to_categorical(np.asarray(testDf['label']))

    # extract records from each dataset
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
    tableRecordsList = leftTableRecordsList + rightTableRecordsList

    # concat previously defined lists
    leftTableRecords = pd.concat(leftTableRecordsList)
    rightTableRecords = pd.concat(rightTableRecordsList)
    tableRecords = pd.concat(tableRecordsList)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=maxNumWords)
    tokenizer.fit_on_texts(tableRecords)
    wordToIndexMap = tokenizer.word_index
    leftTableVectors = tokenizer.texts_to_sequences(leftTableRecords)
    rightTableVectors = tokenizer.texts_to_sequences(rightTableRecords)

    # pad with zeros each integer sequence in each table up to
    # maxSequenceLength
    leftTablePaddedVectors = pad_sequences(
        leftTableVectors, maxlen=maxSequenceLength)
    rightTablePaddedVectors = pad_sequences(
        rightTableVectors, maxlen=maxSequenceLength)

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

    # prepare embedding matrix
    vocabSize = min(maxNumWords, len(wordToIndexMap)) + 1
    embeddingMatrix = np.zeros((vocabSize, EMBEDDING_DIM))

    wordsWithNoEmbedding = []
    for word, i in wordToIndexMap.items():
        if i > maxNumWords:
            continue
        embeddingVector = wordToEmbeddingMap.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector
        else:
            # words not found in embedding index will be all-zeros.
            wordsWithNoEmbedding.append(word)
    
    # return training, test and validation splits and embedding matrix (and words with no embeddings)
    trainData = [leftTableTrainData, rightTableTrainData, trainLabels]
    testData = [leftTableTestData, rightTableTestData, testLabels]
    valData = [leftTableValData, rightTableValData, valLabels]

    return trainData, testData, valData, embeddingMatrix, wordsWithNoEmbedding


def calculate_fmeasure(model, test_set, test_labels):
    predictions = model.predict(x=test_set)
    predicted_labels = []
    for pred in predictions:
        if pred[1] > pred[0]:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    truepositives = 0
    falsepositives = 0
    falsenegatives = 0
    for idx, pred in enumerate(predicted_labels):
        if pred == 1 and test_labels[idx][1] == 1:
            truepositives += 1
        elif pred == 0 and test_labels[idx][1] == 1:
            falsenegatives += 1
        elif pred == 1 and test_labels[idx][0] == 1:
            falsepositives += 1
    recall = truepositives / (truepositives + falsenegatives)
    precision = truepositives / (truepositives + falsepositives)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    return f_measure