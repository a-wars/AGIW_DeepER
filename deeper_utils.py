import os
import numpy as np
import pandas as pd
import fasttext
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# read training, test and validation datasets
def read_dataset(dataset_filepath_fmt):
    trainDf = pd.read_csv(dataset_filepath_fmt.format('train'))
    valDf = pd.read_csv(dataset_filepath_fmt.format('valid'))
    testDf = pd.read_csv(dataset_filepath_fmt.format('test'))
    return trainDf, valDf, testDf


# get training, test and validation set lengths
def compute_split_sizes(trainDf, valDf, testDf):
    trainingSetSize = trainDf.shape[0]
    validationSetSize = valDf.shape[0]
    testSetSize = testDf.shape[0]
    return trainingSetSize, validationSetSize, testSetSize


# extract labels from each dataset
def get_labels(trainDf, valDf, testDf):
    trainLabels = to_categorical(np.asarray(trainDf['label']))
    valLabels = to_categorical(np.asarray(valDf['label']))
    testLabels = to_categorical(np.asarray(testDf['label']))
    return trainLabels, valLabels, testLabels


# extract "attributi" column from each row of the given dataframe
def get_records(df):
    leftTableRecords = df['attributi_x']
    rightTableRecords = df['attributi_y']
    return leftTableRecords, rightTableRecords


# get left table and right table records
def get_left_right_tables_records(trainDf, valDf, testDf):
    # extract records from each dataset
    leftTableTrainRecords, rightTableTrainRecords = get_records(trainDf)
    leftTableValRecords, rightTableValRecords = get_records(valDf)
    leftTableTestRecords, rightTableTestRecords = get_records(testDf)

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

    return leftTableRecords, rightTableRecords, tableRecords


# pad with zeros each integer sequence in each
# table up to maxSequenceLength
def pad_table_vectors(leftTableVectors, rightTableVectors, maxSequenceLength):
    leftTablePaddedVectors = pad_sequences(leftTableVectors, maxlen=maxSequenceLength)
    rightTablePaddedVectors = pad_sequences(rightTableVectors, maxlen=maxSequenceLength)
    return leftTablePaddedVectors, rightTablePaddedVectors


# returns a dictionary mapping words in the embeddings set
# to their embedding vector
def get_word_to_embedding_map(filepath):
    wordToEmbeddingMap = {}
    with open(filepath,encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            # we found out that glove.840B.300d.txt is supposed to contain 301-columns 
            # (a word and its 300 weights) but some lines contain 
            # more than one word, so we ignore those malformed lines 
            # (whose corresponding "coefs" variable is an empty array)
            if len(coefs) != 0:
                wordToEmbeddingMap[word] = coefs
    return wordToEmbeddingMap


def preprocess_data(
        datasetName,
        baseDir='.',
        usePretrainedModel=True,
        embeddingDir='fasttext-model',
        embeddingFilename='crawl-300d-2M-subword.bin',
        datasetDir='datasets',
        maxSequenceLength=100,
        maxNumWords=20000):
    
    # define contants
    EMBEDDING_DIM = 300
    EMBEDDING_DIR = os.path.join(baseDir, embeddingDir)
    EMBEDDING_FILEPATH = os.path.join(EMBEDDING_DIR, embeddingFilename)
    DATASET_DIR = os.path.join(baseDir, datasetDir, datasetName)
    DATASET_FILENAME_FMT = datasetName + '_{}.csv'
    DATASET_FILEPATH_FMT = os.path.join(DATASET_DIR, DATASET_FILENAME_FMT)


    if usePretrainedModel:
        model = fasttext.load_model(EMBEDDING_FILEPATH)
    else: 
        # load embedding matrix from a file
        wordToEmbeddingMap = get_word_to_embedding_map(EMBEDDING_FILEPATH)

    # read training, test and validation datasets
    trainDf, valDf, testDf = read_dataset(DATASET_FILEPATH_FMT)

    # compute training, test and validation set sizes
    trainingSetSize, validationSetSize, testSetSize = compute_split_sizes(trainDf, valDf, testDf)

    # extract labels from each dataset
    trainLabels, valLabels, testLabels = get_labels(trainDf, valDf, testDf)

    # get left table and right table records
    leftTableRecords, rightTableRecords, tableRecords = get_left_right_tables_records(trainDf, valDf, testDf)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=maxNumWords)
    tokenizer.fit_on_texts(tableRecords)
    wordToIndexMap = tokenizer.word_index

    leftTableVectors = tokenizer.texts_to_sequences(leftTableRecords)
    rightTableVectors = tokenizer.texts_to_sequences(rightTableRecords)

    # pad with zeros each integer sequence in each
    # table up to maxSequenceLength
    leftTablePaddedVectors, rightTablePaddedVectors = pad_table_vectors(leftTableVectors, rightTableVectors, maxSequenceLength)

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

        #get word embeddings
        if usePretrainedModel:
            embeddingVector = model.get_word_vector(word)
        else:
            embeddingVector = wordToEmbeddingMap.get(word)

        # add computed word embedding into our embedding matrix
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