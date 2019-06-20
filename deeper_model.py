from keras import Input
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Subtract, Activation
from keras.optimizers import Adam

def build_model(
        embeddingMatrix,
        maxSequenceLength=1000,
        lstmUnits=150,
        denseUnits=64):
    vocabSize = embeddingMatrix.shape[0]
    embeddingDim = embeddingMatrix.shape[1]
    leftInput = Input(shape=(maxSequenceLength,), dtype='int32')
    rightInput = Input(shape=(maxSequenceLength,), dtype='int32')

    leftEmbeddingLayer = Embedding(
        vocabSize,
        embeddingDim,
        input_length=maxSequenceLength,
        weights=[embeddingMatrix],
        trainable=True,
        mask_zero=True)(leftInput)
    rightEmbeddingLayer = Embedding(
        vocabSize,
        embeddingDim,
        input_length=maxSequenceLength,
        weights=[embeddingMatrix],
        trainable=True,
        mask_zero=True)(rightInput)

    sharedLstmlayer = LSTM(lstmUnits, dropout=0.1)
    leftLSTMLayer = sharedLstmlayer(leftEmbeddingLayer)
    rightLSTMLayer = sharedLstmlayer(rightEmbeddingLayer)

    leftSamplingLayer = Dense(50, name="left_tuple_embedding")(leftLSTMLayer)
    rightSamplingLayer = Dense(50, name="right_tuple_embedding")(rightLSTMLayer)
    similarityLayer = Subtract()([leftSamplingLayer, rightSamplingLayer])
    denseLayer = Dense(denseUnits, activation='relu')(similarityLayer)
    outputLayer = Dense(2, activation='softmax')(denseLayer)

    model = Model(inputs=[leftInput, rightInput], outputs=[outputLayer])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.01),
        metrics=['accuracy'])

    return model