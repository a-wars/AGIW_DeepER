from keras import Input
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Subtract, Activation
from keras.optimizers import Adam

def build_model(
        embeddingMatrix,
        maxSequenceLength=100,
        lstmUnits=150,
        denseUnits=[64],
        mask_zero=True,
        lstm_dropout=0.1):
    vocabSize = embeddingMatrix.shape[0]
    embeddingDim = embeddingMatrix.shape[1]
    leftInput = Input(shape=(maxSequenceLength,), dtype='int32')
    rightInput = Input(shape=(maxSequenceLength,), dtype='int32')


    embeddingLayer = Embedding(
    vocabSize,
    embeddingDim,
    input_length=maxSequenceLength,
    weights=[embeddingMatrix],
    trainable=True,
    mask_zero=mask_zero)
    
    leftEmbeddingLayer = embeddingLayer(leftInput)
    rightEmbeddingLayer = embeddingLayer(rightInput)
    
    sharedLSTMLayer = Bidirectional(LSTM(lstmUnits, dropout=lstm_dropout), merge_mode='concat')
    leftLSTMLayer = sharedLSTMLayer(leftEmbeddingLayer)
    rightLSTMLayer = sharedLSTMLayer(rightEmbeddingLayer)

    similarityLayer = Subtract()([leftLSTMLayer, rightLSTMLayer])
    
    # adding dense layers
    for i in range(len(denseUnits)):
        if i == 0:
            denseLayer = Dense(denseUnits[i], activation='relu')(similarityLayer)
        else:
            denseLayer = Dense(denseUnits[i], activation='relu')(denseLayer)

    outputLayer = Dense(2, activation='softmax')(denseLayer)

    model = Model(inputs=[leftInput, rightInput], outputs=[outputLayer])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.01, decay=0.001),
        metrics=['accuracy'])

    return model