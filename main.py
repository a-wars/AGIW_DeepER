from deeper_utils import preprocess_data
from deeper_model import build_model
from keras.callbacks import EarlyStopping


trainData, testData, valData, embeddingMatrix = preprocess_data(
    'Fodors_Zagats')
leftTableTrainData, rightTableTrainData, trainLabels = trainData
leftTableTestData, rightTableTestData, testLabels = testData
leftTableValData, rightTableValData, valLabels = valData

model = build_model(embeddingMatrix)
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
