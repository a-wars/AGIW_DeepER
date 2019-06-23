from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.valFMeasureHistory = []
        self.valRecallHistory = []
        self.valPrecisionHistory = []
        self.bestFMeasure = 0

    def on_epoch_end(self, epoch, logs={}):
        valPredictedLabels = self.model.predict(
            [self.validation_data[0], self.validation_data[1]])

        valLabels = self.validation_data[2]

        valPredictedBinaryLabels = valPredictedLabels.argmax(axis=1)
        valBinaryLabels = valLabels.argmax(axis=1)

        valPrecision, valRecall, valFMeasure, _ = precision_recall_fscore_support(
            valBinaryLabels, valPredictedBinaryLabels, average='binary')

        if valFMeasure is None:
            valFMeasure = 0.0


        if valFMeasure > self.bestFMeasure:
        	print('Updating best model')
        	print('Current best model comes from epoch {}'.format(str(epoch)))
            self.bestFMeasure = valFMeasure
            self.model.save('best-model.h5')
            
        self.valFMeasureHistory.append(valFMeasure)
        self.valRecallHistory.append(valRecall)
        self.valPrecisionHistory.append(valPrecision)

        valMessageFmt = "val_f1:{}\tval_precision: {}\tval_recall: {}"
        print(valMessageFmt.format(round(valFMeasure,2), round(valPrecision,2), round(valRecall, 2)))
        print()

        return
