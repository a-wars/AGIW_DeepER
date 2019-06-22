import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict([self.validation_data[0],self.validation_data[1]])
        val_targ = self.validation_data[2]
        predictedLabels = val_predict.argmax(axis=1)
        testLabels = val_targ.argmax(axis=1)
        precision, recall, fMeasure, support = precision_recall_fscore_support(testLabels,predictedLabels,average='binary')
        self.val_f1s.append(fMeasure)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        message_fmt="- val_f1:{} - val_precision: {} - val_recall: {}"
        print (message_fmt.format(fMeasure,precision,recall))
        return
