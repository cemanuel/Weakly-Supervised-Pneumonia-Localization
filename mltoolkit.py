from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class MultiClassClassifier(object):
    '''Fits a multi-class classification model using mllib classification method and
    returns classification metrics.

    Attributes
    ----------
        predictor: type of multi-class classifier
        label (str): classification label
        testFraction (float): test set fraction [0.3]
        seed (int): random seed
        average (str): Type of averaging for multiclass classification metrics
                   - micro
                   - macro
                   - weighted (default)
                   - samples
    '''


    def __init__(self, predictor, label="indexedLabel", testFraction=0.3, seed=1, average = 'weighted'):

        self.predictor = predictor
        self.label = label
        self.testFraction = testFraction
        self.seed = seed
        self.prediction = None
        self.average = average

    def fit(self, df):
        '''Dataset must at least contain the following two columns:
        label: the class labels
        features: feature vector

        Attributes
        ----------
            data (DataFrame): input data

        Returns
        -------
            map with metrics
        '''

        start = time.time()

        numClass = len(df[self.label].unique())

        # Convert label to indexedLabel
        #labelIndexer, labelConverter = indexer(df, self.label)
        #df['indexedLabel'] = df[self.label].map(labelIndexer)

        # Split the data into trianing and test set
        train, test = train_test_split(df, test_size=self.testFraction)

        print("\n Class\tTrain\tTest")
        for l in df[self.label].unique():
            print(f"\n{l}\t{train[train[self.label] == l].shape[0]}\t{test[test[self.label] == l].shape[0]}")

        # Train model
        clf = self.predictor.fit(train.features.tolist(), train.indexedLabel.tolist()) # Set predicting class to numeric values

        # Make prediction
        pred = clf.predict(test.features.tolist())
        scores = clf.predict_proba(test.features.tolist())

        test = test.reset_index()
        test['predictions'] = pd.Series(pred) #.map(labelConverter)
        positive_prob = [s[1] for s in scores]
        self.prediction = test

        metrics = OrderedDict()
        metrics["Methods"] = str(self.predictor).split('(')[0]

        precision, recall, fscore, support = precision_recall_fscore_support(test.indexedLabel, pred)
        metrics_table = OrderedDict([('Labels', df[self.label].unique()) ,
                                    ('Precision', ["%.2f"%p + '%' for p in precision]),
                                    ('Recall', ["%.2f"%r  + '%' for r in recall]),
                                    ('FScore' , ["%.2f"%f for f in fscore]),
                                    ('Support' , support),]
                                    )
        self.metrics_table = pd.DataFrame(metrics_table)

        if numClass == 2:
            metrics['AUC'] = str(roc_auc_score(test.indexedLabel, positive_prob))
            self.TPR, self.FPR, thresholds = roc_curve(test.indexedLabel, positive_prob)
            self.AUC = auc(self.TPR, self.FPR)

            metrics['F Score'] = str(f1_score(test.indexedLabel, pred))
            metrics['Accuracy'] = str(accuracy_score(test.indexedLabel, pred))
            metrics['Precision'] = str(precision_score(test.indexedLabel, pred))
            metrics['Recall'] = str(recall_score(test.indexedLabel, pred))

        else:
            metrics['F Score'] = str(f1_score(test.indexedLabel, pred, average = self.average))
            metrics['Accuracy'] = str(accuracy_score(test.indexedLabel, pred))
            metrics['Precision'] = str(precision_score(test.indexedLabel, pred, average = self.average))
            metrics['Recall'] = str(recall_score(test.indexedLabel, pred, average = self.average))

        confusionMatrix = confusion_matrix(test.indexedLabel, pred)

        if numClass > 2:
            FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
            FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
            TP = np.diag(confusionMatrix)
            TN = confusionMatrix.sum() - (FP + FN + TP)
        else:
            FP = confusionMatrix[0][1]
            FN = confusionMatrix[1][0]
            TP = confusionMatrix[0][0]
            TN = confusionMatrix[1][1]

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)

        average_rate = lambda x: sum(x)/float(len(x)) if type(x) == np.ndarray else x
        TPR = average_rate(TPR)
        FPR = average_rate(FPR)

        metrics["False Positive Rate"] = str(FPR)
        metrics["True Positive Rate"] = str(TPR)

        metrics[""] = f"\nConfusion Matrix\n{df[self.label].unique()}" \
                      + f"\n{confusionMatrix}"

        end = time.time()
        print(f"\nTotal time taken: {end-start}\n")

        return metrics


def plot_roc(TPR, FPR, AUC):
    plt.title('Receiver Operating Characteristic')
    plt.plot(TPR, FPR, 'b',
    label='AUC = %0.2f'% AUC)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
