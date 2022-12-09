from sklearn.metrics import confusion_matrix
import torch
import numpy as np

class ConfusionMatrix:

    def __init__(self, net, testloader, classes):
        self.classes = classes
        self.metrics = ['Tasa error', 'Exactitud', 'Precision', 'Recall', 'Specificity', 'F1 score']
        self.y_pred = []
        self.y_true = []
        self.get_preds(net, testloader)
        self.cm = confusion_matrix(self.y_pred, self.y_true)

    def get_preds(self, net, testloader):

        # iterate over test data
        for inputs, labels in testloader:
            output = net(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            self.y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            self.y_true.extend(labels)  # Save Truth

    def class_confusion_matrix(self, class_num):
        tp = self.cm[class_num, class_num]
        fp = self.cm[class_num, :].sum() - tp
        fn = self.cm[:, class_num].sum() - tp
        tn = self.cm.sum() - tp - fp - fn
        ccm = np.array([[tp, fp],
                        [fn, tn]])

        return ccm

    def get_metrics(self):
        nrows = len(self.classes)
        ncols = len(self.metrics)

        cme = np.zeros([nrows, ncols])

        for class_num in range(nrows):
            tp = self.cm[class_num, class_num]
            fp = self.cm[class_num, :].sum() - tp
            fn = self.cm[:, class_num].sum() - tp
            tn = self.cm.sum() - tp - fp - fn

            te = (fp + fn) / (tp + fp + fn + tn)
            ex = (tp + tn) / (tp + fp + fn + tn)
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            es = tn / (tn + fp)
            fs = 2 * pr * re / (pr + re)

            metric_list = [te, ex, pr, re, es, fs]

            for metric_num in range(ncols):
                cme[class_num, metric_num] = metric_list[metric_num]

        return cme




