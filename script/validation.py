#!/usr/bin/env python
# encoding: utf-8
import os, sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

class Validation:
    def __init__(self):
        pass

    def calculateF1(self, y_true, y_scores):
        print classification_report(y_true, y_scores, digits = 4)
        tp = 0; fn = 0; fp = 0

        for i in xrange(len(y_true)):
            if y_true[i] == 1:
                if y_scores[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_scores[i] == 1:
                    fp += 1

        try:
            precision = float(tp)/(tp+fn)
            recall = float(tp)/(tp + fp)
            f1 = 2*precision*recall/(precision + recall)
        except Exception, ex:
            print ex

        print "precision: ", precision
        print "recall: ", recall
        print "f1: ", f1


    #通过改变阈值，计算最优f1，yscores为回归概率
    def precisionRecall(self, y_true, y_scores):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        try:
            max_precision = 0.0;max_recall = 0.0;max_f1 = 0.0; t = 0.0
            for i in range(0,len(precision)):
                f1 = 2*precision[i]*recall[i]/(precision[i] + recall[i])
                if f1 > max_f1:
                    max_f1 = f1
                    max_precision = precision[i]
                    max_recall = recall[i]
                    t = thresholds[i]
            print "thresholds: ", t
            print "precision: ", max_precision
            print "recall: ", max_recall
            print "f1: ", max_f1

        except Exception,ex:
            print "error in precisionRecall function"
            print ex

    def auc(self, y_true, y_scores):

        print "auc: ", roc_auc_score(y_true, y_scores)

    def logLoss(self, y_true, y_scores):
        print "logloss: ", log_loss(y_true, y_scores)

    def classificationValidation(self, y_true, y_scores):
        self.precisionRecall(y_true, y_scores)

    def allValidation(self, y_true, y_scores):
        self.precisionRecall(y_true, y_scores)
        self.auc(y_true, y_scores)
        self.logLoss(y_true, y_scores)
