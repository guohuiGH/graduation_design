#!/usr/bin/env python
# encoding: utf-8
import os, sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
class Validation:
    def __init__(self):
        pass

    def calculateF1(self, y_true, y_scores):
        print classification_report(y_true, y_scores, digits = 4)
        print "accuracy_score", accuracy_score(y_true, y_scores)
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

    def top_accuacy(self, y_true, y_classifer, y_proba, index):
        print y_true
        print y_classifer
        print y_proba
        loss_predict_info = dict(); counter = 0; recoder = dict()
        for i, value in enumerate(y_classifer):
            if str(int(value)) == "0":
                loss_predict_info[i] = y_proba[i]
        print len(loss_predict_info)
        for i, key in enumerate(sorted(loss_predict_info.items(), key=lambda x: x[1], reverse=True)):
            if str(int(y_true[key[0]])) == "0":
                counter+=1;

            recoder[i] = counter

        print "accuracy total ", len(recoder), " true number: ", counter, " rate: ", float(counter)/len(recoder)
        for item in index:
            print "accuracy index ", item, " true number: ", recoder[item-1], " rate: ", float(recoder[item-1])/item


    def classificationValidation(self, y_true, y_scores):
        self.precisionRecall(y_true, y_scores, eps=1e-6)

    def allValidation(self, y_true, y_scores):
        #self.precisionRecall(y_true, y_scores)
        self.auc(y_true, y_scores)
        self.logLoss(y_true, y_scores)

if __name__=="__main__":
    y_true = [1,0,0,0,1]
    y_predic = [0,1,0,0,1]
    y_pro = [0.6,0.7,0.8,0.54,0.8]
    v = Validation()
    v.top_accuacy(y_true, y_predic, y_pro,[2,3])
