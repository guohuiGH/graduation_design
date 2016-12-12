#!/usr/bin/env python
# encoding: utf-8

import sys
sys.path.append("../script")
from convert_input import Convert
from validation import Validation

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

class Model:
    def __init__(self):
        self.x_train = list()
        self.y_train = list()
        self.x_test = list()
        self.y_test = list()
        self.classifier = {"MLP" : MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(5, 2), random_state=1),
                            "Logistic Regression":  LogisticRegression(penalty='l1',max_iter=1000)}
        pass


                            #"RBF SVM" : SVC(gamma=2, C=1),
    def get_result(self, predict_classifier, predict_probability):
        v = Validation()
        v.calculateF1(self.y_test, predict_classifier)
        v.allValidation(self.y_test, predict_probability)

    def MLP(self):
        print "\n" + "*"*20 + "MLP" + "*"*20 + "\n"
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(5, 2), random_state=1)
        mlp.fit(self.x_train, self.y_train)
        predict_classifier = mlp.predict(self.x_test)
        predict_probability = mlp.predict_proba(self.x_test)[:,1]
        self.get_result(predict_classifier, predict_probability)

    def model(self):
        try:
            for name, classifier in self.classifier.iteritems():
                print "\n" + "*"*20 + name + "*"*20 + "\n"
                clf = classifier
                clf.fit(self.x_train, self.y_train)
                predict_classifier = clf.predict(self.x_test)
                predict_probability = clf.predict_proba(self.x_test)[:,1]
                self.get_result(predict_classifier, predict_probability)
        except Exception, ex:
            print ex


    def get_data(self,tag):
        c = Convert()
        if tag == "normal":
            (self.x_train, self.y_train, self.x_test, self.y_test) = c.getDTData()

        elif tag == "oneHot":
            (self.x_train, self.y_train, self.x_test, self.y_test) = c.getDTOneHotData()


if __name__ == "__main__":
    m = Model()
    m.get_data("oneHot")

    m.model()

