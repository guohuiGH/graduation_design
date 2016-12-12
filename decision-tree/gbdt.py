#!/usr/bin/env python
# encoding: utf-8

import sys
sys.path.append("../script")
from convert_input import Convert
from validation import Validation

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

class GBDT:
    def __init__(self):
        self.x_train = list()
        self.y_train = list()
        self.x_test = list()
        self.y_test = list()
        pass

    def get_result(self, predict_classifier, predict_probability):
        v = Validation()
        v.calculateF1(self.y_test, predict_classifier)
        v.allValidation(self.y_test, predict_probability)

    def XGBC(self):
        print "\n" + "*"*20 + "XGBOOST" + "*"*20 + "\n"
        clf_xgb = XGBClassifier(max_depth=6, learning_rate=0.0125, n_estimators=1000, subsample=0.6, colsample_bytree=0.5,seed=4)
        clf_xgb.fit(self.x_train, self.y_train)
        predict_classifier = clf_xgb.predict(self.x_test)
        predict_probability = clf_xgb.predict_proba(self.x_test)[:,1]
        self.get_result(predict_classifier, predict_probability)

    def GBC(self):
        print "\n" + "*"*20 + "NORMAL_GBDT" + "*"*20 + "\n"
        clf_gb = GradientBoostingClassifier(n_estimators=1000,subsample=0.6,max_depth=5)
        clf_gb.fit(self.x_train, self.y_train)
        predict_classifier = clf_gb.predict(self.x_test)
        predict_probability = clf_gb.predict_proba(self.x_test)[:,1]
        self.get_result(predict_classifier, predict_probability)

    def get_data(self,tag):
        c = Convert()
        if tag == "normal":
            (self.x_train, self.y_train, self.x_test, self.y_test) = c.getDTData()
        elif tag == "oneHot":
            (self.x_train, self.y_train, self.x_test, self.y_test) = c.getDTOneHotData()


if __name__ == "__main__":
    g = GBDT()
    g.get_data("oneHot")
    g.XGBC()
    g.GBC()

