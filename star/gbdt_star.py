#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np
sys.path.append("../script")
from convert_input import Convert
from validation import Validation
from read_data import DataReader

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

import lightgbm as lgb

class GBDT:
    def __init__(self):
        self.x_train = list()
        self.y_train = list()
        self.x_test = list()
        self.y_test = list()
        pass

    #评价最后预测结果
    def get_result(self, predict_classifier, predict_probability):
        v = Validation()
        v.calculateF1(self.y_test, predict_classifier)
        #v.allValidation(self.y_test, predict_probability)
        #v.top_accuacy(self.y_test, predict_classifier, predict_probability,[1000,10000,50000,100000])

    def XGBC(self, est):
        print "\n" + "*"*20 + "XGBOOST" + "*"*20 + "\n"
        #clf_xgb = XGBClassifier(max_depth=6, learning_rate=0.0125, n_estimators=300, subsample=0.6, colsample_bytree=0.5,seed=4)
        clf_xgb = XGBClassifier(max_depth=3, n_estimators=est, learning_rate=0.001)
        clf_xgb.fit(self.x_train, self.y_train)
        predict_classifier = clf_xgb.predict(self.x_test)
        predict_probability = clf_xgb.predict_proba(self.x_test)[:,0]
        self.get_result(predict_classifier, predict_probability)

    def GBC(self):
        print "\n" + "*"*20 + "NORMAL_GBDT" + "*"*20 + "\n"
        clf_gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1)
        clf_gb.fit(self.x_train, self.y_train)
        #self.x_test = self.x_test.toarray()
        predict_classifier = clf_gb.predict(self.x_test)
        predict_probability = clf_gb.predict_proba(self.x_test)[:,0]
        self.get_result(predict_classifier, predict_probability)

    #lightGBM
    def GBM(self):
        print "\n" + "*"*20 + "GBM" + "*"*20 + "\n"

        #clf_gbm = lgb.LGBMClassifier(max_depth=6, learning_rate=0.0125, n_estimators=300, subsample=0.6, colsample_bytree=0.5,seed=4)
        clf_gbm = lgb.LGBMClassifier(max_depth=3, n_estimators=100)
        clf_gbm.fit(self.x_train, self.y_train)
        predict_classifier = clf_gbm.predict(self.x_test)
        predict_probability = clf_gbm.predict_proba(self.x_test)[:,0]
        self.get_result(predict_classifier, predict_probability)


    def get_data(self,tag):
        c = Convert()
        if tag == "normal":
            (self.x_train, self.y_train, self.x_test, self.y_test) = c.getDTData()
        elif tag == "oneHot":
            (self.x_train, self.y_train, self.x_test, self.y_test) = c.getDTOneHotData()

    #拿到叶子节点值
    def get_leaf(self):
        self.get_data("oneHot")
        n_estimators = 300
        clf_xgb = XGBClassifier(max_depth=4, learning_rate=0.0125, n_estimators=300, subsample=0.6, colsample_bytree=0.7,seed=4)
        #clf_xgb = XGBClassifier(max_depth=4, n_estimators=300)
        clf_xgb.fit(self.x_train, self.y_train)

        leafes_train = list(clf_xgb.apply(self.x_train))
        leafes_test = list(clf_xgb.apply(self.x_test))

        #补充最大值，最小值,将数据one-hot时统一
        max_train = np.array(leafes_train).max()
        min_train = np.array(leafes_train).min()

        max_test = np.array(leafes_test).max()
        min_test = np.array(leafes_test).min()
        max_value = max(max_train, max_test)
        min_value = min(min_train, min_test)
        for i in range(min_value, max_value+1):
            leafes_train.append([i]*n_estimators)

        enc = OneHotEncoder()
        enc.fit(leafes_train)
        #去除补充的值
        leafes_train_feature = enc.transform(leafes_train).toarray()[:-(max_value-min_value+1),:]
        print leafes_train_feature.shape, len(leafes_train)
        return leafes_train_feature, self.y_train, enc.transform(leafes_test).toarray(), self.y_test

    #libsvm 数据格式的数据，新的数据
    def get_libsvm_data(self):
        reader = DataReader()
        self.x_train, self.y_train = reader.temp_read_train_data('../UCR_TS_Archive_2015/StarLightCurves/StarLightCurves_TRAIN')
        self.x_test, self.y_test = reader.temp_read_test_data('../UCR_TS_Archive_2015/StarLightCurves/StarLightCurves_TEST')

        #self.x_train, self.y_train = reader.temp_read_train_human('../human/train/')
        #self.x_test, self.y_test = reader.temp_read_test_human("../human/test/")

        #self.x_train, self.y_train = load_svmlight_file("../data/1121/train_data/combine/train_data_temp")
        #self.x_test, self.y_test = load_svmlight_file("../data/1121/test_data/combine/test_data.txt")
        #print (self.x_train.shape, self.x_test.shape)
        #result = self.x_train[:, :132]
        #dump_svmlight_file(result, self.y_train,"test")

if __name__ == "__main__":
    g = GBDT()
    #g.get_data("oneHot")
    g.get_libsvm_data()
    #g.get_leaf()
    for i in range(10,110,10):
        print i
        g.XGBC(i)
    #g.GBC()
    #g.GBM()
