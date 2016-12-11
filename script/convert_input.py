#!/usr/bin/env python
# encoding: utf-8

import math, sys, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,GradientBoostingClassifier,GradientBoostingRegressor)
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from scipy import io
import validation
class Convert:
    def __init__(self):
        self.enc = OneHotEncoder()
        self.train_ylabel = list()
        self.test_ylabel = list()

        self.getLabel()
        pass


    #获得标签数据
    def getLabel(self):
        with open("../data/train_user_label") as rf:
            self.train_ylabel = [int(line.strip().split("\t")[1]) for line in rf.readlines()]

        with open("../data/test_user_old_label") as rf:
            self.test_ylabel = [int(line.strip().split("\t")[1]) for line in rf.readlines()]

    #非one-hot的gbdt的输入数据
    def convertDTData(self, tag, input_name):
        all_data = dict()

        result = list()
        for i in range(1, 5):
            if tag == "train":
                file_name = input_name + str(i)
            else:
                file_name = input_name + str(i+3)
            #横向拼接几天的数据
            with open(file_name, "r") as rf:
                lines = np.array([line.strip().split("\t") for line in rf.readlines()])
                if i == 1:
                    result = lines
                else:
                    #去掉冗余列
                    temp = lines
                    #temp = lines[:,1:2]
                    #drop_column = [0,23,26,28,31,33,37,38]
                    #for j in xrange(2,39):
                    #    if j not in drop_column:
                    #        temp = np.column_stack((temp, lines[:,i:i+1]))
                    result = np.column_stack((result, temp))
        #print len(result), len(result[0])
        result = [[float(x) for x in row] for row in result]
        return result

    def getDTData(self):
        train_xlabel = self.convertDTData("train", "../data/train/train_weekly_int_feature_")
        test_xlabel = self.convertDTData("test", "../data/test/test_weekly_int_feature_")

        return (train_xlabel, self.train_ylabel, test_xlabel, self.test_ylabel)

    #生成pandas的索引目录
    def getIndex(self, line,index):
        new_index = list()

        #new_index.append(str(index)+"_id")
        for i in xrange(len(line)):
            if 0 <= i < 7:
                new_index.append(str(index) + "_day_time_" + str(i+1))
            elif 7 <= i < 14:
                new_index.append(str(index) + "_day_interaction_" + str(i-6))
            elif 14 <= i < 21:
                new_index.append(str(index) + "_day_login" + str(i-13))
            elif i == 21:
                third = ["s_total_time", "all_time", "s_time_rate", "s_time_average", "all_average"]
                forth = ["s_total_login","all_login", "s_login_rate", "s_login_average", "all_login_average", "s_interaction", "all_interaction", "interaction_rate"]
                fifth = ["s_gap_average", "s_gap_std", "all_gap_average", "all_gap_std"]
                temp = third
                temp.extend(forth); temp.extend(fifth)
                for tag in temp:
                    new_index.append(str(index) + "_" + tag)
                pass
        return new_index


    #转换成one-hot的数据
    def convertDTOneHot(self, input_name, input_name2):
        data = list()
        for i in xrange(1,5):

            file_name = input_name + str(i)
            file_name2 = input_name2 + str(i+3)
            #训练集和测试集合并做one-hot，防止训练集合测试集分开one-hot导致特征数量不一致
            with open(file_name, "r") as rf1, open(file_name2,"r") as rf2:
                try:
                    #训练集、测试集合并,并且去除第一列id数据，数值太大，爆内存
                    lines = [line.strip().split("\t") for line in rf1.readlines()]
                    lines2 = [line.strip().split("\t") for line in rf2.readlines()]
                    lines = np.row_stack((lines, lines2))
                    lines = np.array(lines)[:,1:]
                    #转化为dataframe结构,然后转换为0,1
                    column_index = self.getIndex(lines[0],i)
                    df = pd.DataFrame(lines, columns=column_index)
                    #先建立一个dataframe类型的索引，再拼接所有数据
                    if i == 1:
                        data = pd.get_dummies(df["1_day_time_1"], prefix="1_day_time_1")
                    else:
                        for f in column_index:
                            if f != "1_id" and f != "1_day_time_1":
                                data = pd.concat([data, pd.get_dummies(df[f],prefix=f)], axis=1)
                except Exception, ex:
                    print ex
        #print len(data)
        return (data[:178172], data[178172:])

    def getDTOneHotData(self):
        train_xlabel, test_xlabel = self.convertDTOneHot("../data/train/train_weekly_int_feature_", "../data/test/test_weekly_int_feature_")
        return train_xlabel, self.train_ylabel,test_xlabel, self.test_ylabel

    def getDTOneHotLabel(self):

        #train_xlabel = self.convertDTOneHot("train", "../data/train/train_weekly_int_feature_")
        #test_xlabel = self.convertDTOneHot("test", "../data/test/test_weekly_int_feature_")
        train_xlabel, test_xlabel = self.convertDTOneHot("../data/train/train_weekly_int_feature_", "../data/test/test_weekly_int_feature_")
        clf = GradientBoostingClassifier(n_estimators=100,subsample=0.7)
        clf.fit(train_xlabel, self.train_ylabel)
        y_target = clf.predict(test_xlabel)
        #y_target2 = clf.predict_proba(test_xlabel)
        #target = [str(i) for i in y_target]
        #print "\n".join(target)
        print classification_report(self.test_ylabel, y_target)

        #v = validation.Validation()
        #v.calculateF1(self.test_ylabel, y_target)
        #print f1_score(self.test_ylabel, y_target)
        #v.allValidation(self.test_ylabel,y_target2[:,1])
        #v.logLoss(self.test_ylabel, y_target)
if __name__ == "__main__":
    convert = Convert()
    convert.getDTOneHotLabel()




