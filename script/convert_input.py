#!/usr/bin/env python
# encoding: utf-8

import math, sys, os
from sklearn.preprocessing import OneHotEncoder
class Convert:
    def __int__(self):
        self.enc = OneHotEncoder()
        pass

    def getOneHotFit(self):
        one_hot_data = list()
        with open("../data/train/train_feature") as rf:
            one_hot_data = [line.strip().split("\t") for line in rf.readlines()]
        self.enc.fit(one_hot_data)


    def convertDTData(self, tag, input_name, output_name):
        all_data = dict()

        for i in range(1, 5):
            if tag == "train":
                file_name = input_name + str(i)
                output_name = output_name + str(i)
            else:
                file_name = input_name + str(i+4)
                output_name = output_name + str(i+4)
            with open(file_name, "r") as rf, open(output_name, "w+") as wf:
                lines = [line.strip().split("\t") for line in rf]

                wf.write(key + "\t" + "\t".join(person_info) + "\n")

if __name__ == "__main__":
    changeData = ChangeData()
    changeData.merge("train", "../data/train/train_weekly_feature_", "../data/train/onehot_train_weekly_feature_")


