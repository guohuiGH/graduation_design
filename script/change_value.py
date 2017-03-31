#!/usr/bin/env python
# encoding: utf-8

import math, sys, os

class ChangeData:
    def __init__(self):
        self.mapping_key = dict()

        pass

    def getMapping(self):
        with open("../data/train_weekly_feature_1", "r") as rf:
            lines = [line.strip().split("\t")[0] for line in rf.readlines()]
            i = 1
            for line in lines:
                if line not in self.mapping_key:
                    self.mapping_key[line] = str(i)
                    i += 1

    #生成共同用户的label
    def generateTestLabel(self):
        with open("../data/test_user_label") as rf, open("../data/test_user_old_label", "w+") as wf:
            lines = [line.strip().split("\t") for line in rf.readlines()]
            for line in lines:
                if line[0] in self.mapping_key:
                    wf.write("\t".join(line) + "\n")
    #改变用户id
    def merge(self, tag, input_name, output_name, new_user_name):
        all_data = dict()
        new_user = dict()
        counter = 1

        for i in range(1, 5):
            if tag == "train":
                file_name = input_name + str(i)
                file_output_name = output_name + str(i)
                file_new_user_name = new_user_name + str(i)
            else:
                file_name = input_name + str(i+3)
                file_output_name = output_name + str(i+3)
                file_new_user_name = new_user_name + str(i+3)
            with open(file_name, "r") as rf, open(file_output_name, "w+") as wf, open(file_new_user_name, "w+") as wf2:
                lines = [line.strip().split("\t") for line in rf]

                for line in lines:
                    person_info = list()
                    key = line[0]; value = line[1:]
                    #新的用户
                    if key not in self.mapping_key and key not in new_user:
                        new_user[key] = str(len(self.mapping_key) + counter)
                        counter +=1
                        #continue
                    person_info = value[:34]
                    person_info.extend(value[34].split(" "))
                    person_info.extend(value[35].split(" "))
                    #调整数值范围,规范化，采用log,和*


                    if key in self.mapping_key:
                        wf.write(self.mapping_key[key] + "\t" + "\t".join(person_info) + "\n")
                    if key in self.mapping_key:
                        wf2.write(self.mapping_key[key] + "\t" + "\t".join(person_info) + "\n")
                    else:
                        wf2.write(new_user[key] + "\t" + "\t".join(person_info) + "\n")


    #log 计算
    def normal_log(self, x, tag):
        if tag == "time":
            base_line = 16
        elif tag == "interaction":
            base_line = 12
        elif tag == "interaction_login":
            base_line = 9
        elif tag == "total_time":
            base_line = 19
        elif tag == "rate":
            base_line = 9
        elif tag == "average":
            base_line = 16
        elif tag == "total_login":
            base_line = 15
        elif tag == "login_rate":
            base_line = 10
        elif tag == "average_login":
            base_line = 14
        elif tag == "total_interaction":
            base_line = 14
        elif tag == "average_interaction_rate":
            base_line = 10
        elif tag == "gap_average_time":
            base_line = 16
        elif tag == "gap_std_time":
            base_line = 100

        #区分比率
        if "rate" not in tag:
            #加1是为了明显区分0,1次数
            x = float(x) + 1
            if x >= 0.0:
                temp = int(math.log(x)/math.log(2))
                temp = temp if temp < base_line else base_line
                return str(temp)
            else:
                print x
                return str(0)
        else:
            if float(x) > 0.0:
                temp = int(math.log(float(x))/math.log(2))
                temp = temp if temp < base_line else base_line
                #5为防止负数
                return str(temp + 6)
            else:
                return str(0)

    #改变值
    def change_value(self, tag, input_name, output_name):
        all_data = dict()
        new_user = dict()
        counter = 1

        for j in range(1, 5):
            if tag == "train":
                file_name = input_name + str(j)
                file_output_name = output_name + str(j)

            else:
                file_name = input_name + str(j+3)
                file_output_name = output_name + str(j+3)

            with open(file_name, "r") as rf, open(file_output_name, "w+") as wf:
                lines = [line.strip().split("\t") for line in rf]

                for line in lines:
                    person_info = list()
                    key = line[0]; value = line[1:]
                    #新的用户
                    if key not in self.mapping_key and key not in new_user:
                        new_user[key] = str(len(self.mapping_key) + counter)
                        counter +=1

                    #调整数值范围,规范化，采用log2,和*

                    for i in range(0,len(value)):
                        #登录时间用对数2为底分级最高16
                        if 0 <= i < 7:
                            person_info.append(self.normal_log(value[i], "time"))

                        #每天互动次数,用对数2为底分级最高12
                        if 7 <= i < 14:
                            person_info.append(self.normal_log(value[i], "interaction"))

                        #每天登录登出次数,用对数2为底分级最高9
                        if 14 <= i < 21:
                            person_info.append(self.normal_log(value[i], "interaction_login"))

                        #登录登出，互动时间,用对数2为底分级,最高19
                        if 21 <= i < 23:
                            person_info.append(self.normal_log(value[i], "total_time"))

                        #比率,用对数2为底分级,最高10
                        if i == 23:
                            person_info.append(self.normal_log(float(value[i])*1000, "rate"))

                        #平均时间,用对数2为底分级,最高16
                        if 24 <= i < 26:
                            person_info.append(self.normal_log(value[i], "average"))

                        #总的登录登出次数,用对数2为底分级,最高15
                        if 26 <= i < 28:
                            person_info.append(self.normal_log(value[i], "total_login"))

                        #登录的比率,最高10
                        if i == 28:
                            person_info.append(self.normal_log(float(value[i])*100, "login_rate"))

                        #平均登录登出次数,用对数2为底分级,最高14
                        if 29 <= i < 31:
                            person_info.append(self.normal_log(value[i], "average_login"))

                        #总的互动登出次数,用对数2为底分级,最高14
                        if 31 <= i < 33:
                            person_info.append(self.normal_log(value[i], "total_interaction"))

                        #平均互动比率,用对数2为底分级,最高10
                        if i == 33:
                            person_info.append(self.normal_log(float(value[i])*100, "average_interaction_rate"))

                        #7天,28天时间间隔均值,用对数2为底分级,最高16
                        if i == 34 or i == 36:
                            person_info.append(self.normal_log(value[i], "gap_average_time"))

                        #7天，28天时间间隔方差,用对数2为底分级,
                        if i == 35 or i == 37:
                            person_info.append(self.normal_log(value[i], "gap_std_time"))

                    wf.write(str(key) + "\t" + "\t".join(person_info) + "\n")

if __name__ == "__main__":
    changeData = ChangeData()
    changeData.getMapping()
    changeData.generateTestLabel()
    changeData.merge("train", "../data/train_weekly_feature_", "../data/train/train_weekly_feature_", "../data/train/train_new_user_feature")
    changeData.merge("test", "../data/test_weekly_feature_", "../data/test/test_weekly_feature_", "../data/test/test_new_user_feature_")

    changeData.change_value("train", "../data/train/train_weekly_feature_", "../data/train/train_weekly_int_feature_")
    changeData.change_value("test", "../data/test/test_weekly_feature_", "../data/test/test_weekly_int_feature_")
    changeData.change_value("test", "../data/test/test_new_user_feature_", "../data/test/test_new_user_int_feature_")


