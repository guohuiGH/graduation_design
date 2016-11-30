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
                    person_info = value[:28]

                    #调整数值范围,规范化，采用log,和*
                    for i in range(28,len(value)):
                        if 28 <= i < 31 or i == 33:
                            if value[i] != '0':
                                temp = str(int(math.log(float(value[i])*100000)/math.log(2)))
                                person_info.append(temp)
                            else:
                                person_info.append(str(int(value[i])))
                        if 31 <= i < 33:
                            person_info.append(value[i])
                        if 34 <= i < 36:
                            #aver = value[i].split(" ")[0]
                            #vari = value[i].split(" ")[1]
                            for t in value[i].split(" "):
                                try:
                                    if float(t) > 0.0:
                                        person_info.append(str(int(math.log(float(t))/math.log(2))))
                                    else:
                                        person_info.append(str(int(float(t))))
                                except:
                                    print line

                    if key in self.mapping_key:
                        wf.write(self.mapping_key[key] + "\t" + "\t".join(person_info) + "\n")
                    if key in self.mapping_key:
                        wf2.write(self.mapping_key[key] + "\t" + "\t".join(person_info) + "\n")
                    else:
                        wf2.write(new_user[key] + "\t" + "\t".join(person_info) + "\n")

if __name__ == "__main__":
    changeData = ChangeData()
    changeData.getMapping()
    #changeData.merge("train", "../data/train_weekly_feature_", "../data/train/train_weekly_feature_")
    changeData.merge("test", "../data/test_weekly_feature_", "../data/test/test_weekly_feature_", "../data/test/test_new_user_feature_")


