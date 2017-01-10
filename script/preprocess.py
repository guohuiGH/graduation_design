#!/usr/bin/env python
# encoding: utf-8
# File Name: preprocess.py
# Author: guohui
# mail: guohui2011915@gmail.com
# Created Time: 2017年01月04日 星期三 20时51分31秒
##########################################################################

import os
from multiprocessing import Pool

file_dir_name = ["../data/20161228-1031/", "../data/20161228-1107/", "../data/1114/", "../data/1121/"]
path_name = ["train_data", "test_data"]



def extract_user_data(name):
    all_user = dict()
    def extract(file_name):
        with open(file_name) as f:
            user = [line.strip().split("\t")[0] for line in f.readlines()]
            for u in user:
                all_user[u] = 0 if u not in all_user else all_user[u] + 1

    users_path_name = "/combine/users.txt"
    for dir_name in file_dir_name:
        extract(dir_name + name + "/combine/users.txt")

    common_data = [u for u in all_user if all_user[u] == 3]
    return common_data

def extract_common_data(data_name):

    common_data = extract_user_data(data_name)
    for dir_name in file_dir_name:
        temp_data_path = dir_name + data_name + "/combine/"
        if data_name == "train_data":
            full_data_path = temp_data_path +  "train_data_temp"
            full_user_path = temp_data_path + "users.txt"
            full_common_path = temp_data_path + "train_data_common_132"

        else:
            full_data_path = temp_data_path + "test_data.txt"
            full_user_path = temp_data_path + "users.txt"
            full_common_path = temp_data_path + "test_data_common_132"

        with open (full_data_path) as f1, open(full_user_path) as f2, open(full_common_path,"w+") as f3:
            users = [line.strip().split("\t")[0] for line in f2.readlines()]
            datas = [line.strip() for line in f1.readlines()]
            temp_data = dict()
            print len(common_data)
            for i, value in enumerate(users):
                if i%10000 == 0:
                    print i
                if value in common_data:
                    temp_data[value] = datas[i]
            result = list()
            for key in sorted(temp_data):
                result.append(temp_data[key])
            f3.write("\n".join(result))

#extract_common_data(path_name[0])

#pool = Pool()
#pool.map(extract_common_data, path_name)
#pool.close()
#pool.join()

def extract_recent_user(name):
    path = file_dir_name[3] + name + "/combine/users.txt"
    with open(path) as f1:
        return [line.strip().split("\t")[0] for line in f1.readlines()]

def extract_recent_week_data(data_name):
    recent_user = extract_recent_user(data_name)
    for dir_name in file_dir_name:
        temp_data_path = dir_name + data_name + "/combine/"
        if data_name == "train_data":
            full_data_path = temp_data_path +  "train_data_temp"
            full_user_path = temp_data_path + "users.txt"
            full_recent_path = temp_data_path + "train_data_recent_132"
            full_recent_label_path = temp_data_path + "train_data_recent_label"
        else:
            full_data_path = temp_data_path + "test_data.txt"
            full_user_path = temp_data_path + "users.txt"
            full_recent_path = temp_data_path + "test_data_recent_132"
            full_recent_label_path = temp_data_path + "test_data_recent_label"
        with open(full_data_path) as f1, open(full_user_path) as f2, open(full_recent_path, "w+") as f3:
            #lables = [line.strip().split(" ")[0] for line in f3.readlines()]
            #f4.write("\n".join(lables))
            users = [line.strip().split("\t")[0] for line in f2.readlines()]
            datas = [line.strip() for line in f1.readlines()]
            user_data = dict()
            for i, v in enumerate(users):
                user_data[v] = datas[i]
            result = list()
            print len(recent_user)
            for i, value in enumerate(recent_user):
                if i%10000 == 0:
                    print i
                if value in user_data:
                    result.append(user_data[value])
                else:
                    #扩充成132维或者是201维
                    result.append("0 131:0")


            f3.write("\n".join(result))


pool = Pool()
pool.map(extract_recent_week_data, path_name)
pool.close()
pool.join()
