#!/usr/bin/env python
# encoding: utf-8
import sys,math
import numpy as np
import pandas as pd
from sklearn import tree
from multiprocessing import Pool
import pathos.multiprocessing as mp

def out_process(i, x_temp, y_temp, x_test_temp):
    y = np.array(y_temp)[:,i]
    #为了提升速度，用sklearn 开源工具 建立回归树
    clf = tree.DecisionTreeRegressor(max_depth=2)
    clf.fit(x_temp, y)
    predict_train_value = clf.predict(x_temp)

    predict_test_value = clf.predict(x_test_temp)
    return predict_train_value, predict_test_value


class ForestNet:
    def __init__(self, x_num, forest_num, forest_learning_rate, cluster_size, feature_size, iteration, thread_core=2):

        self.forest_num = forest_num
        #初始化forest的y森林层的值
        self.y = list()
        mu,sigma=0,1
        for i in xrange(cluster_size):
            self.y.append([np.random.normal(mu,sigma,self.forest_num) for k in range(x_num)])
        self.y = np.array(self.y)

        #森林学习率,迭代轮数,特征数
        self.learning_rate = forest_learning_rate

        self.cluster_size = cluster_size
        self.feature_size = feature_size

        #收集整个森林的分类器
        self.forest_collection = [[[] for j in range(forest_num)]for i in range(cluster_size)]
        self.forest_out = [0]*cluster_size
        self.forest_train_out = [0]*cluster_size
        self.forest_test_out = [0]*cluster_size
        self.iteration = iteration
        self.thread_core = thread_core
        pass

    #累加更新树值
    def update_f_value(self, y_temp, current_predict):

        return np.add(np.array(y_temp), self.learning_rate*np.transpose(np.array(current_predict)))


    def generate_forest(self, x_train, y_train):
        try:
            tag = 0

            if len(y_train) == 0:
                tag = 1
                y_train = self.y

            x_train = np.array(x_train)
            y_train = np.array(y_train)


            for t in xrange(self.cluster_size):
                #初始化训练集的x,y

                start = t*self.feature_size; end = (t+1)*self.feature_size
                x_temp,y_temp = x_train[:,start:end], y_train[t]

                current_predict = list() # 存储forest的值

                for i in xrange(self.forest_num):
                    y = np.array(y_temp)[:,i]
                    #用sklearn 工具 建立回归树
                    clf = tree.DecisionTreeRegressor(max_depth=3)
                    clf.fit(x_temp, y)
                    predict_value = clf.predict(x_temp)
                    current_predict.append(predict_value)
                    self.forest_collection[t][i].append(clf)
                #不要随机初始化值

                single_cluster = self.forest_train_out[t] if tag != 1 else np.zeros_like(y_temp)
                #single_cluster = self.forest_out[t] if tag != 1 else y_temp

                self.forest_train_out[t] = self.update_f_value(single_cluster, current_predict)

            return self.forest_out
        except Exception, ex:
            print ex
            #raise ValueError("wrong in generate model")

    def forest_test_predict(self, x_test):
        try:
            print("test data generate starting")
            x_test = np.array(x_test)
            length = len(x_test)
            forest_test_out = list()
            for t in xrange(self.cluster_size):
                current_predict = list()
                start = t*self.feature_size; end = (t+1)*self.feature_size
                x_temp = x_test[:,start:end]
                for i in xrange(self.forest_num):
                    sum_clf = np.array([0]*length)
                    for j in xrange(self.iteration):
                        #拿到其中一个分类器预测
                        clf = self.forest_collection[t][i][j]
                        predict_value = clf.predict(x_temp)
                        sum_clf = self.update_f_value(sum_clf, predict_value)
                    current_predict.append(sum_clf)
                forest_test_out.append(np.transpose(np.array(current_predict)))
            return np.array(forest_test_out)
        except Exception, ex:
            print ex


    def thread_tree_model(self, x_temp, y_temp, x_test_temp, current_train_predict, current_test_predict):
        index = [t for t in xrange(self.forest_num)]
        def run_model(i):
            y = np.array(y_temp)[:,i]
            #用开源sklearn 工具 建立回归树
            clf = tree.DecisionTreeRegressor(max_depth=3)
            clf.fit(x_temp, y)
            predict_train_value = clf.predict(x_temp)
            #current_train_predict[i] = predict_train_value

            predict_test_value = clf.predict(x_test_temp)
            #current_test_predict[i] = predict_test_value
            return (predict_train_value, predict_test_value)
        pool = mp.ProcessingPool(self.thread_core)
        temp = pool.map(run_model, index)
        for i, value in enumerate(temp):
            current_train_predict[i] = value[0]
            current_test_predict[i] = value[1]



    #由于gbdt不存在更新参数，为了节省内存，将测试与训练合并
    #sklearn每次训练会记录所有训练数据，非常消耗内存
    def generate_forest_data(self, x_train, y_train, x_test):
        try:
            tag = 0
            #初始化隐藏层特征数据
            if len(y_train) == 0:
                tag = 1
                y_train = self.y

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            length = len(x_test)
            for t in xrange(self.cluster_size):
                #初始化训练集的x,y
                start = t*self.feature_size; end = (t+1)*self.feature_size
                x_temp,y_temp = x_train[:,start:end], y_train[t]
                x_test_temp = x_test[:, start:end]

                current_train_predict = list() # 存储forest的train临时值
                current_test_predict = list() # 存储forest的test临时值
                for i in xrange(self.forest_num):
                    y = np.array(y_temp)[:,i]
                    #用sklearn 工具 建立回归树
                    clf = tree.DecisionTreeRegressor(max_depth=3)
                    clf.fit(x_temp, y)
                    predict_train_value = clf.predict(x_temp)
                    current_train_predict.append(predict_train_value)

                    predict_test_value = clf.predict(x_test_temp)
                    current_test_predict.append(predict_test_value)

                #训练预测值不要随机初始化值
                single_cluster = self.forest_train_out[t] if tag != 1 else np.zeros_like(y_temp)
                self.forest_train_out[t] = self.update_f_value(single_cluster, current_train_predict)

                if tag != 1:
                    self.forest_test_out[t] = self.update_f_value(self.forest_test_out[t], current_test_predict)
                else:
                    self.forest_test_out[t] = self.learning_rate*np.transpose(np.array(current_test_predict))

            return np.array(self.forest_train_out)
        except Exception, ex:
            print ex
            raise ValueError("wrong in generate model")


    def get_train_data(self, x_train, y_train, x_test):
        return self.generate_forest_data(x_train, y_train, x_test)

    def get_test_data(self):
        return self.forest_test_out

    def get_libsvm_train_data(self, x_train, y_train, x_test):
        return self.generate_forest_libsvm_data(x_train, y_train, x_test)


    #由于gbdt不存在更新参数，为了节省内存，将测试与训练合并
    #sklearn每次训练会记录所有训练数据，非常消耗内存
    def generate_forest_libsvm_data(self, x_train, y_train, x_test):
        try:
            tag = 0
            #初始化隐藏层特征数据
            if len(y_train) == 0:
                tag = 1
                y_train = self.y

            y_train = np.array(y_train)
            length = x_test[0].shape
            for t in xrange(self.cluster_size):
                #初始化训练集的x,y
                x_temp,y_temp = x_train[t], y_train[t]
                x_test_temp = x_test[t]


                current_train_predict = [0]*self.forest_num # 存储forest的train临时值
                current_test_predict = [0]*self.forest_num # 存储forest的test临时值

                '''
                #单线程
                for i in xrange(self.forest_num):
                    y = np.array(y_temp)[:,i]
                    #用sklearn 工具 建立回归树
                    clf = tree.DecisionTreeRegressor(max_depth=3)
                    clf.fit(x_temp, y)
                    predict_train_value = clf.predict(x_temp)
                    current_train_predict[i] = predict_train_value

                    predict_test_value = clf.predict(x_test_temp)
                    current_test_predict[i] = predict_test_value

                #多线程并行建树
                '''
                self.thread_tree_model(x_temp, y_temp, x_test_temp, current_train_predict, current_test_predict)
                #pool = mp.ProcessingPool(4)
                #pool = Pool(4)
                #for i in xrange(self.forest_num):
                #    current_train_predict[i], current_test_predict[i] = pool.map(out_process, (i, x_temp, y_temp, x_test_temp))
                    #a = pool.apply_async(out_process, (i, x_temp, y_temp, x_test_temp,))
                    #current_train_predict[i], current_test_predict[i] = pool.apply_async(out_process, (i, x_temp, y_temp, x_test_temp,))
                #pool.close()
                #pool.join()
                #print "hi"
                #exit()
                #for i in xrange(self.forest_num):
                #    current_train_predict[i] = current_train_predict[i].get()
                #    current_test_predict[i] = current_test_predict[i].get()

                #训练预测值不要随机初始化值
                single_cluster = self.forest_train_out[t] if tag != 1 else np.zeros_like(y_temp)
                self.forest_train_out[t] = self.update_f_value(single_cluster, current_train_predict)

                if tag != 1:
                    self.forest_test_out[t] = self.update_f_value(self.forest_test_out[t], current_test_predict)
                else:
                    self.forest_test_out[t] = self.learning_rate*np.transpose(np.array(current_test_predict))
            return np.array(self.forest_train_out)
        except Exception, ex:
            print ex
            raise ValueError("wrong in generate model")





