import sys
import numpy as np

class DataReader:
    def __init__(self,d = 152, n = 2):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.dim = d
        self.n_classes = n

    def read_train_data(self,path):
        datas = np.loadtxt(path,delimiter = ",")
        self.train_x = datas[:,1:]
        self.train_x = self.train_x.reshape((self.dim,self.train_x.shape[0],self.train_x.shape[1]/self.dim))
        self.train_y = datas[:,0:1]
        #self.train_y = [1 if item == -1 else 2 for item in self.train_y]
        temp = []
        for item in self.train_y:
            tmp = [0] * self.n_classes
            tmp[int(item)-1] = 1
            temp.append(tmp)
        self.train_y = np.array(temp)
        return self.train_x,self.train_y

    def read_test_data(self,path):
        datas = np.loadtxt(path,delimiter = ",")
        self.test_x = datas[:,1:]
        self.test_x = self.test_x.reshape((self.dim,self.test_x.shape[0],self.test_x.shape[1]/self.dim))
        self.test_y = datas[:,0:1]
        #self.test_y = [1 if item == -1 else 2 for item in self.test_y]
        temp = []
        for item in self.test_y:
            tmp = [0] * self.n_classes
            tmp[int(item)-1] = 1
            temp.append(tmp)
        self.test_y = np.array(temp)
        return self.test_x, self.test_y


    def temp_read_train_data(self,path):
        datas = np.loadtxt(path,delimiter = ",")
        self.train_x = datas[:,1:]

        self.train_y = datas[:,0:1]


        return self.train_x,self.train_y

    def temp_read_test_data(self,path):
        datas = np.loadtxt(path,delimiter = ",")
        self.test_x = datas[:,1:]

        self.test_y = datas[:,0:1]

        return self.test_x, self.test_y

    def read_train_human(self,path):
        datas = np.loadtxt(path + "x_train.txt", delimiter = " ")
        self.train_x = datas.reshape((self.dim, datas.shape[0], datas.shape[1]/self.dim))
        self.train_y = [line.strip() for line in open(path + "y_train.txt")]

        temp = []
        for item in self.train_y:
            tmp = [0] * self.n_classes
            tmp[int(item)-1] = 1
            temp.append(tmp)
        self.train_y = np.array(temp)
        return self.train_x, self.train_y

        pass
    def read_test_human(self,path):
        datas = np.loadtxt(path + "x_test.txt",delimiter = " ")
        self.test_x = datas.reshape((self.dim, datas.shape[0], datas.shape[1]/self.dim))
        self.test_y = [line.strip() for line in open(path + "y_test.txt")]
        temp = []
        for item in self.test_y:
            tmp = [0] * self.n_classes
            tmp[int(item)-1] = 1
            temp.append(tmp)
        self.test_y = np.array(temp)
        return self.test_x, self.test_y

    def temp_read_train_human(self,path):
        self.train_x = np.loadtxt(path + "x_train.txt", delimiter = " ")

        self.train_y = [line.strip() for line in open(path + "y_train.txt")]

        return self.train_x, np.array(self.train_y)

    def temp_read_test_human(self,path):
        self.test_x = np.loadtxt(path + "x_test.txt",delimiter = " ")

        self.test_y = [line.strip() for line in open(path + "y_test.txt")]

        return self.test_x, np.array(self.test_y)
