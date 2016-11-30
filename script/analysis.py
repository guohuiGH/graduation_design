#coding:utf-8
import os
import pandas as pd
from matplotlib import pyplot

def drawBar():
    with open("../data/train_feature") as rf:
        lines = [line.strip().split("\t") for line in rf.readlines()]
    length = len(lines[0])
    for i in range(1, length):
        data = [float(line[i]) for line in lines]
        pd_data = pd.DataFrame(data)
        print "feaure " + str(i)
        print pd_data.describe() 
        xtricks = set(data)
        ytricks = dict()
        if len(xtricks) > 100:
            continue
        for value in data:
            ytricks[value] = ytricks.get(value, 0) + 1
        pyplot.pie([ytricks.get(label, 0) for label in xtricks], labels = xtricks, autopct='%1.1f%%')
        #pyplot.bar(range(len(xtricks)), [ytricks.get(lable,0) for lable in xtricks] )
        pyplot.title("figure of feature " + str(i))
        #pyplot.xticks(range(len(xtricks)), xtricks)
        #pyplot.ylabel('Frequency')
        pyplot.show()
        #break
drawBar()


