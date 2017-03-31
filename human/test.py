#!/usr/bin/env python
# encoding: utf-8
# File Name: test.py
# Author: guohui
# mail: guohui2011915@gmail.com
# Created Time: 2017年03月20日 星期一 00时29分26秒
#########################################################################
#!/bin/bash


with open("./train/y_train.txt") as rf:
    lines = [line.strip() for line in rf.readlines()]
    print set(lines)



