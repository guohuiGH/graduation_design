#!/usr/bin/env python
# encoding: utf-8

import sys, os

class MergeData:
    def __int__(self):
        pass

    def merge(self, tag, input_name, output_name):
        all_data = dict()
        for i in range(1, 5):
            if tag == "train":
                file_name = input_name + str(i)
            else:
                file_name = input_name + str(i+4)
            with open(file_name, "r") as rf:
                lines = [line.strip().split("\t") for line in rf]

