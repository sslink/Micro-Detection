# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:19:44 2020

@author: admin
"""
import copy
import random
import numpy as np

def underSample(x_m,y_m,x_l,y_l):
    ratio = len(y_m)//len(y_l)
    min_num = ratio +1
    min_lenth = int(len(y_m)/min_num)
    lenth = len(y_m)
    print(lenth,min_lenth,min_num)
    x_y_set = []
    rest_x = copy.copy(x_m)
    rest_y = copy.copy(y_m)
    for i in range(ratio):
        slice = random.sample(np.array(range(lenth)).tolist(), min_lenth)
        new_x = rest_x[slice]
        new_y = rest_y[slice]
        x_y_set.append((new_x,new_y))
        rest_x = np.delete(rest_x,slice)
        rest_y = np.delete(rest_y,slice)
        lenth = lenth-min_lenth
        print(lenth)
        print(len(rest_x))
    x_y_set.append((rest_x,rest_y))
    return x_y_set


x_m = np.array(range(10950))
y_m = x_m*2

x_l = np.array(range(2819))
y_l = x_l*2

a = underSample(x_m,y_m,x_l,y_l)

