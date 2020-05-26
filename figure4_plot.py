# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:54:05 2020

@author: admin
"""

import torch
import os 
import numpy as np
import cv2
from preprocess import cal_ROI
from model_ssnet_res import SSNet

#输入一个图像 给出结果

# 1. 加载模型
model = torch.load('./model/SSnet_256_res4_v2_model.pkl').cuda()

# 2. 加载图像和文件名, 生成ROI和Label
def dataIO(file_name):
    src_img = []
    label = []
    name_list = []
    for maindir, subdir, file_name_list in os.walk(file_name):
        for dir in subdir:
            for filename in os.listdir(file_name+dir):
                img = cv2.imread(file_name+dir+'/'+filename)    
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度
                src_img.append(gray)
                name_list.append(filename)
            label.extend([int(dir) for i in range(len(os.listdir(file_name+dir)))])
    return src_img,label,name_list

num0 = 10946
src_img, src_label, src_name = dataIO('./Data/')
new_ROI5,_,src_image,del_list = cal_ROI(src_img[num0:],src_label[num0:])
# 3. 输出结果
prediction = model(src_image.cuda(),new_ROI5.cuda(),1)
# 4. loop 找到分类成错误的文件名

