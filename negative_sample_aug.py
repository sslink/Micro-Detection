# -*- coding: utf-8 -*-
"""
    Function: 对正样本：负样本 = 30：1 数据划分，实现不平衡样本 
    Easy Strategy: 
        1. Augmentation  --- image classic augmentation
        2. Under-sampling --- Random(temp) --> KNN
        3. Generate M train set and test set in single fold, totally K-fold --- package: scikit-learn
"""
import numpy as np
import os
import cv2
import copy
import random
from torchvision import transforms as tfs
from sklearn.model_selection import train_test_split, KFold
from preprocess import resizeImage,cal_ROI

"""
    Function: 读入所有的数据和对应的标签列，子文件名 1/0 P/N 隐/健
    Input: filefolder name
    Output: (image list, label list)
"""
def dataIO(file_name):
    src_img = []
    label = []
    for maindir, subdir, file_name_list in os.walk(file_name):
        for dir in subdir:
            for filename in os.listdir(file_name+dir):
                img = cv2.imread(file_name+dir+'/'+filename)    
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度
                src_img.append(gray)
            label.extend([int(dir) for i in range(len(os.listdir(file_name+dir)))])
    return src_img,label
"""
    Function: 单个图片增强, 8种方法 318*9 = 2862 
    Input: (image, the way of aug)
    Output: augmented image
"""
im_aug = tfs.Compose([tfs.RandomCrop(37),tfs.Resize(40)]) #随机切割并resize
def augDim(img,flag=0):
    #随机切割并resize
    if flag == 0:
        img = np.rot90(img, 1)
    if flag == 1:
        img = np.rot90(img, 2)
    if flag == 2:
        img = np.rot90(img, -1)
    if flag == 3:
        img = np.flipud(img) #反转
    if flag == 4:
        img = np.fliplr(img)
    if flag == 5:
        img = np.rot90(img, 1) 
        img = np.flipud(img)
    if flag == 6:
        img = np.rot90(img, 1)
        img = np.fliplr(img)
    if flag == 7:
        img = tfs.ToPILImage()(img) # array to PIL
        img = np.array(im_aug(img)) # PIL to array
    return img
"""
    Function: 拿出所有样本的20% 作为test，共五份   
    Output: 5 set of (train,test) pairs --- list,tuple
"""
def dataKfold(src_img,src_label,K=5):
    src_img = np.array(src_img)
    src_label = np.array(src_label)
    train_X,test_X, train_Y, test_Y = train_test_split(src_img,src_label,\
                                                       test_size = 0.2,\
                                                       random_state = 1,\
                                                       shuffle=1)       
    kf = KFold(n_splits=K, shuffle=True, random_state=5)
    train_val_pairs = dict(key=['train','val'])
    train_set = []
    val_set = []
    for train_index, val_index in kf.split(train_X):
        train_set.append((train_X[train_index],train_Y[train_index]))
        val_set.append((train_X[val_index],train_Y[val_index]))
    train_val_pairs['train'] = train_set
    train_val_pairs['val'] = val_set
    test_set = (test_X,test_Y)
    return train_val_pairs,test_set
"""
    Function: 一个不均衡的train test set分成几个小的数据对
    Input: x_p, x_n
    Output: tuple list
"""
def underSample(x_m,y_m,x_l,y_l):
    ratio = len(y_m)//len(y_l)+1

    data_l = [[a,b] for a,b in zip(x_l,y_l)]
    data_m = [[a,b] for a,b in zip(x_m,y_m)]
    random.shuffle(data_m)
    x_set = []
    y_set = []    
    for i in range(ratio):
        new_data = data_m[i::ratio]+data_l
        x_set.append([a[0] for a in new_data])
        y_set.append([a[1] for a in new_data])
    return (x_set,y_set)
if __name__ == '__main__':
    src_img, src_label = dataIO('./Data/')
    #1.正样本增广
    num0 = 10946
    K = 5
    temp_X = copy.deepcopy(src_img[:num0])
    temp_Y = copy.deepcopy(src_label[:num0])
    positve_img = src_img[num0:]
    positve_label = src_label[num0:]
    aug_positve_img = copy.copy(positve_img)  #浅拷贝
    aug_positve_label = copy.copy(positve_label) 
    for i in range(8):
        new = [augDim(img,i) for img in positve_img]   
        aug_positve_img.extend(new)
        aug_positve_label.extend(positve_label) 

    #2.对P,N分别划分
    train_val_N,test_N = dataKfold(resizeImage(temp_X),temp_Y)
    train_val_P,test_P = dataKfold(resizeImage(aug_positve_img),aug_positve_label)
    #3.对每一个fold下采样重新划分 一个flod生成4组train,val,一共20组
    new_train = [];new_val=[];new_test=[];
    for j in ['train','val']:
        for i in range(K): #跟kfold折数相关
            temp_neg_x = copy.deepcopy(train_val_N[j][i][0])
            temp_neg_y = copy.deepcopy(train_val_N[j][i][1])
            temp_pos_x = copy.deepcopy(train_val_P[j][i][0])
            temp_pos_y = copy.deepcopy(train_val_P[j][i][1])
            if j == 'train':
                new_train.append(underSample(temp_neg_x,temp_neg_y,temp_pos_x,temp_pos_y)) #正负样本merged
            if j == 'val':
                new_val.append(underSample(temp_neg_x,temp_neg_y,temp_pos_x,temp_pos_y))
    new_test = underSample(test_N[0],test_N[1],test_P[0],test_P[1])     
    
    # 预处理和ROI信息 ID: fold/x-y/under-sample num 4    所以构建4个弱分类器
    print('Start fold set')
    fold_train_set = [] # K组 每组4个(train,val)
    for i in range(K):
        print(i)
        train_val_set = []
        for j in range(len(new_train[i][0])):
            print(j)
            data_train = dict(key=['ROI','X_dst','X_org'])
            data_val = dict(key=['ROI','X_dst','X_org'])
            data_train['ROI'],data_train['X_dst'],data_train['X_org'] = cal_ROI(new_train[i][0][j],new_train[i][1][j])
            data_val['ROI'],data_val['X_dst'],data_val['X_org'] = cal_ROI(new_val[i][0][j],new_val[i][1][j])
            
            train_val_set.append((data_train,data_val))
        fold_train_set.append(train_val_set)  
    fold_test_set = []
    for i in range(len(new_test[0])):
        data_test = dict(key=['ROI','X_dst','X_org'])
        data_test['ROI'],data_test['X_dst'],data_test['X_org'] = cal_ROI(new_test[0][i],new_test[1][i])
        fold_test_set.append(data_test)
    
    print('Start Loading')
    flod_set = {'train':fold_train_set,'test':fold_test_set}
    np.save('dataset_single.npy', flod_set)

