# -*- coding: utf-8 -*-
'''
    Function: 定义各类网络，并进行训练
        Categary: 
            1. 网络，超参数接口
            2. 定义训练函数，评价函数
'''
import torchvision.models as models
from net import SSNet
def paraWC(model_class,batch_size,EPOCH,lr):
    if model_class == 'resnet':
        net = models.resnet18()
    if model_class == 'alexnet':
        net = models.alexnet()
    if model_class == 'vgg':
        net = models.vgg16()
    if model_class == 'googlenet':
        net = models.googlenet()
    if model_class == 'densenet':   
        net = models.densenet161()
    if model_class == 'shufflenet':
        net = models.shufflenet_v2_x1_0()
    if model_class == 'ours':
        net = SSNet()
    return net,batch_size,EPOCH,lr