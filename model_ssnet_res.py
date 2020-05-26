# -*- coding: utf-8 -*-
"""
    Function: 所有Net
    Categary:
        1. SSNet  --- Ours
        2. resnet34, resnet101
        3. AlexNet
        4. VGG
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class SSNet(nn.Module):
    def __init__(self, count=5, roi_layer=1, init_weights=False):
        super(SSNet, self).__init__()
        self.count = count
        self.roi_layer = roi_layer
        self.in_channel = 64
        self.layer1 = nn.Sequential(
            # Conv2d参数：图片的高度，卷积核个数（输出的feature map个数），核的大小，移动的步长(stride)，填充(padding)
            nn.Conv2d(3, roi_layer, kernel_size=3),  # 252
            nn.BatchNorm2d(roi_layer),
            nn.ReLU(inplace=True)  # 去除图像中<0的像素
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 45, kernel_size=7,stride = 2, bias=False),
            nn.BatchNorm2d(45),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # ROI卷积层和原图卷积的汇入
        self.layer3 = nn.Sequential(
            nn.Conv2d(count * roi_layer +45, 64, kernel_size=3,padding=1),  # 64-5+1/2 = 30
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = self._make_layer(BasicBlock, 64, 1)
        self.layer5 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer6 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer7 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.layer_x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),  # 64-5+1/2 = 30
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        print('hello')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    # nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x, ROI, batch_size):
        roiX = self.layer1(x)
        x_batch = []
        for k in range(batch_size):
            x_set = []
            for j in range(self.roi_layer):
                for i in range(self.count):
                    y1, y2, x1, x2 = (ROI[k][i].squeeze(0).float() * 0.9).int()
                    image = F.adaptive_max_pool2d(roiX[k, j][y1:y2, x1:x2].unsqueeze(0), output_size=(62, 62))
                    x_set.append(image.cpu().detach().numpy())
            x_batch = x_set + x_batch
        x_batch = torch.FloatTensor(x_batch).cuda().squeeze(1)
        x_batch = x_batch.view(batch_size, self.roi_layer * self.count, 62, 62)

        # 得到roi_layer*count张图 都是一张图的特征图 要转成tensor
        norX = self.layer2(x)

        x = torch.cat((norX, x_batch), 1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer_x(x)
        #x = self.layer5(x)
        #x = self.layer6(x)
        #x = self.layer7(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return F.softmax(x, dim=1)

