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
batch_size = 2


class SSNet(nn.Module):
	def __init__(self,count=5,roi_layer=3):
		super(SSNet, self).__init__()
		self.count = count
		self.layer1 = nn.Sequential(
			# Conv2d参数：图片的高度，卷积核个数（输出的feature map个数），核的大小，移动的步长(stride)，填充(padding)
            nn.Conv2d(1, roi_layer, kernel_size=5), #252
            nn.BatchNorm2d(roi_layer),
            nn.ReLU(inplace=True)  #去除图像中<0的像素
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
		self.layer2 = nn.Sequential(
			nn.Conv2d(1, 20, kernel_size=5), 
			nn.BatchNorm2d(20),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=4, stride=2)
			)
		# ROI卷积层和原图卷积的汇入
		self.layer3 = nn.Sequential(
			nn.Conv2d(count*3+20 , 50, kernel_size=5), # 64-5+1/2 = 30
			nn.BatchNorm2d(50),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
			)
		self.layer4 = nn.Sequential(
			nn.Conv2d(50 , 25, kernel_size=5), # 30-5+1/2
			nn.BatchNorm2d(25),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
			)
		self.fc = nn.Sequential(
			nn.Linear(25 * 28 * 28, 1024),   
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.ReLU(inplace=True),
			)
	def forward(self, x, ROI,roi_layer):
		roiX = self.layer1(x)

		x_batch = []
		for k in range(batch_size):
			x_set = []
			for j in range(roi_layer):
				for i in range(self.count):
					y1,y2,x1,x2 = ROI[k][i].squeeze(0)
					y1 = int(y1.float()*0.9);y2 = int(y2.float()*0.9);x2 = int(x2.float()*0.9);x1 = int(x1.float()*0.9);
					image = F.adaptive_max_pool2d(roiX[k,j][y1:y2,x1:x2].unsqueeze(0), output_size=(125, 125))
					x_set.append(image.cpu().detach().numpy())
			x_batch = x_set + x_batch
		x_batch = torch.FloatTensor(x_batch).cuda().squeeze(1)
		x_batch = x_batch.view(batch_size,15,125,125)
		
		#得到roi_layer*count张图 都是一张图的特征图 要转成tensor
		norX = self.layer2(x)
		x = torch.cat((norX, x_batch),1)
		x = self.layer3(x)
		x = self.layer4(x)
		print(x.shape)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return F.softmax(x, dim=1)

