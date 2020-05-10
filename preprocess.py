# -*- coding: utf-8 -*-
"""
    Function: 得到每张图片5个ROI的坐标，以及预处理后的图片
    Easy Strategy: 
        1. Preprocess  --- img->median_filtering->haar->resize->normalized->threshold
        2. ROI --- package: selectivesearch
"""
import cv2
import pywt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
from torchvision import transforms as tfs
import torch
###################################################################################################
    #ROI提取
###################################################################################################
"""
    读入图像，做canny变换，返回一个表示边缘位置的list
"""
def _get_edgeinfo_(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)
    canny = cv2.Canny(image, 150, 250)
    edges = []
    for y in range(canny.shape[0]):
        for x in range(canny.shape[1]):
            if canny[y][x] > 250:
                edges.append([x,y])
                #print(x,y)
    return edges
"""
    做SelectiveSearch检测，参数为scale=100，sigma=0.1, min_size=10, 控制r['size']<=120
    返回一组框信息
"""
def _get_ssinfo_(image):
    # 单通道 to 三通道
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    img_lbl, regions = selectivesearch.selective_search(image, scale=100, sigma=0.1, min_size=10)
    # 计算一共分割了多少个原始候选区域
    temp = set()
    for i in range(img_lbl.shape[0]):
        for j in range(img_lbl.shape[1]):
            temp.add(img_lbl[i, j, 3])
    # 创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
    candidates = set() # 注意，set只能遍历
    for r in regions:
        x, y, w, h = r['rect']
        # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
        if r['size'] > 120:
            continue
        # 排除扭曲的候选区域边框  即只保留近似正方形的
        x,y,w,h = r['rect']
        if w == 0 or h == 0:
            continue
        try:
            if w / h < 0.25 or h / w < 0.25:
                continue
        except ZeroDivisionError:
            print(1)
            continue
        candidates.add(r['rect'])
    return candidates
"""
    画出selectedsearch选出的候选框
    Input: 对应顺序的image set和roi set，索引
    Output: None
"""
def _plot_ss_(image_set,roi_set,index):
    image = image_set[index]
    candidates = roi_set[index]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)
    for x, y, w, h in candidates:
        #print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()

"""
    筛选框，目前的筛选策略有个数/占比
    Input: canny edges info, ROI candidate bouding box
    Output: selected ROI
"""
def _selectedROI_(edges, candidates):
    len_edges = len(edges)
    candidates2 = []
    # 将set的内容给list，便于拿取指定位置的信息
    for rect in candidates:
        candidates2.append(rect)
    selected_rec = [] # 存放筛选后的框信息
    num_rec = [] #记录每个框包含的边缘点的个数
    # 判断每个框内含有多少个边缘点
    if len(edges) == 0:
        return candidates2
    else:
        for x, y, w, h in candidates2:  # 边框的遍历
            in_rec_num = 0
            for xp, yp in edges:  # 边缘点的遍历
                if x <= xp <= (x + w) and y <= yp <= (y + h):
                    in_rec_num += 1
            num_rec.append(in_rec_num)
        for i in range(len(num_rec)):
            x, y, w, h = candidates2[i]
            size = w * h
            try:
                if num_rec[i] >= np.floor(len_edges / 4) or num_rec[i] / size > 0.4:
                    selected_rec.append(candidates2[i])
            except ZeroDivisionError:
                print(len(edges))
        return selected_rec
"""
    Resize ROI
"""
def resize_ROI(selected_rec, size):
    fullsize_rec = []
    for x,y,w,h in selected_rec:
        x *= size
        y *= size
        w *= size
        h *= size
        fullsize_rec.append([x,y,w,h])
    return fullsize_rec
"""
    画出候选框结果
"""
def show_selected_rect(selected_rec, image):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image, 'gray')
    for x, y, w, h in selected_rec:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()
"""
    对ROI排序，以便于后期补全和删除来控制ROI个数
"""
def evaluation_rec(selected_rec, dst):
    average = np.mean(dst)  # 图像平均像素
    new_selected_rec = []
    """
    优先级标准：
    1、面积大于400的框
    2、框内像素均值小于整幅图均值时，按照面积和灰度降序排列（若面积相同，则选灰度大的）
    3、框内像素均值大于整幅图均值，
    """
    # 定义三个字典，面积大于400，像素低于average、低于average
    size_400 = {}
    low_average = {}
    high_average = {}
    for i in range(len(selected_rec)):
        y1 = selected_rec[i][1]
        y2 = selected_rec[i][1] + selected_rec[i][3]
        x1 = selected_rec[i][0]
        x2 = selected_rec[i][0] + selected_rec[i][2]
        img = dst[y1:y2, x1:x2]  # 复制出框的图像
        average_img = np.mean(img)  # 框内像素均值
        size = selected_rec[i][2] * selected_rec[i][3]  # 框的面积
        if size > 400:
            size_400.update({i: [size, average_img]})
        else:
            if average_img < average:
                low_average.update({i: [size, average_img]})
            else:
                high_average.update({i: [size, average_img]})
    """注意：sorted后返回的是list，其元素是tuple"""
    size_400 = sorted(size_400.items(), key=lambda x: x[1], reverse=True)  # 按照面积降序排列的list
    low_average = sorted(low_average.items(), key=lambda x: x[1], reverse=True)
    high_average = sorted(high_average.items(), key=lambda x: x[1], reverse=True)

    for key in range(len(size_400)):
        i = size_400[key][0]
        # 字典键值，=存放的是边框selected_rec中的位置信息！
        new_selected_rec.append(selected_rec[i])
        pass
    for key in range(len(low_average)):
        i = low_average[key][0]
        new_selected_rec.append(selected_rec[i])
        pass
    for key in range(len(high_average)):
        i = high_average[key][0]
        new_selected_rec.append(selected_rec[i])
        pass
    return new_selected_rec
"""
    把ROI控制到5个
    Output:position 
"""
def normalize_ROI(selected_rec, dst):
    length = len(selected_rec)
    normalized_rec = []
    if length == 1:  # 增加5次第一个边框
        for i in range(5):
            normalized_rec.append(selected_rec[0])
    if length == 2:  # 扩充2次最优边框，1次次优边框
        normalized_rec = selected_rec
        normalized_rec.append(selected_rec[0])
        normalized_rec.append(selected_rec[0])
        normalized_rec.append(selected_rec[1])
    if length == 3:  # 增加前两个边框
        normalized_rec = selected_rec
        normalized_rec.append(selected_rec[0])
        normalized_rec.append(selected_rec[1])
    if length == 4:  # 增加第一个边框
        normalized_rec = selected_rec
        normalized_rec.append(selected_rec[0])
    if length == 5:  # 不用增减
        normalized_rec = selected_rec
    if length > 5:  # 个数大于5则归一化到5
        for i in range(5):
            normalized_rec.append(selected_rec[i])
    position = []
    one_vector = []
    for i in range(len(normalized_rec)):
        x1 = normalized_rec[i][0]
        y1 = normalized_rec[i][1]
        w = normalized_rec[i][2]
        h = normalized_rec[i][3]
        x2 = x1 + w
        y2 = y1 + h
        pos = [y1, y2, x1, x2]
        position.append(pos)
        one_vector.append(y1)
        one_vector.append(y2)
        one_vector.append(x1)
        one_vector.append(x2)
    return one_vector, position
###################################################################################################
    #图像预处理
###################################################################################################
"""
    图像预处理：原图>中值>小波（返回）>resize>正规化>阈值
    1. normalize_transform: 灰度正规化
    2. preprocessing: 小波，resize, 直方图正规化
        Output: 小波和正规化之后的
"""
def normalize_transform(image):
    Imin, Imax = cv2.minMaxLoc(image)[:2]
    Omin, Omax = 0, 255
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * image + b
    out = out.astype(np.uint8)
    return out
def preprocessing(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转为灰度图
    #dst1 = cv2.medianBlur(image, 3)  # 中值滤波
    cA, (cH, cV, cD) = pywt.dwt2(image, 'haar')  # 小波
    resize = cv2.resize(src=cA, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # 双线性插值resize
    dst = normalize_transform(resize)  # 直方图正规化 --> 增强
    average = np.mean(dst)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] > (average - 20):
                dst[i][j] = 255  # 低于平均像素值的地方标为白色
    return cA, dst
"""
    网络输入预处理，resize
    Output: 处理完的Image
"""
def resizeImage(img):
    im_aug = tfs.Compose([tfs.Resize((256,256))]) 
    img_set = []
    for i in img:
        i = tfs.ToPILImage()(i)
        i = np.array(im_aug(i))
        img_set.append(i)
    return img_set
"""
    网络输入预处理，img to tensor
    Output: 处理完的Image
"""
def img_2_Torch(img):
    im_aug = tfs.Compose([tfs.ToTensor()]) 
    img_set = []
    img_set_3 = []
    # 求mean std
    means = []
    stds = []
    for i in img:
        i = tfs.ToPILImage()(i)
        i = im_aug(i)
        means.append(torch.mean(i))
        stds.append(torch.std(i))
    mean = torch.mean(torch.tensor(means)) # 0.3971
    std = torch.mean(torch.tensor(stds))   # 0.0598
    
    im_aug = tfs.Compose([tfs.ToTensor(),tfs.Normalize([mean,mean,mean],[std,std,std])])
    for i in img:
        # 单通道变三通道
        image = np.expand_dims(i, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = tfs.ToPILImage()(image)
        image = im_aug(image)
        # 三通道变单通道
        img_set.append(image[0]) 
        img_set_3.append(image)
    return np.array(img_set),np.array(img_set_3)
"""
    整个ROI提取和预处理流程
    Output: ROI pointset, normalize_transform image
"""
def cal_ROI(src_image,src_label):
    ROI5 = []
    image_set_dst = []
    for k in range(len(src_image)):
        print(k,len(src_image))
        image = src_image[k]
        label = src_label[k]     
        cA, dst = preprocessing(image)  # 此时dst已经resize过
        #在小波图上做ROI提取
        edges = _get_edgeinfo_(cA)  # 获取边缘信息
        candidates = _get_ssinfo_(cA)  # 获取候选框
        selected_rec = _selectedROI_(edges=edges, candidates=candidates)  # 第一次筛选候选框
        if len(selected_rec) == 0: # 会跳图
            continue
        selected_rec = resize_ROI(selected_rec, 2)  # 扩大候选框面积,小波变换是1/2，恢复到原来的大小
        selected_rec = evaluation_rec(selected_rec, dst)  # 对selected_rec做策略排序
        # 将候选框的数量定格在5或10，返回值是一个list，其中包含了5or10个ROI坐标信息，依次为（y1,y2,x1,x2）
        one_vector,_= normalize_ROI(selected_rec, dst)
        roi_count =  len(one_vector)//4
        for i in range(4):
            one_vector.insert(0, int(label))
        if roi_count==5:
            ROI5.append(np.array(one_vector))
            image_set_dst.append(dst)
    src_image,_ = img_2_Torch(src_image)
    image_set_dst,_ = img_2_Torch(image_set_dst)
    new_ROI5 = torch.tensor(ROI5)
    new_ROI5 = new_ROI5.view(len(new_ROI5),6,1,4)
    return ROI5,torch.tensor(image_set_dst),torch.tensor(src_image)
