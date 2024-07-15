import torch
from torchvision.datasets import ImageFolder


#####################################################################
# the PyTorch way of calculating the mean and std (recommend)
#####################################################################
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


#####################################################################
# the OpenCV and Numpy way of calculating the mean and std
#####################################################################
# import os
# import random
#
# import cv2
# import numpy as np
#
# # calculate means and std
# train_txt_path = './data/Label/TR.txt'
# base_path = './data/food'
#
# CNum = 66071  # 挑选多少图片进行计算
#
# img_h, img_w = 256, 256
# imgs = np.zeros([img_w, img_h, 3, 1])
# means, stdevs = [], []
#
# with open(train_txt_path, 'r') as f:
#     lines = f.readlines()
#     random.shuffle(lines)  # shuffle , 随机挑选图片
#
#     for i in range(CNum):
#         # img_path = os.path.join(base_path, lines[i].rstrip().split()[0])
#         img_path = base_path + lines[i].rstrip().split()[0]
#
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (img_h, img_w))
#         img = img[:, :, :, np.newaxis]
#
#         imgs = np.concatenate((imgs, img), axis=3)
#
# imgs = imgs.astype(np.float32) / 255.
#
# for i in range(3):
#     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
#
# # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
# means.reverse()  # BGR --> RGB
# stdevs.reverse()
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
# print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


#####################################################################
# calculating the mean and std of the images in the file
#####################################################################
# import numpy as np
# import cv2
# import os
#
# # img_h, img_w = 32, 32
# img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
# means, stdevs = [], []
# img_list = []
#
# imgs_path = 'D:/database/VOCdevkit/VOC2012/JPEGImages/'
# imgs_path_list = os.listdir(imgs_path)
#
# len_ = len(imgs_path_list)
# i = 0
# for item in imgs_path_list:
#     img = cv2.imread(os.path.join(imgs_path, item))
#     img = cv2.resize(img, (img_w, img_h))
#     img = img[:, :, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     print(i, '/', len_)
#
# imgs = np.concatenate(img_list, axis=3)
# imgs = imgs.astype(np.float32) / 255.
#
# for i in range(3):
#     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
#
# # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
# means.reverse()
# stdevs.reverse()
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))


if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'./data/food/', transform=None)
    print(getStat(train_dataset))

