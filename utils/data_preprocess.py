# -*- coding: utf-8 -*-
"""
@Author: Zhoufu Ren <ren.scsh@outlook.com>
@license: MIT-license

This is a demo python script

Overview:

- fun print hello world
"""
import sys
import os
import cv2
import time
import torch


def label_clean():
    print('Hello World!')
    data_dir = r'D:\code\projects\data\dataset_1000'
    data_new_dir = r'D:\code\projects\data\dataset_1000_new'
    data_num = 1000
    img_list = []
    for i in range(data_num):
        print(i,'/',data_num)
        img = cv2.imread(os.path.join(data_dir, '{:06d}.png'.format(i)))
        img_list.append(img)
        seg = cv2.imread(os.path.join(data_dir, '{:06d}_seg.png'.format(i)))

        '''clean the faceware label'''
        seg[seg == 18] = 0
        seg[seg == 255] = 0
        cv2.imwrite(os.path.join(data_new_dir,'{:06d}.png'.format(i)),img)
        cv2.imwrite(os.path.join(data_new_dir,'{:06d}_seg.png'.format(i)),seg)



def gen_descriptions():
    print('Hello World!')
    data_dir = r'D:\\code\\projects\\data\\dataset_1000_new'
    train_data_num = 900
    val_data_num = 100
    with open('../dataset/train_mine.odgt', 'w') as f:
        for i in range(train_data_num):
            flag = '{{"fpath_img": "{0}", "fpath_segm": "{1}", "width": "{2}", "height": "{3}"}}'.\
                format(data_dir + r'\\{:06d}.png'.format(i), data_dir + r'\\{:06d}_seg.png'.format(i), 512, 512)
            f.write(flag+'\n')

    with open('../dataset/val_mine.odgt', 'w') as f:
        for i in range(val_data_num):
            i += train_data_num
            flag = '{{"fpath_img": "{0}", "fpath_segm": "{1}", "width": "{2}", "height": "{3}"}}'. \
                format(data_dir + r'\\{:06d}.png'.format(i), data_dir + r'\\{:06d}_seg.png'.format(i), 512, 512)
            f.write(flag+'\n')

def get_info():
    pix = 1.0
    sum_tot = torch.tensor([0, 0, 0])
    sum_sq_tot = torch.tensor([0, 0, 0])
    sum_pix = 0.0

    data_dir = r'D:\code\projects\data\dataset_1000'
    data_num = 1000
    for i in range(data_num):
        # NOTE: must convert to "int", otherwise image**2 will overflow
        #       since results are limited to uint8 [0~255]
        #
        image_cv = cv2.imread(os.path.join(data_dir, '{:06d}.png'.format(i)))
        image_cv = image_cv.astype(int)
        # np.sum(np.sum(image_cv<0, axis=0), axis=0)  --> count number of negative values
        # image = np.asarray(image_cv)                --> convert to numpy array if needed
        # image_cv = np.transpose(image_cv, 0, -1)    --> transpose the axis if needed

        image = torch.from_numpy(image_cv)
        if (3 != image.shape[2]):
            continue
        sum = torch.sum(image, dim=[0, 1])
        sumsq = torch.sum(image ** 2, dim=[0, 1])
        pix = image.shape[0] * image.shape[1]

        # NOTE: you can test each image with the following codes (it should be: std1 == std2)
        # avg = sum/pix
        # image_avg = torch.ones(image.shape) * avg
        # image_dif = image - image_avg
        # std1 = torch.sum(image_dif**2, [0, 1])/(pix - 1)
        # std2 = (sumsq - pix* avg**2)/(pix -1)

        sum_tot = sum_tot + sum
        sum_sq_tot = sum_sq_tot + sumsq
        sum_pix = sum_pix + pix
        print(i, '/', data_num)

    mean = sum_tot / sum_pix
    std = (sum_sq_tot - sum_pix * mean ** 2) / (sum_pix - 1.0)
    std = std ** 0.5

    print(mean)
    print(std)

if __name__ == '__main__':
    label_clean()
    # gen_descriptions()
    # get_info()



