#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:04:19 2020

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images_mqae_subBK/20X MQAE 7.5mM well 1_subBK .png",0)
img1 = cv2.imread("images_mqae_subBK/20X MQAE 7.5mM well 1 + IVM t1_subBK.png", 0)
img2 = cv2.imread("images_mqae_subBK/20X MQAE 7.5mM well 1 + IVM t2_subBK.png", 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img)

clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img1 = clahe1.apply(img1)

clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img2 = clahe2.apply(img2)


blur = cv2.GaussianBlur(clahe_img,(5,5),0)
ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


blur1 = cv2.GaussianBlur(clahe_img1,(5,5),0)
ret1,th1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


blur2 = cv2.GaussianBlur(clahe_img2,(5,5),0)
ret2,th2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plt.subplot(1, 3, 1)
# plt.hist(blur.flat, bins =20, range=(0,15))
# plt.xlabel('Intensity')
# plt.ylabel('Pixel Count')
# plt.title('Well 1')


# plt.subplot(1, 3, 2)
# plt.hist(blur1.flat, bins =20, range=(0,15))
# plt.xlabel('Intensity')
# plt.ylabel('Pixel Count')
# plt.title('Well 1+IVM t1')

# plt.subplot(1, 3, 3)
# plt.hist(blur2.flat, bins =20, range=(0,15))
# plt.xlabel('Intensity')
# plt.ylabel('Pixel Count')
# plt.title('Well 1+IVM t2')

# plt.tight_layout()
# plt.show()

plt.subplot(1, 3, 1)
plt.hist(img.flat, bins =5, range=(0,5))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 1')


plt.subplot(1, 3, 2)
plt.hist(img1, bins =5, range=(0,5))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 1+IVM t1')

plt.subplot(1, 3, 3)
plt.hist(img2, bins =5, range=(0,5))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 1+IVM t2')

plt.tight_layout()
plt.show()

######################################

np.random.seed(6789)
x = np.random.gamma(4, 0.5, 1000)


plt.subplot(1, 3, 1)
result =plt.hist(img.flat, bins =5, range=(0,5), color='c', edgecolor='k', alpha=0.65)
plt.axvline(img.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(img.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(img.mean()))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 1')


plt.subplot(1, 3, 2)
result1 =plt.hist(img1.flat, bins =5, range=(0,5), color='c', edgecolor='k', alpha=0.65)
plt.axvline(img1.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim1, max_ylim1 = plt.ylim()
plt.text(img1.mean()*1.1, max_ylim1*0.9, 'Mean: {:.2f}'.format(img1.mean()))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 1+IVM t1')

plt.subplot(1, 3, 3)
result2 =plt.hist(img2.flat, bins =5, range=(0,5), color='c', edgecolor='k', alpha=0.65)
plt.axvline(img2.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim2, max_ylim2 = plt.ylim()
plt.text(img2.mean()*1.1, max_ylim2*0.9, 'Mean: {:.2f}'.format(img2.mean()))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 1+IVM t2')


import seaborn as sns

sns.set_style('darkgrid')
sns.distplot(img.flat,  kde=False)