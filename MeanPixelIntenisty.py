#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:29:31 2021

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/kesaprm/Downloads/M0_NO_Plate_D_p00_0_C03f00d3.tif",0)


blur = cv2.GaussianBlur(img,(5,5),0)
ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


hist = cv2.calcHist([img],[0],None,[256],[ret,256])  
tot_px = cv2.calcHist([img],[0],None,[256],[0,256])  

# plt.plot(hist) 
# plt.show() 
I_Pixels = np.sum(hist)

mean_px = I_Pixels/np.sum(tot_px)

print(mean_px)




