#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:19:36 2020

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images_Sanh/01targethmagcomp_Bottom Slide_D_p01_0_A01f10d2.JPG')

hist = cv2.calcHist([img], [0], None, [256],  [0,256])



plt.hist(img.ravel(),256,[0,100]); plt.show()




cv2.waitKey(0)
cv2.destroyAllWindows()
