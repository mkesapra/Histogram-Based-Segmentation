#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:07:05 2020

@author: kesaprm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:04:19 2020

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("20X well 1 IVM t1_Plate_R_p00_0_B02f01d0.PNG",0)
img2 = cv2.imread("20X well 1 IVM t2_Plate_R_p00_0_B02f01d0.PNG", 0)
img3 = cv2.imread("20X well 1_Plate_R_p00_0_B02f00d0.PNG", 0)
img4 = cv2.imread("20X well 2 IVM t1_Plate_R_p00_0_B03f01d0.PNG", 0)
img5 = cv2.imread("20X well 2 IVM t2_Plate_R_p00_0_B03f01d0.PNG", 0)

img6 = cv2.imread("20X well 1 IVM 20X well 2_Plate_R_p00_0_B03f00d0.PNG", 0)
img7 = cv2.imread("20X well 20X well 3 IVM t1_Plate_R_p00_0_B04f01d0.PNG", 0)
img8 = cv2.imread("20X well 2 IVM 20X well 3 IVM t2_Plate_R_p00_0_B04f01d0.PNG", 0)
img9 = cv2.imread("20X well 2 IVM 20X well 3_Plate_R_p00_0_B04f00d0.PNG", 0)

blur = cv2.GaussianBlur(img,(5,5),0)
ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img2,(5,5),0)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img3,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img4,(5,5),0)
ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img5,(5,5),0)
ret5,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


blur = cv2.GaussianBlur(img2,(5,5),0)
ret6,th6 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img3,(5,5),0)
ret7,th7 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img4,(5,5),0)
ret8,th8 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img5,(5,5),0)
ret9,th9 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)




plt.hist(img.flat, bins =10, range=(0,220))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Day23_1_Plate_D_p00_0_A01f00d0.PNG')


np.random.seed(6789)
x = np.random.gamma(4, 0.5, 1000)


#plt.subplot(1, 3, 1)
result =plt.hist(img.flat, bins =5, range=(0,5), color='c', edgecolor='k', alpha=0.65)
plt.axvline(img.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(img.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(img.mean()))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 2')


plt.subplot(1, 3, 2)
result =plt.hist(img1.flat, bins =5, range=(0,5), color='c', edgecolor='k', alpha=0.65)
plt.axvline(img1.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(img1.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(img1.mean()))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 2+IVM t1')

plt.subplot(1, 3, 3)
result =plt.hist(img2.flat, bins =5, range=(0,5), color='c', edgecolor='k', alpha=0.65)
plt.axvline(img2.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(img2.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(img2.mean()))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Well 2+IVM t2')

#plt.tight_layout()
#plt.show()


import seaborn as sns

sns.set_style('darkgrid')
sns.distplot(img2.flat,  kde=False)

