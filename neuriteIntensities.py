#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:57:39 2020

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("NeuO_from_imageJ/Day23_1_Plate_D_p00_0_A01f00d0.tif",0)
img2 = cv2.imread("NeuO_from_imageJ/Day23_2_Plate_D_p00_0_A01f01d0.tif", 0)
img3 = cv2.imread("NeuO_from_imageJ/Day23_3_Plate_D_p00_0_A01f02d0.tif", 0)
img4 = cv2.imread("NeuO_from_imageJ/Day23_4_Plate_D_p00_0_A01f03d0.tif", 0)
img5 = cv2.imread("NeuO_from_imageJ/Day23_5_Plate_D_p00_0_A01f04d0.tif", 0)
imgN = cv2.imread("NeuO_output_sub_BG_imageJ/Day23_1_Plate_D_p00_0_A01f00d0.png",0)
img2N = cv2.imread("NeuO_output_sub_BG_imageJ/Day23_2_Plate_D_p00_0_A01f01d0.png",0)
img3N = cv2.imread("NeuO_output_sub_BG_imageJ/Day23_3_Plate_D_p00_0_A01f02d0.png",0)
img4N = cv2.imread("NeuO_output_sub_BG_imageJ/Day23_4_Plate_D_p00_0_A01f03d0.png",0)
img5N = cv2.imread("NeuO_output_sub_BG_imageJ/Day23_5_Plate_D_p00_0_A01f04d0.png",0)



blur = cv2.GaussianBlur(img,(5,5),0)
ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(imgN,(5,5),0)
retN,thN = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


blur = cv2.GaussianBlur(img2,(5,5),0)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img2N,(5,5),0)
ret2N,th2N = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


blur = cv2.GaussianBlur(img3,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img4,(5,5),0)
ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img5,(5,5),0)
ret5,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



blur = cv2.GaussianBlur(img3N,(5,5),0)
ret3N,th3N = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img4N,(5,5),0)
ret4N,th4N = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img5N,(5,5),0)
ret5N,th5N = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


hist = cv2.calcHist([img5],[0],None,[256],[18,256])  
histN = cv2.calcHist([img5N],[0],None,[256],[5,256])  

plt.plot(hist) 
plt.show() 

plt.plot(histN) 
plt.show() 
I_Pixels = np.sum(hist)
N_Pixels= np.sum(histN)

I_Pixels

N_Pixels

hist = plt.hist(img.flat, bins =20, range=(18,100))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Day23_1_Plate_D_p00_0_A01f00d0 - Neurites only')

#Plates = ['D4_2','D4_3','D4','D9_2','D9_3','D9_4','D9','D16_2','D16_3','D16','D23_1','D23_2','D23_3','D23_4','D23_5']

Plates = ['4_2','4_3','4','9_2','9_3','9_4','9','16_2','16_3','16','23_1','23_2','D23_3','23_4','23_5']

pixels_cells = [1187098,1176373,1274579,688696,621137,642555,692466,328617,329110,375071,402006,459814,303072,344671,619427]

pixels_neurites = [386138,318936,267297,399738,356725,390577,391941,374465,301698,437035,391310,434421,372320,434623,410874]

plt.plot(Plates,pixels_cells)
plt.xlabel('Plates')
plt.ylabel('Total no. of pixels')
plt.title('NeuO: Cells - Total no. of pixels for each plate')


plt.plot(Plates,pixels_neurites)
plt.xlabel('Plates')
plt.ylabel('Total no. of pixels')
plt.title('NeuO: Neurites - Total no. of pixels for each plate')

ratio=[]

for i in (pixels_neurites,pixels_cells):
    print(i)
    print(pixels_neurites[i]/pixels_cells[i])


