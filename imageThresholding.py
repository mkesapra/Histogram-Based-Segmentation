#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:57:15 2020

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread("images/20X MQAE 7.5mM well 1 .PNG", 0)
img = cv2.imread("macrophagepolarizedto_anode.tif",0)
equ = cv2.equalizeHist(img)

plt.hist(equ.flat, bins=8, range=(0,260))

cv2.imshow("Original Image", img)
cv2.imshow("Equalized", equ)


#Histogram Equalization considers the global contrast of the image, may not give good results.
#Adaptive histogram equalization divides images into small tiles and performs hist. eq.
#Contrast limiting is also applied to minimize aplification of noise.
#Together the algorithm is called: Contrast Limited Adaptive Histogram Equalization (CLAHE)

# Start by creating a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
cl1 = clahe.apply(img)

cv2.imshow("CLAHE", cl1)

cv2.waitKey(0)          
cv2.destroyAllWindows() 


######################################################
###################################################
#Image thresholding

import cv2
import matplotlib.pyplot as plt

#img = cv2.imread("images/20X MQAE 7.5mM well 1 .PNG", 0)
img = cv2.imread("macrophagepolarizedto_anode.tif",0)

#Adaptive histogram equalization using CLAHE to stretch the histogram. 
#Contrast Limited Adaptive Histogram Equalization covered in the previous tutorial. 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img)
plt.hist(clahe_img.flat, bins =10, range=(0,255))
#plt.hist(clahe_img.flat, bins =80, range=(40,90))


#Thresholding. Creates a uint8 image but with binary values.
#Can use this image to further segment.
#First argument is the source image, which should be a grayscale image.
#Second argument is the threshold value which is used to classify the pixel values. 
#Third argument is the maxVal which represents the value to be given to the thresholded pixel.

ret,thresh1 = cv2.threshold(clahe_img,185,150,cv2.THRESH_BINARY)  #All thresholded pixels in grey = 150
ret,thresh2 = cv2.threshold(clahe_img,185,255,cv2.THRESH_BINARY_INV) # All thresholded pixels in white

cv2.imshow("Original", img)
cv2.imshow("Binary thresholded", thresh1)
cv2.imshow("Inverted Binary thresholded", thresh2)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

############################################
#OTSU Thresholding, binarization
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/20X MQAE 7.5mM well 1 .PNG", 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img)

plt.hist(clahe_img.flat, bins =100, range=(30,80))

# binary thresholding
ret1,th1 = cv2.threshold(clahe_img,185,200,cv2.THRESH_BINARY)

# Otsu's thresholding, automatically finds the threshold point. 
#Compare wth above value provided by us (185)
ret2,th2 = cv2.threshold(clahe_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow("Otsu", th2)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

# If working with noisy images
# Clean up noise for better thresholding
# Otsu's thresholding after Gaussian filtering. Canuse median or NLM for beteer edge preserving

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("macrophagepolarizedto_anode.tif",0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img)
blur = cv2.GaussianBlur(clahe_img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#from skimage import filters

#filters.frangi(blur, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True)

plt.hist(blur.flat, bins =100, range=(0,255))
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.imshow(th3)

##regionprops
from skimage import measure, io, img_as_ubyte
import numpy as np
from skimage.color import label2rgb

img_ubyte = img_as_ubyte(io.imread("macrophagepolarizedto_anode.tif"))

from skimage.filters import threshold_otsu
threshold = threshold_otsu(img_ubyte)

label_image = measure.label(img_ubyte < threshold, connectivity=img_ubyte.ndim)

image_label_overlay = label2rgb(label_image, image = img_ubyte)

plt.imshow(image_label_overlay)

props = measure.regionprops_table(label_image,img_ubyte, properties = ['label','area','equivalent_diameter','mean_intensity','solidity'])

import pandas as pd
df = pd.DataFrame(props)
print(df.head())


fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img_ubyte, cmap=plt.cm.gray)
regions = measure.regionprops(label_image)
# Add labels to the plot
style = dict(size=10, color='red') #Style for the text to be added to the image

for props in regions:
    y, x = props.centroid   #Gives coordinates for the object centroid
    label = props.label  #Gives the label number for each object/region
    ax.text(x, y, label, **style)   #Add text to the plot at given coordinates. In this case the text is label number. 
plt.show()


cv2.imshow("Original", img)
cv2.imshow("OTSU Gaussian cleaned", th3)



cv2.waitKey(0)          
cv2.destroyAllWindows() 