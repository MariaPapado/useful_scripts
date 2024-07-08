from PIL import Image
import numpy as np
import cv2
from skimage import io
from shapely.geometry import Polygon
import os
import cv2

pred1 = Image.open('./MY_RESULTS/0/9360795.tif')
pred2 = Image.open('./MY_RESULTS/1/9360795.tif')

pred1, pred2 = np.array(pred1), np.array(pred2)

pred1_contours, hierarchy = cv2.findContours(pred1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
pred2_contours, hierarchy = cv2.findContours(pred2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_contour = np.zeros((pred1.shape[0], pred1.shape[1], 3))

####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
intersection = np.logical_and(pred1, pred2)  
idx = np.where(intersection==True)
pred1[idx] = 0
pred2[idx] = 0

old_c, hierarchy = cv2.findContours(pred1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
new_c, hierarchy = cv2.findContours(pred2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.fillPoly(img_contour, old_c, (255, 255, 255))
cv2.fillPoly(img_contour, new_c, (255, 0, 255))
#cv2.imwrite('check12.png', img_contour)

img_contour = np.array(img_contour, dtype=np.uint8)
out = Image.fromarray(img_contour)
out.save('check_intersect.tif')

