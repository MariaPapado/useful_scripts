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

#disappearing buidlings
old_cnts = []
for cnt1 in pred1_contours:
    if len(cnt1)>2:
        pol_cnt1 = Polygon(np.squeeze(cnt1)).buffer(0)
        flag = False
        for cnt2 in pred2_contours:
            if len(cnt2)>2:
                pol_cnt2 = Polygon(np.squeeze(cnt2)).buffer(0)
                if pol_cnt1.intersects(pol_cnt2):
                    flag = True            
                    #print('ok')
                    #break
        if flag==False:
            old_cnts.append(cnt1)

#appearing buildings
new_cnts = []
for cnt2 in pred2_contours:
    if len(cnt2)>2:
        pol_cnt2 = Polygon(np.squeeze(cnt2)).buffer(0)
        flag = False
        for cnt1 in pred1_contours:
            if len(cnt1)>2:
                pol_cnt1 = Polygon(np.squeeze(cnt1)).buffer(0)
                if pol_cnt2.intersects(pol_cnt1):
                    flag = True            
                    #print('ok')
                    #break
        if flag==False:
            new_cnts.append(cnt2)

print('olds', len(old_cnts)) 
print('news', len(new_cnts)) 
cv2.fillPoly(img_contour, old_cnts, (255, 255, 255))
cv2.fillPoly(img_contour, new_cnts, (255, 0, 255))

img_contour = np.array(img_contour, dtype=np.uint8)
out = Image.fromarray(img_contour)
out.save('check.tif')

#cv2.imwrite('check.png', img_contour)

