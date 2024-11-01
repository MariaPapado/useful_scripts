from PIL import Image
import numpy as np
import cv2

im = Image.open('5996663.tif')
im = np.array(im)
print(im.shape)

height, width = im.shape[0], im.shape[1]
print(height, width)
psize=512
step=512


img_x, img_y = [], []

for x in range(0, height, step):
    print('x', x)
    if x+psize<=height:
        img_x.append(x)
    else:
        img_x.append(height-psize)
        break
    for y in range(0, width, step):
        if y+psize<=width:
            img_y.append(y)
        else:
            img_y.append(width-psize)
            break
        

print(img_x)
print(img_y)




for x in img_x:
    for y in img_y:
        img_patch = im[x:x+psize,y:y+psize,:]
        cv2.imwrite('./check_patches/patch_{}_{}.png'.format(x,y), img_patch)
