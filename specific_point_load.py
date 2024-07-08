import shapely
from shapely import wkt
import pandas as pd
import rasterio
import rasterio.features
from PIL import Image
import numpy as np
import cv2

valid_area_region = wkt.loads('POLYGON ((-106.55 23.586000000000002, -106.54 23.586000000000002, -106.54 23.592, -106.55 23.592, -106.55 23.586000000000002))')

print(type(valid_area_region))



included_points = [shapely.geometry.Point(-106.542932, 23.587319).buffer(0.00001)]

img = Image.open('new_img.png')
img = np.array(img)
width, height = img.shape[1], img.shape[0]

geotiff_transform = rasterio.transform.from_bounds(valid_area_region.bounds[0], valid_area_region.bounds[1],
                                                       valid_area_region.bounds[2], valid_area_region.bounds[3],
                                                       width, height)
mask = rasterio.features.rasterize(included_points, out_shape=img.shape[:2],transform=geotiff_transform)

print(mask.shape)
print(np.unique(mask))
cv2.imwrite('mask.png', mask*255)
