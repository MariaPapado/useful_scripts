import shapely
from shapely import wkt
import pandas as pd
import rasterio
import rasterio.features
from PIL import Image
import numpy as np
import cv2

valid_area_region = wkt.loads('POLYGON ((-106.63 23.802, -106.62 23.802, -106.62 23.808, -106.63 23.808, -106.63 23.802))')

print(type(valid_area_region))

csv_points = pd.read_csv('tcenergy_building_points.csv')
print(len(csv_points['Lat']))

Lats = csv_points['Lat']
Longs = csv_points['Long']
count = len(csv_points['Lat'])


included_points = []
for p in range(0, count):
    bp = shapely.geometry.Point([csv_points['Long'][p], csv_points['Lat'][p]]).buffer(0.00001)
    
    if valid_area_region.contains(bp):
      print(bp)
      included_points.append(bp)


included_points = [included_points[7]]
img = Image.open('tc_energy.png')
img = np.array(img)
width, height = img.shape[1], img.shape[0]

geotiff_transform = rasterio.transform.from_bounds(valid_area_region.bounds[0], valid_area_region.bounds[1],
                                                       valid_area_region.bounds[2], valid_area_region.bounds[3],
                                                       width, height)
mask = rasterio.features.rasterize(included_points, out_shape=img.shape[:2],transform=geotiff_transform)

print(mask.shape)
print(np.unique(mask))
cv2.imwrite('mask.png', mask*255)
