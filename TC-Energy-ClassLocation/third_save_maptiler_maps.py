import rasterio
import rasterio.features
from PIL import Image
import numpy as np
import os
from shapely import geometry
import cv2



def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    # Open the input TIF file
#    image = np.expand_dims(image, 2)

    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
#    print('type', type(image))
    print('image', image.shape)
    for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)

    new_dataset.close()

    return True



b_ids = os.listdir('./BASEMAP_BUILDINGS/')
print(len(b_ids))
m_ids = os.listdir('/cephfs/work/coregistration_test/basemaps/TC-Energy-Pilot_basemap')
m_ids = [mid[7:] for mid in m_ids]
print(len(m_ids))

#####remove_ids = ['9300682.tif', '9302189.tif', '9358281.tif', '9300180.tif', '9300431.tif', '9301939.tif', '9358030.tif', '9304201.tif', '9302441.tif', '9302190.tif']

set1 = set(b_ids)
set2 = set(m_ids)
ids = list(set2.intersection(set1))
print(len(ids))

cnt=0
for id in ids:
    print('{}/{}'.format(cnt, len(ids)))
    pred_img = rasterio.open('./BASEMAP_BUILDINGS/{}'.format(id))
    maptiler_map = rasterio.open('/cephfs/work/coregistration_test/basemaps/TC-Energy-Pilot_basemap/region_{}'.format(id))
#    print(id)

    bounds = pred_img.bounds
#    print(bounds)
    bounds_window = rasterio.features.bounds(geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top))
        # Define window
    window = rasterio.windows.from_bounds(*bounds_window, maptiler_map.transform)
        # Read image array
    img = maptiler_map.read(window=window)    
    img = np.transpose(img, (1,2,0))


    save_tif_coregistered('./MAPTILER_BASEMAPS_BOUNDS/{}'.format(id), img, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)
    cnt = cnt+1
