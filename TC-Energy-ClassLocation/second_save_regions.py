import rasterio
import rasterio.features
from pimsys.regions.RegionsDb import RegionsDb
import pyproj
from shapely import geometry, ops
import numpy as np
import cv2
import pandas as pd
import pickle
from predict_function import *
from PIL import Image

settings = {"settings_client": {
     "user": "root",
     "password": "9YxeCg4R2Un%",
     "host": "tcenergy.orbitaleye.nl",
     "port": 9949
     },
    "settings_db" : {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
        }
    }

def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    # Open the input TIF file
#    image = np.expand_dims(image, 2)

    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename + '.tif', 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
#    print('shape', image.shape)
    for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
#    new_dataset.write(image, indexes=1)
    new_dataset.close()

    return True


def normalise_bands(image, percentile_min=2, percentile_max=98):
    # Get and normalize tif tiles
    tmp = []
    for i in range(image.shape[0]):
        image_band = image[i, :, :].astype(np.float32)
        min_val = np.nanmin(image_band[np.nonzero(image_band)])
        max_val = np.nanmax(image_band[np.nonzero(image_band)])
        image_band = (image_band - min_val) / (max_val - min_val)
        image_band_valid = image_band[np.logical_and(image_band > 0, image_band < 1.0)]
        perc_2 = np.nanpercentile(image_band_valid, percentile_min)
        perc_98 = np.nanpercentile(image_band_valid, percentile_max)
        band = (image_band - perc_2) / (perc_98 - perc_2)
        band[band < 0] = 0.0
        band[band > 1] = 1.0
        tmp.append(band)
    return np.array(tmp).astype(np.float32)

def load_tif_image_window(layer, poly_bounds, norm=True):
    # Define mapserver path
    mapserver_path = "/cephfs/mapserver/data"
    # Get path to after image
    layer_datetime = layer["capture_timestamp"]
    path_tif = (
        "/".join(
            [
                mapserver_path,
                layer_datetime.strftime("%Y%m%d"),
                layer["wms_layer_name"],
            ]
        )
        + ".tif"
    )
    # Load image
    tif = rasterio.open(path_tif)
    # Define crs transform
    proj = pyproj.Transformer.from_crs(4326, str(tif.crs), always_xy=True)
    bounds = ops.transform(proj.transform, poly_bounds)
    bounds_window = rasterio.features.bounds(bounds)
    # Define window
    window = rasterio.windows.from_bounds(*bounds_window, tif.transform)
    # Read image array
    img = tif.read(window=window)
    transform_image = tif.transform
    # Normalize bands on min and max
    if norm:
        img_norm = normalise_bands(img)
    else:
        if img.dtype == "uint8":
            img_norm = img / 255.0
        else:
            img_norm = img

    tif.close()
    return img, img_norm, tif.transform


customer_name = 'TC-Energy-Pilot'
database = RegionsDb(settings['settings_db'])
regions = database.get_regions_by_customer(customer_name)
database.close()

layers_all = pd.read_pickle('basemap_regions.p')

#layers_all = layers_all[0:1]
#print(layers_all)

model = torch.jit.load('./ChangeOS/models/changeos_r101.pt')
#print(model.state_dict().keys())
model.eval()
model = ChangeOS(model)

p=512
s=256

cnt = 0
for layer_image in layers_all:
    print('{}/{}'.format(cnt,len(layers_all)))
    regions_image = [region for region in regions if region['bounds'].intersects(layer_image['valid_area'])]
    cnt = cnt + 1
    for region_image in regions_image:
      img, img_norm, _ = load_tif_image_window(layer_image, region_image['bounds'])

      img, img_norm = np.transpose(img, (1,2,0)), np.transpose(img_norm, (1,2,0))
#      if img_norm.shape[0]>=p and img_norm.shape[1]>=p:
      whole_image = img_norm*255
#      print(whole_image.shape)

      whole_image = whole_image[:,:,0:3]

      save_tif_coregistered('./REGIONS/{}'.format(str(region_image['id'])), whole_image, region_image['bounds'], channels = 3)

