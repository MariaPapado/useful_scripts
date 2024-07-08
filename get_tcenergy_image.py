print('o')
import torch
from pimsys.regions.RegionsDb import RegionsDb
from datetime import datetime, timedelta
from shapely import geometry
from src.high_res_image import Image
print('01')
import torch
print('02')
import cv2
import numpy as np

def get_utc_timestamp(x: datetime):
    return int((x - datetime(1970, 1, 1)).total_seconds())

settings = {
  "settings_client": {
    "host": "tcenergy.orbitaleye.nl",
    "port": 9949,
    "user": "root",
    "password": "9YxeCg4R2Un%"
  },
  "settings_db": {
    "host": "sar-ccd-db.orbitaleye.nl",
    "port": 5433,
    "user": "postgres",
    "password": "sarccd-db",
    "database": "sarccd2"
  },
  "customer": "TC-Energy-Pilot",
  "slack_channel": "#project-tcenergy-production",
  "output_folder": "/cephfs/pimsys/highres-out/tcenergy",
  "test_url_cloud1": "https://highresclouds_ml.orbitaleye.nl/api/process",
  "test_url_cloud2": "https://highrescloudrgbn_ml.orbitaleye.nl/api/process",
  "test_url_cd": "https://highrescd_ml.orbitaleye.nl/api/process",
  "n_days_window_reference": 100,
  "image_database": "/home/sarccd/highres-cd-processing/images_database_tcenergy.p",
  "settings_geoserve": {
    "user": "",
    "password": ""
  }
}

interval_days = 2
# interval_start = datetime(2023, 9, 4)
interval_end = datetime(2023, 7, 10)
# interval_end = datetime.combine(date.today(), datetime.min.time())- timedelta(days=1)
interval_start = interval_end - timedelta(days=interval_days)
interval_utc = [get_utc_timestamp(interval_start), get_utc_timestamp(interval_end)]

database = RegionsDb(settings['settings_db'])
regions = database.get_regions_by_customer(settings['customer'])
database.close()
#-106.627021   23.807504
coords = [23.587319, -106.542932][::-1]#[region['bounds'].centroid.x, region['bounds'].centroid.y]

with RegionsDb(settings['settings_db']) as database:
    images = database.get_optical_images_containing_point_in_period(coords, interval_utc)

    wms_images = sorted(images, key=lambda x: x['capture_timestamp'])
    wms_images = [x for x in wms_images if x['source'] != 'Sentinel-2']

region_intersecting = [region for region in regions if region['bounds'].contains(geometry.Point(coords))][0]
layer = wms_images[0]
image = Image(layer, settings, window=region_intersecting['bounds'])
print(region_intersecting['bounds'])

out = image.image
#print(type(image))
out = out[0:3,:,:]
out = np.transpose(out, (1,2,0))

cv2.imwrite('img.png', out*255)


