import sys
sys.path.append('/home/jovyan/highreschange/models')
import models
from models import model_zoo
from customer import Customer, convert_wgs_to_utm
from high_res_image import Image
from high_res_change_detection_al import HighResCD

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely import geometry
from datetime import datetime, timedelta
from pimsys.regions.RegionsDb import RegionsDb
import torch
from skimage import io

import subprocess
import os
subprocess.run(("ssh", "sarccd@sar-ccd-14.orbitaleye.nl", "free", "-m"))


#import os
#import sys
#directory_path = os.path.abspath('/home/jovyan/highreschange/models')
#if directory_path not in sys.path:
#    sys.path.append(directory_path)

################################################

################################################

class_labels = {0: 'unchanged area',
                1: 'water',
                2: 'bare ground',
                3: 'low vegetation',
                4: 'trees',
                5: 'buildings',
                6: 'sports ground',
                7: 'vehicles',
                8: 'clouds'}


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely import geometry
from datetime import datetime, timedelta
from pimsys.regions.RegionsDb import RegionsDb

class_labels = {0: 'unchanged area',
                1: 'water',
                2: 'bare ground',
                3: 'low vegetation',
                4: 'trees',
                5: 'buildings',
                6: 'sports ground',
                7: 'vehicles',
                8: 'clouds'}

def get_utc_timestamp(x: datetime):
    return int((x - datetime(1970, 1, 1)).total_seconds())


settings = {'settings_client':
                {'user': 'root',
                 'password': '9YxeCg4R2Un%',
                 'host': 'ngc.orbitaleye.nl',
                 'port': 9991,
                 }}
config_db = {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
    }
settings['settings_db'] = config_db
settings['test_url_cloud1'] = 'https://highresclouds_ml.orbitaleye.nl' + '/api/process'
settings['test_url_cloud2'] = 'https://highrescloudrgbn_ml.orbitaleye.nl' + '/api/process'
settings['test_url_cd'] = 'https://highrescd_ml.orbitaleye.nl' + '/api/process'
settings['n_days_window_reference'] = 120

# Define customer
customer = Customer('NGC-Pilot', settings=settings)

interval_end = datetime(2023, 4, 30)  # datetime.now() - timedelta(days=9*interval_days)
interval_start = datetime(2022, 1, 1) #interval_end - timedelta(days=interval_days)
interval_utc = [get_utc_timestamp(interval_start), get_utc_timestamp(interval_end)]

# Get optical User Made TPIs
tpis_optical = customer.get_usermade_tpis(interval_utc)
print(str(interval_start.date()) + ' - ' + str(interval_end.date()))
print(len(tpis_optical))

import model_zoo
model = model_zoo.get_model('pspnet', 'hrnet_w18', False, 7 - 1, True)
model.load_state_dict(torch.load('pspnet_hrnet_w18_36.20.pth', map_location=torch.device('cpu')))
model.eval() 

def get_changes_for_tpi(tpi, settings, customer):
    tpis_skipped = []
    outputs = []

    coords = tpi['point']
    tpi_poly = geometry.Polygon(tpi['coordinates']).buffer(0.)
    tpi_timestamp = tpi['referencedate2']
    tpi_date = datetime.fromtimestamp(tpi_timestamp)

    database = RegionsDb(config_db)
    images = database.get_optical_images_containing_point_in_period(coords, [tpi_timestamp - 1000, tpi_timestamp + 1000])
    images = [x for x in images if x['source'] != 'Sentinel-2']
    images = [x for x in images if x['user_classification'] == 'GOOD']
    images = sorted(images, key=lambda x: x['capture_timestamp'])
    print(images)
    if not images:
        tpis_skipped.append(tpi['id'])
        return [[],tpis_skipped, []]

    # Select after image
    try:
        #layer_after = [x for x in images if x['capture_timestamp'] == tpi_date.date()][0]
        layer_after = images[-1]
    except:
        #print('NO AFTER IMAGE: ' + str(region['id']))
        #print(images)
        tpis_skipped.append(tpi['id'])
        return [[],tpis_skipped, []]
        
    image = Image(layer_after, settings=settings)
    image.tile_image()
    image.predict_clouds()
    image_before, uiq2_image = image.get_reference_images([0,tpi_timestamp], corridor=customer.corridor)
    print('imgbefore', image_before)
    if image_before is None:
        return [[],tpis_skipped, []]
    
    image_before.tile_image()
    image_before.predict_clouds()

    highrescd = HighResCD(settings)
    
#    changes = highrescd.predict_change_detection(image_before, image, model, uiq2_image, tpi_poly, True, True)
    changes = highrescd.predict_change_detection(image_before, image, uiq2_image, tpi_poly, True, True)


    images = [{'image':image, 'image_before':image_before,'uiq2_image':uiq2_image, 'tpi_poly': tpi_poly, 'confirmation_factor': tpi['confirmation_factor']}]
    return [changes, tpis_skipped, images]

tpi_to_check = [tpi for tpi in tpis_optical if tpi['id'] == 25364][0]
#print(tpi_to_check)

outputs = get_changes_for_tpi(tpi_to_check, settings, customer)

changes = outputs[0]
images  = outputs[2]

print(len(changes))



    


tpi_poly = geometry.Polygon(tpi_to_check['coordinates']).buffer(0.)
tpi_found = False
for change in changes:
        change_poly = geometry.shape(change['poly'])
        if change_poly.intersects(tpi_poly):
            tpi_found = True
            #output['change_detected'] = change
            if tpi_to_check['confirmation_factor'] == 2.0:
                print('tpi ok')
                #change['tpi_id'] = j
                #tpi_changes.append(change)
                
            break


import rasterio
import rasterio.plot
import geopandas as gpd

def print_tpi_on_image(images, key_id):
    
    _, height, width = images[0][key_id].shape
    poly = images[0][key_id].image_poly
    image = (255*images[0][key_id].image[:3]).astype(np.uint8)
    #image_resized = resize(image, (image.shape[0], image.shape[1], image.shape[2]), order=3).astype(np.uint8)
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                   poly.bounds[2], poly.bounds[3],
                                                   width, height)

    new_dataset = rasterio.open('tmp_file.tiff', 'w', driver='GTiff',
                            height=height, width=width,
                            count=3, dtype='uint8',
                            crs='+proj=latlong',
                            transform=geotiff_transform)

    # Write bands
    new_dataset.write(image)
    new_dataset.close()

    src = rasterio.open('tmp_file.tiff')

    poly_changes = geometry.MultiPolygon([change['poly'] for change in changes])

    p = gpd.GeoSeries(poly_changes)

    return p, src


p, src = print_tpi_on_image(images, 'image')

fig, ax = plt.subplots(figsize=(20, 20))
rasterio.plot.show(src, ax=ax, alpha=1)
p.plot(ax=ax, facecolor='none', edgecolor='red')
plt.show()
plt.savefig('image.png')

p, src = print_tpi_on_image(images, 'image_before')

fig, ax = plt.subplots(figsize=(20, 20))
rasterio.plot.show(src, ax=ax, alpha=1)
p.plot(ax=ax, facecolor='none', edgecolor='red')
plt.show()
plt.savefig('image_before.png')




tpi_changes = []
all_changes = []

n_tpis_conf2 = 0

'''
for j, changes in enumerate(tqdm(outputs)):
    if not changes:
         continue
                  
    changes = [ch for ch in changes if not (ch['label_before'] == 'bare ground'    and ch['label_after'] == 'low vegetation')]
    changes = [ch for ch in changes if not (ch['label_before'] == 'bare ground'    and ch['label_after'] == 'trees')]
    changes = [ch for ch in changes if not (ch['label_before'] == 'low vegetation' and ch['label_after'] == 'trees')]
    changes = [ch for ch in changes if not (ch['label_before'] == 'trees'          and ch['label_after'] == 'trees')]
    changes = [ch for ch in changes if not (ch['label_before'] == 'clouds'         and ch['label_after'] == 'clouds')]
    changes = [ch for ch in changes if not (ch['label_before'] != 'clouds'         and ch['label_after'] == 'clouds')]
    
    changes = [ch for ch in changes if not geometry.shape(ch['poly']).area < 1e-9]
    
    changes = [ch for ch in changes if ch['ndvi_change'] < 0.0 ]
    changes = [ch for ch in changes if ch['shadow_before'] > np.exp(-8) and ch['shadow_after'] > np.exp(-6)]

    tpi_poly            = images[j][0]['tpi_poly']
    confirmation_factor = images[j][0]['confirmation_factor']
    
    if confirmation_factor == 2:
        n_tpis_conf2 += 1
    
    changes = sorted(changes, key=lambda x: geometry.shape(x['poly']).area)[::-1]

#     for change in changes:
#         change['shadow_after_diff']  = change['shadow_after'] - images[j][0]['image'].image.mean()
#         change['shadow_before_diff'] = change['shadow_before'] - images[j][0]['image_before'].image.mean()
    all_changes.append(changes)
'''    

