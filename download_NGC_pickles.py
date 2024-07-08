import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely import geometry
from datetime import datetime, timedelta
from pimsys.regions.RegionsDb import RegionsDb
import torch
from skimage import io

from customer import Customer, convert_wgs_to_utm
from high_res_image import Image
from high_res_change_detection import HighResCD
#import os
#import sys
#directory_path = os.path.abspath('/home/jovyan/highreschange/models')
#if directory_path not in sys.path:
#    sys.path.append(directory_path)

################################################

################################################

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

interval_end = datetime(2022, 12, 31)  # datetime.now() - timedelta(days=9*interval_days)
interval_start = datetime(2022, 1, 1) #interval_end - timedelta(days=interval_days)
interval_utc = [get_utc_timestamp(interval_start), get_utc_timestamp(interval_end)]

# Get optical User Made TPIs
tpis_optical = customer.get_usermade_tpis(interval_utc)
print(str(interval_start.date()) + ' - ' + str(interval_end.date()))
print(len(tpis_optical))

def get_image_and_reference_for_tpi(tpi, settings, customer):
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
    
    if image_before is None:
        return [[],tpis_skipped, []]
    
    image_before.tile_image()
    image_before.predict_clouds()

    highrescd = HighResCD(settings)
    
#    changes = highrescd.predict_change_detection(image_before, image, model, uiq2_image, tpi_poly, True, True)
    #changes = highrescd.predict_change_detection(image_before, image, uiq2_image, tpi_poly, True, True)


    images = [{'image':image, 'image_before':image_before,'uiq2_image':uiq2_image, 'tpi_poly': tpi_poly, 'confirmation_factor': tpi['confirmation_factor']}]
    return [tpis_skipped, images]


from joblib import Parallel, delayed
import pickle
outputs_all = Parallel(n_jobs=2, prefer='processes')(delayed(get_image_and_reference_for_tpi)(tpi, settings, customer) for tpi in tqdm(tpis_optical))
#outputs = []
counter = 0
for outputs_tmp in tqdm(outputs_all):
#    outputs_tmp = get_changes_for_tpi(tpi, settings, customer)
    print('infor')
    with open('./results/outputs_ngc_tpis'+str(counter)+'.p', 'wb') as f:
        print('inpickle')
        pickle.dump(outputs_tmp, f, protocol=4)
    counter += 1
    #outputs.append(outputs_tmp) 



