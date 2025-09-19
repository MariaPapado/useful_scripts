import os
import cv2
import json
import psycopg
import requests
import numpy as np
import rasterio.features
import rasterio.transform
import orbital_vault as ov
from CustomerDatabase import CustomerDatabase
from tqdm import tqdm
from datetime import datetime
from shapely import geometry, wkb
from geopy.distance import distance
from pimsys.regions.RegionsDb import RegionsDb
from funcs import *

def download_from_geoserve(image, region, width, height, auth=None):

    arguments = {"layer_name": image["wms_layer_name"], "bbox": "%s,%s,%s,%s" % (region[0], region[1], region[2], region[3]), "width": width, "height": height,}

    if "image_url" in image.keys() and image["image_url"] is not None:
        image_url = image["image_url"]
        if not image_url[-1] == "?":
            image_url = image_url + "?"
        arguments["bbox"] = "%s,%s,%s,%s" % (region[1], region[0], region[3], region[2],)
        url = (image_url + "&VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&CRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments)

    try:
        resp = requests.get(url, auth=auth)
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]
    except:
        print(resp)
        print(url)
        return None
    
    return image 

def download_from_mapserver(image, region, width, height, auth=None):
    url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_"
    image_url = url + f'{image["wms_layer_name"]}.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX={region[0]},{region[1]},{region[2]},{region[3]}&WIDTH={width}&HEIGHT={height}&FORMAT=image/png&LAYERS={image["wms_layer_name"]}'
    resp = requests.get(image_url, auth=auth)
    try:
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]
    except:
        print(image_url)
        return None
    
    return image


def download_image(wms_image, indicator_window, width, height, creds_mapserver, creds_geoserve):
    if wms_image['downloader'] == 'geoserve':
        if creds_geoserve is None:
            image = download_from_geoserve(wms_image, indicator_window.bounds, width, height)
        else:
            image = download_from_geoserve(wms_image, indicator_window.bounds, width, height, auth=(creds_geoserve['user'], creds_geoserve['password']))

    else:
        image = download_from_mapserver(wms_image, indicator_window.bounds, width, height, auth=(creds_mapserver['username'], creds_mapserver['password']))
    return image

def get_wms_image_by_id(database, image_id, creds_mapserver):
    image_broker_url = 'https://maps.orbitaleye.nl/image-broker/products?id={}&_skip=0'.format(image_id)
    response = requests.get(image_broker_url, auth=(creds_mapserver['username'], creds_mapserver['password']))
    wms_image = json.loads(response.text)[0]

    wms_image = database.get_optical_image_by_wms_layer_name(wms_image['wms_layer_name'])
        
    return wms_image

def get_bbox(coords, image_buffer=0.128):
    tpi_buffer_north = distance(kilometers=image_buffer).destination((coords[1], coords[0]), bearing=0)
    tpi_buffer_east = distance(kilometers=image_buffer).destination((coords[1], coords[0]), bearing=90)
    vertical_buffer = tpi_buffer_north[0] - coords[1]
    horizontal_buffer = tpi_buffer_east[1] - coords[0]
    tpi_north = coords[1] + vertical_buffer
    tpi_east = coords[0] + horizontal_buffer
    tpi_south = coords[1] - vertical_buffer
    tpi_west = coords[0] - horizontal_buffer
    tpi_region = geometry.Polygon([[tpi_east, tpi_north], [tpi_west, tpi_north], [tpi_west, tpi_south],[tpi_east, tpi_south], [tpi_east, tpi_north]])
    return tpi_region

def get_mask_valid_area(wms_image, window, width, height):
    valid_area = wms_image['valid_area']
    geotiff_transform = rasterio.transform.from_bounds(*window.bounds, width, height)

    mask_valid = rasterio.features.rasterize([valid_area], out_shape=(width, height), transform=geotiff_transform).astype(bool)
    
    return mask_valid


creds = ov.get_sarccdb_credentials()
creds_mapserver = ov.get_image_broker_credentials()

customer_db_creds = ov.get_customerdb_credentials()
customer_db = CustomerDatabase(customer_db_creds['username'], customer_db_creds['password'])

width, height = 512, 512
output_folder = './dataset_sql/'
os.makedirs(output_folder, exist_ok=True)

connection_string = f"dbname={creds['database']} host={creds['host']} password={creds['password']} port={creds['port']} user={creds['user']}"


# GRTGAZ-2022          40
# GRTGAZ-2023-Lyon     18
# GRTGAZ-2023-Nantes   19
# GRTGAZ-2023-Paris     17
#GRTGAZ-2024            38  !! exclude december
##look also for RTE!!!!


project_id = 77
query = """SELECT tpi_id,project_id,geometry,latitude,longitude,source_date,description FROM tpi_dashboard.indicators WHERE project_id = 77 AND source = 'optical' AND description LIKE '%construction%'
"""

with psycopg.connect(connection_string) as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        data_tmp = cur.fetchall()
        desc = cur.description
        col_names = [x[0] for x in desc]
        results = [dict(zip(col_names, x)) for x in data_tmp]

#if project_id==38:
#    for res in results:
#        dt = datetime.strptime(str(res['source_date']), "%Y-%m-%d %H:%M:%S")
#        formatted_date = dt.strftime("%Y%m%d")
#        if formatted_date > '20241201':
#            results.remove(res)


print('construction samples found: ', len(results))

for res in results:
    res['geometry'] = wkb.loads(bytes.fromhex(res['geometry']))
    

project = customer_db.get_project(project_id)

result_project = results #[res for res in results if res['project_id'] == project_id]
print('len', len(result_project))

creds_geoserve = None
for res in tqdm(result_project):

    dt = datetime.strptime(str(res['source_date']), "%Y-%m-%d %H:%M:%S")
    formatted_date = dt.strftime("%Y%m%d")

    print("")
    print(f"[INFO] Processing {res['source_date']} - {formatted_date}")

    # Get image for project and TPI
    with RegionsDb(ov.get_sarccdb_credentials()) as db:
        sql = """
                    SELECT tpi_id, geometry, source_date, source from tpi_dashboard.indicators where project_id = %s and tpi_id = %s
                    """
        arguments = [project_id, res['tpi_id']]
        db._cur.execute(sql, arguments)
        results = db._cur.fetchall()[0]
        result = {'id': results[0], 'geometry': wkb.loads(results[1], hex=True), 'source_date': results[2], 'source': results[3]} 

    # verander dit zodat ie de referentie datum van de TPI pakt!!!!!!
    reference_date = datetime.strptime(str(int(formatted_date)), '%Y%m%d')

    # Get images from image broker
    image_broker = {'user': 'mapserver_user',
                    'password': 'tL3n3uoPnD8QYiph',
                    'cosmic-eye-api': 'https://maps.orbitaleye.nl/image-broker/optical_images_covering_geometry?_geometry=POINT%20({}%20{})',
                    'admin-api': 'https://maps.orbitaleye.nl/image-broker/products?wms_layer_name={}&_skip=0'}

    if result['geometry'].geom_type == 'Polygon':
        tpi = result['geometry'].centroid
    else: 
        tpi = result['geometry']

    url_request = image_broker['cosmic-eye-api'].format(tpi.x, tpi.y)
    response = requests.get(url_request, auth=(image_broker['user'], image_broker['password']))
    wms_images = json.loads(response.text)

    # find image where capture_timstap is closest to reference_date
    closest_image = min(wms_images, key=lambda x: abs(datetime.fromisoformat(x['capture_timestamp']) - reference_date))

    with RegionsDb(ov.get_sarccdb_credentials()) as db:
        wms_layer_name = db.get_optical_image_by_wms_layer_name(closest_image['wms_layer_name'])

    width, height = 1024, 1024

    print(closest_image)
    image_buffer = round(width * closest_image['pixel_resolution_y'] / 2.)

    utm_epsg = convert_wgs_to_utm(tpi.x, tpi.y)
    proj = pyproj.Transformer.from_crs(4326, utm_epsg, always_xy=True)
    proj_inverse = pyproj.Transformer.from_crs(utm_epsg, 4326, always_xy=True)

    indicator_point_utm = ops.transform(proj.transform, geometry.Point(tpi))
    indicator_window_utm = indicator_point_utm.buffer(image_buffer)
    indicator_window = ops.transform(proj_inverse.transform, indicator_window_utm)

    indicator_window = move_small_square_inside(wms_layer_name['valid_area'], indicator_window)

    image = download_image(closest_image, indicator_window, width, height, creds_mapserver, creds_geoserve)
    if image is None:
        print("[INFO] Failed to download image!")
    else:   
        from PIL import Image
        Image.fromarray(image).save('./dataset_77_date/{}_{}_{}.png'.format(project_id, res['tpi_id'], formatted_date))
        print("[INFO] Image saved as {}_{}_{}.png".format(project_id, res['tpi_id'], formatted_date))
