import pandas as pd
from funcs import *

# Specify the path to the CSV file in the downloads folder
file_path = "mycranes_table.csv"

# Read the CSV file into a pandas dataframe
df = pd.read_csv(file_path, delimiter=",")

# keep only from row number 239 to 265
df = df.iloc[:107, :]

# Display the dataframe
print(df.head())
print("")
incidents = len(df['class'])
print(f'Number of incidents: {incidents}')

# Create a dictionary with per row the customer as key and the TPI as value added to a list
customer_tpi = {}
customer_latlon = {}
excluded_incidents = 0
for index, row in df.iterrows():
    customer = row['customer']
    tpi = row['tpi']
    raw_date1 = str(row['date'])
    dt = datetime.strptime(raw_date1, "%d/%m/%Y")
    date1 = dt.strftime("%Y%m%d")

    date2 = row['date2']
    date3 = row['date3']
    latlon = row['latlon']
    # CHECK IF THE CUSTOMER AND TPI ARE NOT nan
    no_date = pd.isna(date1) and pd.isna(date2) and pd.isna(date3)
    if pd.isna(customer) or pd.isna(tpi) or no_date:
        excluded_incidents += 1
        customer_latlon[latlon] = [date1, date2, date3]
        continue
    else:
        if customer in customer_tpi:
            customer_tpi[customer].append({int(tpi): [date1, date2, date3]})
        else:
            customer_tpi[customer] = [{int(tpi): [date1, date2, date3]}]


print(f'Number of incidents with latlon instead of TPI: {excluded_incidents}')
remaining_incidents = incidents - excluded_incidents
print(f'Number of incidents with TPI: {remaining_incidents}')

# Check if the number of TPIs in the dictionary is correct 
total = 0
print("")
print(customer_tpi)
print("")
print(customer_latlon)
for key, value in customer_tpi.items():
    total += len(value)

check = remaining_incidents == total
if check:
    check = "Yes"

print("")
print(f'Is the number of TPIs in the dictionary correct? {check}')




# Get customer id list
customer_login = ov.get_customerdb_credentials()
creds_mapserver = ov.get_image_broker_credentials()
customer_db = CustomerDatabase(customer_login['username'], customer_login['password'])
customers = customer_db.get_customers()
projects = sum([x['projects'] for x in customers], [])  # Flatten list of projects

# # Uncomment to test for one customer
# customer = 'GRTGAZ-2024'
# tpi_id = 38008
# Loop through incidents dictionary




for customer, tpis in customer_tpi.items():
  if customer!='GRTGAZ-2024':  

    print(f"[INFO] Processing {customer}")
    # Get project by name and active status
    customer_db = CustomerDatabase(customer_login['username'],  customer_login['password'])
    project = customer_db.get_project_by_name(customer)
    project_id = project['id']

    settings = get_settings(customer)
#    if not settings['project']['flags']['use_geoserve_hosting']:
    creds_mapserver = ov.get_image_broker_credentials()
    creds_geoserve = None
#    else:
#        creds_geoserve = ov.get_geoserve_credentials(customer)
#        creds_mapserver = ov.get_image_broker_credentials()


    for tpi_dict in tpis: 
        for tpi_id, tpi_date_list in tpi_dict.items():
            for tpi_date in tpi_date_list:
                if not pd.isna(tpi_date):
                    #if tpi_id==65782:
                    
                        print("")
                        print(f"[INFO] Processing {customer} - {tpi_id} - {tpi_date}")

                        # Get image for project and TPI
                        with RegionsDb(ov.get_sarccdb_credentials()) as db:
                            sql = """
                                        SELECT tpi_id, geometry, source_date, source from tpi_dashboard.indicators where project_id = %s and tpi_id = %s
                                        """
                            arguments = [project_id, tpi_id]
                            db._cur.execute(sql, arguments)
                            results = db._cur.fetchall()[0]
                            result = {'id': results[0], 'geometry': wkb.loads(results[1], hex=True), 'source_date': results[2], 'source': results[3]} 

                        # verander dit zodat ie de referentie datum van de TPI pakt!!!!!!
                        reference_date = datetime.strptime(str(int(tpi_date)), '%Y%m%d')

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
                            Image.fromarray(image).save(f'./Crane_tpis/{customer}_{tpi_id}_{tpi_date}.png')
                            print(f"[INFO] Image saved as {customer}_{tpi_id}_{tpi_date}.png")
