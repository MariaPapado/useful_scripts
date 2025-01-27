import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

def decode_cvat_rle_mask(rle_string, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Split RLE string into pairs of numbers
    pairs = list(map(int, rle_string.split(',')))
    
    # Initialize variables
    pos = 0
    current_value = 0
    
    # Decode RLE
    for length in pairs:
        for i in range(length):
            x = pos % width
            y = pos // width
            mask[y, x] = current_value
            pos += 1
        current_value = 255 if current_value == 0 else 0
    
    return mask

# Parse the XML file
def read_xml_annotations(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    # List to store results
    image_data_list = []

    # Iterate over each 'image' element
    for image in root.findall('.//image'):
        image_data = {}
        image_data['id'] = image.get('id')
        image_data['name'] = image.get('name')
        image_data['width'] = image.get('width')
        image_data['height'] = image.get('height')

        # List to store masks data for current image
        masks_data_list = []
        polys_data_list = []
        # Iterate over each 'mask' element within the 'image' element
        for mask in image.findall('mask'):
            mask_data = {}
            mask_data['label'] = mask.get('label')
            mask_data['source'] = mask.get('source')
            mask_data['occluded'] = mask.get('occluded')
            mask_data['rle'] = mask.get('rle')
            mask_data['left'] = mask.get('left')
            mask_data['top'] = mask.get('top')
            mask_data['width'] = mask.get('width')
            mask_data['height'] = mask.get('height')
            mask_data['z_order'] = mask.get('z_order')

            masks_data_list.append(mask_data)
            
        for poly in image.findall('polygon'):
            poly_data = {}
            poly_data['label'] = poly.get('label')
            poly_data['source'] = poly.get('source')
            poly_data['occluded'] = poly.get('occluded')
            poly_data['points'] = poly.get('points')
            poly_data['z_order'] = poly.get('z_order')

            polys_data_list.append(poly_data)

        image_data['masks'] = masks_data_list
        image_data['polys'] = polys_data_list
        image_data_list.append(image_data)
        
    return image_data_list

class_labels = {
    0: "background",
    1: "cloud",
}

def color_map():
    cmap = np.zeros((7, 3), dtype=np.uint8)
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 255, 255])

    return cmap

cmap = color_map()

def parse_polygon_coordinates(points_string):
    """
    Parse polygon coordinates from the points string.
    
    Args:
        points_string (str): String containing coordinates of the polygon.
        
    Returns:
        list of tuples: List of (x, y) coordinate pairs.
    """
    coordinates = []
    points = points_string.split(';')
    for point in points:
        x, y = map(float, point.split(','))
        coordinates.append((int(x), int(y)))
    return coordinates

def create_mask_from_polygon(coordinates, width, height):
    """
    Create a binary mask from polygon coordinates.
    
    Args:
        coordinates (list of tuples): List of (x, y) coordinate pairs of the polygon.
        width (int): Width of the mask.
        height (int): Height of the mask.
        
    Returns:
        numpy.ndarray: Binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def get_mask_from_annotation(annotations):

    width, height = 512, 512
    mask_total = np.zeros((width, height),dtype=np.uint8)

    for ann in annotations:

        for mask_rle in ann['masks']:
            mask = decode_cvat_rle_mask(mask_rle['rle'], int(mask_rle['width']), int(mask_rle['height'])).astype(bool)
            mask = class_labels[mask_rle['label']]*mask
            mask_total[int(mask_rle['top']):int(mask_rle['top'])+int(mask_rle['height']),int(mask_rle['left']):int(mask_rle['left'])+int(mask_rle['width'])][mask!=0] = mask[mask!=0]#.copy()

        for poly_string in ann['polys']:
            polygon_coordinates = parse_polygon_coordinates(poly_string['points'])
            mask_from_polygon = create_mask_from_polygon(polygon_coordinates, width, height).astype(bool)
            mask_from_polygon = (class_labels[poly_string['label']]*mask_from_polygon).astype(np.uint8)
            mask_total[mask_from_polygon!=0] = mask_from_polygon[mask_from_polygon!=0]#.copy()

    return mask_total
    
    
dataset_directory = './cvat/'

annotation_folder = 'annotated/' 

output_dataset_directory = './cvat/dataset/'

for folder in ['train','val','test']:
    os.makedirs(output_dataset_directory+folder+'/im1/', exist_ok=True)
    os.makedirs(output_dataset_directory+folder+'/im2/', exist_ok=True)
    os.makedirs(output_dataset_directory+folder+'/label1/', exist_ok=True)
    os.makedirs(output_dataset_directory+folder+'/label2/', exist_ok=True)
    os.makedirs(output_dataset_directory+folder+'/label1_rgb/', exist_ok=True)
    os.makedirs(output_dataset_directory+folder+'/label2_rgb/', exist_ok=True)
    
annotated_files = sorted(os.listdir(dataset_directory+annotation_folder))

max_list = []

for annotated_file in annotated_files:
    annotated_folder = [folder for folder in os.listdir(dataset_directory) if os.path.isdir(dataset_directory+folder) and annotated_file.split('.')[0].split('_')[1] in folder][0]

    if 'train' in annotated_file:
        folder_todo = 'train'
    elif 'val' in annotated_file:
        folder_todo = 'val'
    else:
        folder_todo = 'test'
    
    images_before_path = sorted([file for file in os.listdir(dataset_directory+annotated_folder) if "_0.png" in file])
    images_after_path  = sorted([file for file in os.listdir(dataset_directory+annotated_folder) if "_1.png" in file])
    
    assert(len(images_before_path) == len(images_after_path))
    
    for i in tqdm(range(len(images_before_path))):
        
        image_before = Image.open(dataset_directory+annotated_folder+'/'+images_before_path[i])
        image_after  = Image.open(dataset_directory+annotated_folder+'/'+images_after_path[i])

        image_data_list = read_xml_annotations(dataset_directory+annotation_folder+annotated_file)

        annotations_before = [image_data for image_data in image_data_list if image_data['name'].split('/')[-1] == images_before_path[i]]
        annotations_after  = [image_data for image_data in image_data_list if image_data['name'].split('/')[-1] == images_after_path[i]]

        if len(annotations_before) > 0:
            mask_before = get_mask_from_annotation(annotations_before)
            mask_after  = get_mask_from_annotation(annotations_after)
            
            if not ((mask_before>0)==(mask_after>0)).all():
                mask_or = np.logical_or((mask_before>0),(mask_after>0))
                mask_xor_after = np.logical_xor(mask_or,(mask_after>0))
                mask_xor_before = np.logical_xor(mask_or,(mask_before>0))
                mask_or_2 = np.logical_or((mask_xor_after>0),(mask_xor_before>0))
                mask_before[mask_or_2] = 0
                mask_after[mask_or_2] = 0

            assert(((mask_before==0)==(mask_after==0)).all())
            
            image_before.save(output_dataset_directory+folder_todo+'/im1/'+images_before_path[i].replace('_0.png','.png'))
            image_after.save( output_dataset_directory+folder_todo+'/im2/'+images_after_path[i].replace('_1.png','.png'))

            Image.fromarray(mask_before).save(output_dataset_directory+folder_todo+'/label1/'+images_before_path[i].replace('_0.png','.png'))
            Image.fromarray(mask_after).save(output_dataset_directory+folder_todo+'/label2/'+images_after_path[i].replace('_1.png','.png'))

            mask_1 = Image.fromarray(mask_before.astype(np.uint8), mode="P")
            mask_1.putpalette(cmap)

            mask_2 = Image.fromarray(mask_after.astype(np.uint8), mode="P")
            mask_2.putpalette(cmap)
            
            mask_1.save(output_dataset_directory+folder_todo+'/label1_rgb/'+images_before_path[i].replace('_0.png','.png'))
            mask_2.save(output_dataset_directory+folder_todo+'/label2_rgb/'+images_after_path[i].replace('_1.png','.png'))
            
            max_list.append(mask_before.max())
            max_list.append(mask_after.max())
