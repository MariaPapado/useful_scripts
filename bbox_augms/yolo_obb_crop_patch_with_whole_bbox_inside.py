##script for images that contain only one oriented bbox
##the code crops a random patch that always contains the whole bbox inside

import cv2
import os
import numpy as np
from ultralytics.data.split_dota import split_test, split_trainval
import math
import random

def center_of_oriented_bbox(coords):
    # Assuming coords is a list of tuples or a list of 4 values [x1, y1, x2, y2, ..., x4, y4]
    x1, y1, x2, y2, x3, y3, x4, y4 = coords

    # Calculate the center (centroid) of the four points
    x_c = (x1 + x2 + x3 + x4) / 4
    y_c = (y1 + y2 + y3 + y4) / 4

    return x_c, y_c

def find_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to read annotations
def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            # Extract coordinates (as float)
            coords = list(map(float, parts[1:]))
            annotations.append((class_id, coords))
    return annotations


def yolo_obb_polygon_line(points: np.ndarray, W: int, H: int, cls: int = 0, normalize: bool = True) -> str:
    pts = points.astype(float).copy()  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    if normalize:
        pts[:, 0] /= W
        pts[:, 1] /= H
        pts = np.clip(pts, 0.0, 1.0)
    flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))  # x1 y1 x2 y2 x3 y3 x4 y4
    return f"{cls} {flat}"

def yolo_obb_polygon_line(points: np.ndarray, W: int, H: int, cls: int = 0, normalize: bool = True) -> str:
    pts = points.astype(float).copy()  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    if normalize:
        pts[:, 0] /= W
        pts[:, 1] /= H
        pts = np.clip(pts, 0.0, 1.0)
    flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))  # x1 y1 x2 y2 x3 y3 x4 y4
    return f"{cls} {flat}"

def extract_patch_with_bbox(image, bbox, patch_size):

    # Image dimensions
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    # Extract bbox coordinates
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    #print('coords', x1,y1,x2,y2,x3,y3,x4,y4)
    
    # Get the center of the bounding box (average of 4 corners)
    bbox_center_x = (x1 + x2 + x3 + x4) / 4
    bbox_center_y = (y1 + y2 + y3 + y4) / 4

    # Get the bounding box width and height
    bbox_width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)  ##d13
    bbox_height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4) ##d24
    

    if bbox_width>patch_width or bbox_height>patch_height:
        print('problem')

    dx_left = min(x1,x2,x3,x4)-(patch_width-bbox_width) 
    dx_right = max(x1,x2,x3,x4) + (patch_width-bbox_width) 
    dy_up = min(y1,y2,y3,y4) - (patch_height-bbox_height) 
    dy_down = max(y1,y2,y3,y4) + (patch_height-bbox_height) 
    print('dxleft', dx_left, 'dxriehgt', dx_right, 'dyup', dy_up, 'dy_down', dy_down)


    dx_left = max(0, dx_left) + patch_width//2
    dy_up = max(0, dy_up)+ patch_height//2

    dx_right = min(dx_right, img_width)- patch_width//2
    dy_down = min(dy_down, img_height)- patch_height//2


    x_center_area = (dx_left+dx_right)//2
    y_center_area = (dy_up+dy_down)//2

    xpc = random.randint(min(dx_left,dx_right), max(dx_left,dx_right))
    ypc = random.randint(min(dy_up,dy_down), max(dy_up,dy_down))

    # Calculate the patch's top-left corner
    x_offset = xpc - patch_width // 2 +2  ##add 2 to avoid edge cases
    y_offset = ypc - patch_height // 2 +2 ##add 2 to avoid edge cases
    print('xoff', 'yoff', x_offset, y_offset)

    # Extract the patch from the image
    patch = image[y_offset:y_offset + patch_height, x_offset:x_offset + patch_width]
    print('patch', patch.shape)


    # Visualize the patch and the bounding box inside it (for testing)
    points = np.array([(x1 - x_offset, y1 - y_offset),
                       (x2 - x_offset, y2 - y_offset),
                       (x3 - x_offset, y3 - y_offset),
                       (x4 - x_offset, y4 - y_offset)], dtype=np.int32)

    # Draw the rotated bounding box on the patch
    #cv2.polylines(patch, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    return patch, points

            


############@@@@@@@@@@@@@@@@@@@@@@@@STILL at one point the bbox is outside the patch

# Define paths
image_dir = "./GRTGAZ2024-1/images/"  # Change to your images directory
label_dir = "./GRTGAZ2024-1/labels/"  # Change to your labels directory

psize = (640,640)

# Example usage
ids = os.listdir('./GRTGAZ2024-1/images/')
#print(ids)
for id in ids:
    if '38_41314_20240625.png' in id:
        img = cv2.imread('./GRTGAZ2024-1/images/{}'.format(id))
        print('imgshape', img.shape)
        annotations = read_annotations('./GRTGAZ2024-1/labels/{}'.format(id[:-4] + '.txt'))
        #print(annotations)
        if len(annotations)==1:  ##check if labels contain only one bounding box
            x1,y1,x2,y2,x3,y3,x4,y4 = annotations[0][1]
        # Convert to int for cv2
            pts = np.array([(int(x1 * img.shape[1]), int(y1 * img.shape[0])),
                            (int(x2 * img.shape[1]), int(y2 * img.shape[0])),
                            (int(x3 * img.shape[1]), int(y3 * img.shape[0])),
                            (int(x4 * img.shape[1]), int(y4 * img.shape[0]))], dtype=np.int32)
#            distance = find_dist(pts[0][0],pts[0][1],pts[2][0],pts[2][1])

#            center = center_of_oriented_bbox(pts.flatten())
#            print(center)
            #print(pts)
            img_x_center = img.shape[1]//2
            img_y_center = img.shape[0]//2
            img_center = (img_x_center, img_y_center)
            bbox = pts.flatten()

            patch, points = extract_patch_with_bbox(img, bbox, psize)
            line = yolo_obb_polygon_line(points, psize[1], psize[0])
            
            cv2.imwrite('patch.png', patch)

            with open("patch.txt","w") as f: f.write(line+"\n")            
            

