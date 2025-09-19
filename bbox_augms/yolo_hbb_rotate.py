import cv2
from pathlib import Path
from tqdm import tqdm
import os
import shutil
import numpy as np

def parse_yolo_label(label_line):
    """Parse a YOLO format label line"""
    parts = label_line.strip().split()
    class_id = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:5])
    return class_id, x_center, y_center, width, height

def yolo_to_absolute(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO normalized coordinates to absolute coordinates"""
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return x1, y1, x2, y2

def absolute_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """Convert absolute coordinates to YOLO normalized coordinates"""
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def rotate_and_adjust_labels(image_path, label_path, scale, output_dir):
    """Crop image into squares and adjust labels accordingly"""

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    
    # Read labels
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    labels.append(parse_yolo_label(line))
    k=3
    rot_image = np.rot90(image, k=k)

    rot_labels = []
    for class_id, x_center, y_center, w, h in labels:
        if k == 1:  # 90 degrees rotation
            new_x_center = y_center
            new_y_center = 1 - x_center
            new_w = h  # Swap width and height
            new_h = w
        elif k == 2:  # 180 degrees rotation
            new_x_center = 1 - x_center
            new_y_center = 1 - y_center
            new_w = w
            new_h = h
        elif k == 3:  # 270 degrees rotation
            new_x_center = 1 - y_center
            new_y_center = x_center
            new_w = h  # Swap width and height
            new_h = w
            
        rot_labels.append((class_id, new_x_center, new_y_center, new_w, new_h))

    # Save crop and labels
    if rot_labels or True:  # Save even if no labels
        stem = image_path.stem
        rot_image_path = output_dir / 'images/' / f"{stem}_rot.png"
        rot_label_path = output_dir / 'labels/' f"{stem}_rot.txt"
        
        cv2.imwrite(str(rot_image_path), rot_image)
        
        with open(rot_label_path, 'w') as f:
            for label in rot_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        





def process_dataset(image_dir, label_dir, output_dir):
    """Process entire dataset"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    output_dir_images = output_dir / 'images/'
    if os.path.exists(output_dir_images):
        shutil.rmtree(output_dir_images)
    os.makedirs(output_dir_images)

    output_dir_labels = output_dir / 'labels/'
    if os.path.exists(output_dir_labels):
        shutil.rmtree(output_dir_labels)
    os.makedirs(output_dir_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]


    for image_path in tqdm(image_files, desc="Processing images"):
      if '2544' in str(image_path):
        # Find corresponding label file
        label_path = label_dir / f"{image_path.stem}.txt"
        
        rotate_and_adjust_labels(image_path, label_path, 1, output_dir)
    print(f"Processed {len(image_files)} images.")

def main():
    
    CONFIG = {
        "image_dir": "./crane_images/",
        "label_dir": "./crane_labels/",
        "output_dir": "./rotated/",
    }

    process_dataset(CONFIG["image_dir"], CONFIG["label_dir"], CONFIG["output_dir"])
    print("Processing complete!")

if __name__ == "__main__":
    main()
