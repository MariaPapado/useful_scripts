import cv2
from pathlib import Path
from tqdm import tqdm
import os
import shutil

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

def resize_and_adjust_labels(image_path, label_path, scale, output_dir):
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
    
    new_img_height = int(img_height*0.3/scale)
    new_img_width = int(img_width*0.3/scale)

    resized_image = cv2.resize(image, (new_img_width, new_img_height))

    resized_labels = []
    for class_id, x_center, y_center, width, height in labels:
        # Convert to absolute coordinates
        abs_x1, abs_y1, abs_x2, abs_y2 = yolo_to_absolute(
            x_center, y_center, width, height, img_width, img_height
        )

        resized_x1, resized_y1, resized_x2, resized_y2 = abs_x1*0.3/scale, abs_y1*0.3/scale, abs_x2*0.3/scale, abs_y2*0.3/scale


        # Convert back to YOLO format relative to crop
        norm_x_center, norm_y_center, norm_width, norm_height = absolute_to_yolo(
            resized_x1, resized_y1, resized_x2, resized_y2, new_img_width, new_img_height
        )
        
        resized_labels.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height))

    # Save crop and labels
    if resized_labels or True:  # Save even if no labels
        stem = image_path.stem
        resized_image_path = output_dir / 'images/' / f"{stem}_resized.png"
        resized_label_path = output_dir / 'labels/' f"{stem}_resized.txt"
        
        cv2.imwrite(str(resized_image_path), resized_image)
        
        with open(resized_label_path, 'w') as f:
            for label in resized_labels:
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
        
        resize_and_adjust_labels(image_path, label_path, 1, output_dir)
    print(f"Processed {len(image_files)} images.")

def main():
    
    CONFIG = {
        "image_dir": "./crane_images/",
        "label_dir": "./crane_labels/",
        "output_dir": "./resized/",
    }

    process_dataset(CONFIG["image_dir"], CONFIG["label_dir"], CONFIG["output_dir"])
    print("Processing complete!")

if __name__ == "__main__":
    main()
