from xml.dom import minidom
import xml.etree.ElementTree as ET
from scipy.ndimage import label
from PIL import Image
import numpy as np
import os


class_labels_out = {
    1: "cloud"
}

def encode_cvat_rle_mask(mask):
    if mask[0] == 1:
        rle_encoded = [0]
    else:
        rle_encoded = []
    current_value = mask[0]
    current_length = 1

    for i in range(1, len(mask)):
        if mask[i] == current_value:
            current_length += 1
        else:
            rle_encoded.append(current_length)
            current_value = mask[i]
            current_length = 1

    # Append the last run
    rle_encoded.append(current_length)

    # Convert to CVAT format
    cvat_encoded = ",".join(str(length) for length in rle_encoded)
    return cvat_encoded


def create_annotations(output_changes, base_xml_filepath, output_xml_filepath, class_labels):
    base_tree = ET.parse(base_xml_filepath)
    # Get root
    root_tree = base_tree.getroot()

    for change in output_changes:
        # Create polygons
        polygons = []

        for class_label in class_labels.keys():
            if class_label == 0:
                continue
            # Create the image mask
            mask = (change["change"] == class_label).astype(np.uint8)
            # label change in change blob
            blobs, num = label(mask, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
            labels = list(range(1, num + 1))

            # Find all mask labels
            for label_i in labels:
                mask_i = blobs == label_i

                # Find the indices of True values
                true_indices = np.argwhere(mask_i)

                # Get the bounding box of the True region
                min_row, min_col = np.min(true_indices, axis=0)
                max_row, max_col = np.max(true_indices, axis=0)

                # Crop the boolean array to the bounding box
                cropped_array = mask_i[min_row : max_row + 1, min_col : max_col + 1]

                cropped_width = cropped_array.shape[1]
                cropped_height = cropped_array.shape[0]

                coords_str = encode_cvat_rle_mask(cropped_array.flatten())

                polygons.append(
                    {
                        "label": class_labels[class_label],
                        "points": coords_str,
                        "width": cropped_width,
                        "height": cropped_height,
                        "left": min_col,
                        "top": min_row,
                    }
                )

        id_num = len(root_tree.findall("image"))

        subimage = ET.SubElement(root_tree, "image")
        subimage.set("id", str(id_num))
        subimage.set("name", change["name"])
        subimage.set("width", "512")
        subimage.set("height", "512")

        for polygon in polygons:
            subpoly = ET.SubElement(subimage, "mask")
            subpoly.set("label", polygon["label"])
            subpoly.set("rle", str(polygon["points"]))
            subpoly.set("left", str(polygon["left"]))
            subpoly.set("top", str(polygon["top"]))
            subpoly.set("width", str(polygon["width"]))
            subpoly.set("height", str(polygon["height"]))
            subpoly.set("occluded", "0")
            subpoly.set("z_order", "0")

    xmlstr = minidom.parseString(ET.tostring(root_tree)).toprettyxml(indent="   ")
    with open(output_xml_filepath, "w") as f:
        f.write(xmlstr)

ids = os.listdir('./images/')
outputs = []
for id in ids:
###########################################
##########################################
  mask_valid = Image.open('./masks/{}'.format(id))
  mask_valid = np.array(mask_valid)
  idx = np.where(mask_valid==255)
  mask_valid[idx]=1
  base_xml_filepath = "annotations_base.xml"

  #for out in outputs:
  outputs.append({"name": 'images/{}'.format(id), "change": mask_valid})

create_annotations(outputs, base_xml_filepath, 'cloud2.xml', class_labels_out)
