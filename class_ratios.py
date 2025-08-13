from PIL import Image
import numpy as np
import os

ids = os.listdir('/home/maria/previews_clouds/new_training/DATA_previews_cvat_download/previews_dataset/LABELS/')

class_sums = [0]*2


for id in ids:
    mask = Image.open('/home/maria/previews_clouds/new_training/DATA_previews_cvat_download/previews_dataset/LABELS/{}'.format(id))
    mask = np.array(mask)/255.
    for c in range(0, len(class_sums)):
        idx = np.where(mask==c)
        count = len(idx[0])
        class_sums[c] = class_sums[c] + count


total_samples = class_sums[0] + class_sums[1]

for i in range(0, len(class_sums)):
    print('class {} '.format(i), class_sums[i]/total_samples)



# Count occurrences of each class
class_counts = np.array(class_sums)
total_samples = class_counts[0] + class_counts[1]

# Calculate class frequencies
class_freqs = class_counts / total_samples
print("Class frequencies:", class_freqs)

# Calculate alpha for each class as the inverse of class frequency
alpha = 1.0 / class_freqs
print("Alpha values:", alpha)


# Normalize alpha to sum to 1
alpha /= alpha.sum()
print("Normalized alpha values:", alpha)
