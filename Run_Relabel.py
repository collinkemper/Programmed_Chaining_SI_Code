import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage import segmentation
from matplotlib.colors import hsv_to_rgb
import Chain_Analysis_Functions as caf
import os


if __name__ == "__main__":
    wafer_number_sf = '53'
    label_mask_name_sf = '53-labels_2.csv'

    print('load data')
    labels = os.path.join(os.getcwd(), 'Vector Magnetometry', wafer_number_sf, label_mask_name_sf)
    chain_mask_sf = np.loadtxt(labels, delimiter=",")
    chain_mask_sf = chain_mask_sf[0:2000,0:2000]
    print('relabel mask')
    
    new_chain_mask_sf = caf.relabel_mask_parallel(chain_mask_sf, small_hole_size = 2, min_region_size=200,
                                            branch_length_fraction=0.025, prune=True, prune_branch_length=0, vertical_prune_length=4,
                                            global_min_branch_length=2, vertical_prune_length2 = 3, debug_plots = False)
    print('relabel finished')

    unique_labels = np.unique(new_chain_mask_sf)
    unique_labels = unique_labels[unique_labels != 0]
    num_labels = len(unique_labels)

    hues = np.linspace(0, 1, num_labels, endpoint=False)
    np.random.seed(np.random.randint(0,100))  # Optional: fix randomness
    np.random.shuffle(hues)  # Shuffle to avoid nearby labels looking similar
    colors = hsv_to_rgb(np.stack([hues, np.ones_like(hues)*0.65, np.ones_like(hues)*0.95], axis=1))

    # Create a mapping from label to color
    label_to_color = {label: np.append(colors[i], 1.0) for i, label in enumerate(unique_labels)}  # RGBA

    # Create the overlay image
    overlay_img_sf = np.zeros((*new_chain_mask_sf.shape, 4), dtype=float)
    for label, rgba in label_to_color.items():
        overlay_img_sf[new_chain_mask_sf == label] = rgba

    # Add black boundaries
    boundaries = segmentation.find_boundaries(new_chain_mask_sf.astype(np.int32), mode='outer')
    overlay_img_sf[boundaries] = [0, 0, 0, 1]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.imshow(overlay_img_sf)
    ax1.set_title('Watershed-filled branches (globally unique colors) with black outlines')
    ax1.axis('off')
    plt.show()
