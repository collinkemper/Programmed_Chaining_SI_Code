import numpy as np
import matplotlib.pyplot as plt
import os
import Chain_Analysis_Functions as caf
from PIL import Image
import time

if __name__ == "__main__":

    original_image_name = '53-connectors.jpg'
    image_path = os.path.join(os.getcwd(), '0.2', original_image_name)
    original_image = Image.open(image_path).convert('RGB')
    mask_name = '53-connectors_cleaned_mask_3.png'
    filtered_particle_mask_name = os.path.join(os.getcwd(), '0.2', mask_name)
    filtered_particle_mask = Image.open(filtered_particle_mask_name).convert('L')
    filtered_particle_mask = np.array(filtered_particle_mask) == 255
    caf.display_mask(filtered_particle_mask)

    particle_bounds = (10,250,0,250)
    t0 = time.time()
    chain_mask = caf.label_mask_parallel(filtered_particle_mask, original_image, particle_bounds, disk_size=1, connectivity=1,
                                                                branch_length_fraction=0.007, global_min_branch_length=2, min_region_size=200, debug_plots = False,
                                                                prune=True, prune_branch_length=5, max_hole_size=20, vertical_prune_length = 4)
    t2 = time.time()
    print(t2-t0)

    np.savetxt("53-labels.csv", chain_mask, delimiter=",", fmt="%g")