import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import copy
import os
from skimage.measure import label
from skimage import morphology, measure, color, segmentation
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_fill_holes, rotate, label, uniform_filter, binary_opening, binary_closing, generate_binary_structure,distance_transform_edt, convolve
from skimage.transform import resize
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, binary_dilation, thin, disk, square
from skimage.feature import corner_peaks
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.cm as cm
import colorsys
import random
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
import multiprocessing as mp
import re
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares


def rgb_to_hsv(rgb):

    '''Convert an RGB image represented as a NumPy array into the HSV color space.
        Input:
            rgb - NumPy array containing RGB image data, typically with shape height, width, 3), where the last dimension contains the red, green,
            and blue channel values. The RGB values are expected to be normalized to the range [0, 1].
        Output:
            out - NumPy array with the same shape and data type as rgb, containing
            HSV image data. The three channels represent hue, saturation,
            and value, respectively, with each channel normalized to [0, 1].
    '''
    # Create an output array of zeros with the same shape and data type as the input RGB array.
    out = np.zeros_like(rgb)
    # Extract the red, green, and blue channels from the final dimension of the RGB image array.
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Determine the maximum RGB value at each pixel, which corresponds to the HSV value (brightness).
    maxc = np.max(rgb, axis=-1)
    # Determine the minimum RGB value at each pixel, which is used to calculate saturation and hue.
    minc = np.min(rgb, axis=-1)
    # Set the HSV value channel equal to the maximum RGB channel value at each pixel.
    v = maxc
    # Calculate saturation as the range of RGB values divided by the maximum RGB value.
    s = (maxc - minc) / (maxc + 1e-8)
    # Set saturation to zero for pixels with zero value, preventing undefined saturation for black pixels.
    s[maxc == 0] = 0

    # Calculate the normalized difference between the maximum RGB value and the red channel.
    rc = (maxc - r) / (maxc - minc + 1e-8)
    # Calculate the normalized difference between the maximum RGB value and the green channel.
    gc = (maxc - g) / (maxc - minc + 1e-8)
    # Calculate the normalized difference between the maximum RGB value and the blue channel.
    bc = (maxc - b) / (maxc - minc + 1e-8)

    # Initialize the hue channel as an array of zeros with the same shape as the maximum-value array.
    h = np.zeros_like(maxc)
    # Calculate hue for pixels where red is the maximum RGB channel.
    h[(maxc == r)] = (bc - gc)[(maxc == r)]
    # Calculate hue for pixels where green is the maximum RGB channel.
    h[(maxc == g)] = 2.0 + (rc - bc)[(maxc == g)]
    # Calculate hue for pixels where blue is the maximum RGB channel.
    h[(maxc == b)] = 4.0 + (gc - rc)[(maxc == b)]
    # Normalize hue from the six-sector representation to the range [0, 1].
    h = (h / 6.0) % 1.0
    # Set hue to zero for pixels with no RGB variation, since hue is undefined for grayscale pixels.
    h[minc == maxc] = 0

    # Store the calculated hue values in the first channel of the output HSV array.
    out[..., 0] = h
    # Store the calculated saturation values in the second channel of the output HSV array.
    out[..., 1] = s
    # Store the calculated value/brightness values in the third channel of the output HSV array.
    out[..., 2] = v
    # Return the HSV image as a NumPy array with the same dimensions and data type as the input.
    return out

def create_binary_mask(image_path, brightness=1.0, contrast=1.0, saturation=1.0,
                       temperature=0, R_min=0, G_min=0, B_min=30, V_min=0.1,
                       method="adaptive", adaptive_block_size=10, adaptive_offset=0.01):
    '''
    Create a binary mask from an input image using manual RGB/HSV thresholding, Otsu thresholding, or adaptive thresholding.
        Input:
            image_path - String containing the file path to the input image.
            output_extension - String specifying the desired output file extension. This parameter is currently not used within the function.
            brightness - Float controlling the brightness adjustment applied to the image, where 1.0 leaves the brightness unchanged.
            contrast - Float controlling the contrast adjustment applied to the image, where 1.0 leaves the contrast unchanged.
            saturation - Float controlling the color saturation adjustment applied to the image, where 1.0 leaves the saturation unchanged.
            temperature - Numeric value controlling the red/blue temperature shift applied to the image. Positive values increase red and decrease blue.
            R_min - Numeric minimum threshold for the red channel when using the manual thresholding method.
            G_min - Numeric minimum threshold for the green channel when using the manual thresholding method.
            B_min - Numeric minimum threshold for the blue channel when using the manual thresholding method.
            V_min - Float minimum value/brightness threshold in the HSV color space when using the manual thresholding method.
            method - String specifying the thresholding method. Accepted values are 'manual', 'otsu', or 'adaptive'.
            adaptive_block_size - Integer specifying the size of the local neighborhood used to calculate the mean for adaptive thresholding.
            adaptive_offset - Float added to the local mean when determining the threshold for adaptive thresholding.
        Output:
            mask - NumPy boolean array with the same height and width as the input image. True pixels represent regions that pass the selected thresholding criteria, while False pixels represent regions that do not.
    '''
    # Open the input image from the specified file path and convert it to RGB format.
    original_image = Image.open(image_path).convert('RGB')
    # Adjust the brightness of the image according to the specified brightness factor.
    image = ImageEnhance.Brightness(original_image).enhance(brightness)
    # Adjust the contrast of the brightness-adjusted image according to the specified contrast factor.
    image = ImageEnhance.Contrast(image).enhance(contrast)
    # Adjust the color saturation of the contrast-adjusted image according to the specified saturation factor.
    image = ImageEnhance.Color(image).enhance(saturation)

    # Apply temperature shift
    # Convert the PIL image into a NumPy array of 32-bit floating-point values for numerical manipulation.
    image_np = np.array(image).astype(np.float32)
    # Increase the red channel by the temperature value while restricting values to the valid RGB range of 0 to 255.
    image_np[:, :, 0] = np.clip(image_np[:, :, 0] + temperature, 0, 255)  # R
    # Decrease the blue channel by the temperature value while restricting values to the valid RGB range of 0 to 255.
    image_np[:, :, 2] = np.clip(image_np[:, :, 2] - temperature, 0, 255)  # B

    # Grayscale version for Otsu or adaptive methods
    # Convert the RGB image to grayscale using standard luminance weights for the red, green, and blue channels.
    # Divide by 255 to normalize the grayscale intensity values to the range [0, 1].
    gray = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0  # Normalize to [0,1]

    # Select the thresholding method specified by the user.
    if method == "manual":
        # === Manual RGB + HSV thresholding ===
        # Extract the red, green, and blue channels from the image array.
        r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
        # Create an RGB mask containing pixels whose red, green, and blue values all exceed their respective minimum thresholds.
        rgb_mask = (r > R_min) & (g > G_min) & (b > B_min)

        # Normalize the RGB image from the range [0, 255] to [0, 1] for HSV conversion.
        norm = image_np / 255.0
        # Convert the normalized RGB image into HSV color space using the rgb_to_hsv function.
        hsv = rgb_to_hsv(norm)
        # Extract the value/brightness channel from the HSV image.
        v = hsv[:, :, 2]
        # Create an HSV mask containing pixels whose value exceeds the specified minimum threshold.
        hsv_mask = v > V_min

        # Combine the RGB and HSV masks so that a pixel must satisfy both thresholding conditions.
        mask = rgb_mask & hsv_mask

    # Use Otsu's method to automatically determine a global grayscale threshold.
    elif method == "otsu":
        # Calculate the Otsu threshold that best separates the grayscale image into two intensity classes.
        threshold = threshold_otsu(gray)
        # Create a binary mask containing pixels whose grayscale intensity exceeds the Otsu threshold.
        mask = gray > threshold

    # Use adaptive thresholding based on the local mean grayscale intensity.
    elif method == "adaptive":
        # Calculate the mean grayscale intensity within a local neighborhood around each pixel.
        local_mean = uniform_filter(gray, size=adaptive_block_size)
        # Create a binary mask containing pixels brighter than their local mean plus the specified offset.
        mask = gray > (local_mean + adaptive_offset)

    # Raise an error if the supplied thresholding method is not recognized.
    else:
        raise ValueError("Unknown method. Choose from: 'manual', 'otsu', or 'adaptive'.")

    # Convert the boolean mask into an 8-bit grayscale image where False = 0 (black) and True = 255 (white).
    mask_img = (mask * 255).astype(np.uint8)

    # === Display ===
    # Create a figure containing two side-by-side axes for displaying the original image and binary mask.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Display the original, unprocessed image on the first axis.
    ax1.imshow(original_image)
    # Add a title identifying the original image.
    ax1.set_title("Original Image")
    # Hide the axis markings around the original image.
    ax1.axis("off")
    # Display the binary mask as a grayscale image on the second axis.
    ax2.imshow(mask_img, cmap='gray')
    # Add a title identifying the binary mask and the thresholding method used.
    ax2.set_title(f"Binary Mask ({method})")
    # Hide the axis markings around the binary mask.
    ax2.axis("off")
    # Automatically adjust the spacing between the two plots to prevent overlap.
    plt.tight_layout()
    # Display the figure containing the original image and generated binary mask.
    plt.show()
    
    # Return the boolean binary mask for use in subsequent image-processing operations.
    return mask

def display_mask(mask,xsize = 20,ysize=20):

    '''Display a binary mask as a grayscale image using a Matplotlib figure.
        Input:
            mask - NumPy boolean array containing the binary mask, where True values represent the selected or masked regions and False values represent the background.
            xsize - Numeric value specifying the width of the displayed figure in inches.
            ysize - Numeric value specifying the height of the displayed figure in inches.
        Output:
            None - The function does not return a value. It displays the binary mask in a Matplotlib figure.
    '''
    # Convert the boolean mask into an 8-bit grayscale image where False = 0 (black) and True = 255 (white).
    mask_img = (mask * 255).astype(np.uint8)

    # Create a Matplotlib figure containing a single set of axes with the specified dimensions.
    fig, ax = plt.subplots(1, 1, figsize=(xsize, ysize))
    # Display the binary mask as a grayscale image.
    ax.imshow(mask_img, cmap='gray')
    # Add a title to the displayed mask.
    ax.set_title(f"Binary Mask")
    # Hide the axis markings around the displayed image.
    ax.axis("off")
    # Automatically adjust the spacing around the plot to prevent unnecessary margins or overlap.
    plt.tight_layout()
    # Display the figure containing the binary mask.
    plt.show()

def first_nonzero_average(mask, col_range, search_rows):

    '''Calculate the average row index of the first nonzero pixel across a specified range of columns.
        Input:
            mask - NumPy array containing a binary mask, where nonzero values represent the regions of interest.
            col_range - Iterable containing the column indices to inspect within the mask.
            search_rows - Integer specifying the maximum row index to search for the first nonzero pixel in each column.
        Output:
            Float - The average row index of the first nonzero pixel found across all columns containing a qualifying nonzero pixel. Returns 0 if no qualifying nonzero pixels are found.
    '''
    # Initialize the running total of the first nonzero row indices.
    total = 0
    # Initialize the number of columns containing a qualifying nonzero pixel.
    count = 0
    # Iterate through each specified column index.
    for i in col_range:
        # Extract all pixel values from the current column of the binary mask.
        col = mask[:, i]
        # Find the row indices where the current column contains nonzero values.
        nz = np.nonzero(col)[0]
        # Check that at least one nonzero pixel exists and that its first occurrence is within the specified search range.
        if nz.size > 0 and nz[0] < search_rows:
            # Add the row index of the first nonzero pixel to the running total.
            total += nz[0]
            # Increment the count of columns containing a qualifying nonzero pixel.
            count += 1
    # Calculate and return the average first-nonzero row index, or return 0 if no qualifying pixels were found.
    return total / count if count > 0 else 0

def rotate_mask_until_balanced(mask, angle_step=0.2, tolerance=0.005, min_step=0.01, max_angle=5, search_rows = 50, search_columns = 3000):

    '''Rotate a binary mask incrementally until the average first nonzero pixel locations on the left and right sides are approximately balanced.
        Input:
            mask - 2D NumPy array containing a binary mask, where nonzero values represent the regions of interest.
            angle_step - Float specifying the initial rotation increment in degrees. Positive values rotate the mask counterclockwise.
            tolerance - Float specifying the maximum allowed difference between the average first nonzero row locations on the two sides of the mask.
            min_step - Float specifying the minimum allowed rotation step size. This parameter is currently not used within the function.
            max_angle - Float specifying the maximum absolute rotation angle in degrees allowed before the function stops searching.
            search_rows - Integer specifying the maximum row index within which the function searches for the first nonzero pixel.
            search_columns - Integer specifying the number of columns to search on each side of the mask.
        Output:
            rot_mask - 2D NumPy array containing the rotated binary mask at the final calculated angle.
            angle - Float containing the final rotation angle in degrees applied to the mask.
    '''
    # Initialize the rotation angle at zero degrees.
    angle = 0.0
    # Initialize the previous difference between the left and right averages as None because no previous iteration exists yet.
    prev_diff = None
    # Determine the total number of columns in the input mask.
    ncols = mask.shape[1]

    # Iteratively rotate the mask and adjust the angle until the two sides are sufficiently balanced or a stopping condition is reached.
    for _ in range(100):  # max iterations to prevent infinite loop
        # Rotate the mask counterclockwise by the current angle while preserving its original dimensions.
        # order=0 uses nearest-neighbor interpolation, which preserves the binary nature of the mask.
        # Pixels rotated outside the original image are filled with zero.
        rot_mask = rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0)
        # Calculate the average row index of the first nonzero pixel within the specified number of columns on the left side.
        av1 = first_nonzero_average(rot_mask, range(0, search_columns),search_rows)
        # Calculate the average row index of the first nonzero pixel within the specified number of columns on the right side.
        av2 = first_nonzero_average(rot_mask, range(ncols-search_columns, ncols), search_rows)
        # Calculate the difference between the left-side and right-side average first-nonzero row positions.
        diff = av1 - av2

        # Print the current angle, the two average positions, and their difference for monitoring the balancing process.
        print(f"Angle: {angle:.4f}°, av1: {av1:.2f}, av2: {av2:.2f}, diff: {diff:.4f}")

        # Stop rotating when the difference between the two sides is smaller than the specified tolerance.
        if abs(diff) < tolerance:
            break

        # Check whether the current difference has crossed zero relative to the previous iteration.
        if prev_diff is not None and np.sign(diff) != np.sign(prev_diff):
            # Reverse the direction of rotation and reduce the step size by half when the balance point is crossed.
            angle_step = -angle_step / 2.0  # flip direction and halve the step

        # Update the rotation angle using the current rotation step.
        angle += angle_step
        # Stop the search if the absolute rotation angle exceeds the specified maximum.
        if abs(angle) > max_angle:
            print("Max angle reached without full balancing.")
            break

        # Store the current difference so it can be compared with the next iteration.
        prev_diff = diff

    # Return the final rotated mask and the rotation angle used to produce it.
    return rot_mask, angle

def show_reference(mask, ref_bbox):

    '''Display a cropped reference region from a binary mask after removing small objects.
        Input:
            mask - 2D NumPy array containing a binary mask, where nonzero values represent the regions of interest.
            ref_bbox - Iterable containing four integer coordinates defining the reference bounding box in the order (minr, minc, maxr, maxc), where minr and minc are the starting row and column and maxr and maxc are the ending row and column.
        Output:
            None - The function does not return a value. It displays the cropped and processed reference region as a grayscale image.
    '''
    # Unpack the bounding box coordinates into minimum and maximum row and column indices.
    minr, minc, maxr, maxc = ref_bbox
    # Crop the mask to the region defined by the reference bounding box and convert the result to a boolean array.
    ref_crop = mask[minr:maxr, minc:maxc].astype(bool)
    # Remove connected objects containing fewer than 100 pixels from the cropped binary mask.
    ref_crop = morphology.remove_small_objects(ref_crop, min_size= 100)
    # Pad the cropped mask with zeros. A pad width of 0 leaves the dimensions unchanged.
    ref_crop = np.pad(ref_crop, pad_width=0, mode='constant')
    # Convert the boolean mask into an 8-bit grayscale image where False = 0 (black) and True = 255 (white).
    mask_img = (ref_crop * 255).astype(np.uint8)

    # Create a Matplotlib figure containing a single set of axes.
    fig, ax = plt.subplots(1, 1)#, figsize=(xsize, ysize))
    # Display the processed reference crop as a grayscale image.
    ax.imshow(mask_img, cmap='gray')
    # Add a title to the displayed reference mask.
    ax.set_title(f"Binary Mask")
    # Hide the axis markings around the displayed image.
    ax.axis('off')
    # Automatically adjust the spacing around the plot to prevent unnecessary margins or overlap.
    plt.tight_layout()
    # Display the figure containing the processed reference mask.
    plt.show()

def fill_row_gaps(mask):

    '''Fill gaps of zero values between the first and last nonzero values in each row of a binary mask.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
        Output:
            filled_mask - 2D NumPy array with the same shape and data type as mask, where all values between the first and last nonzero values in each row have been set to 1.
    '''
    # Create a copy of the input mask so that the original array is not modified.
    filled_mask = mask.copy()
    # Iterate through every row of the mask.
    for i in range(filled_mask.shape[0]):
        # Extract the current row from the mask.
        row = filled_mask[i]
        # Find the column indices where the current row contains nonzero values.
        ones_indices = np.flatnonzero(row)
        # Only fill gaps when the row contains at least two nonzero values.
        if len(ones_indices) >= 2:
            # Determine the column indices of the first and last nonzero values in the row.
            start, end = ones_indices[0], ones_indices[-1]
            # Fill all values between the first and last nonzero values with 1.
            row[start:end+1] = 1  # Fill the entire range between first and last 1
            # Store the modified row back into the copied mask.
            filled_mask[i] = row
    # Return the modified mask containing the filled row gaps.
    return filled_mask

def fill_row_gaps2(mask, max_gap=10):

    '''Fill horizontal gaps in each row of a binary mask when the gaps are surrounded by foreground pixels and do not exceed a specified width.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
            max_gap - Integer specifying the maximum number of consecutive zero-valued pixels that can be filled within a row.
        Output:
            filled_mask - 2D NumPy array with the same shape and data type as mask, containing the binary mask after qualifying horizontal gaps have been filled with 1s.
    '''
    # Create a copy of the input mask so that the original array is not modified directly.
    filled_mask = mask.copy()
    # Store the number of rows and columns in the binary mask.
    n_rows, n_cols = filled_mask.shape

    # Iterate through each row of the binary mask.
    for row in range(n_rows):
        # Extract the current row from the mask.
        row_data = filled_mask[row, :]
        # Find the column indices of all nonzero pixels in the current row.
        ones = np.flatnonzero(row_data)

        # Skip the current row if it contains no nonzero pixels.
        if len(ones) == 0:
            continue

        # Fill from left edge
        # If the first nonzero pixel is within max_gap pixels of the left edge, fill the gap between the edge and that pixel.
        if ones[0] <= max_gap:
            row_data[:ones[0]] = 1

        # Fill from right edge
        # If the last nonzero pixel is within max_gap pixels of the right edge, fill the gap between that pixel and the edge.
        if (n_cols - 1 - ones[-1]) <= max_gap:
            row_data[ones[-1] + 1:] = 1

        # Fill internal gaps
        # Examine each pair of consecutive nonzero pixels to identify gaps between them.
        for i in range(len(ones) - 1):
            # Store the column index of the first nonzero pixel surrounding the current gap.
            start = ones[i]
            # Store the column index of the second nonzero pixel surrounding the current gap.
            end = ones[i + 1]
            # Calculate the number of zero-valued pixels between the two nonzero pixels.
            gap_size = end - start - 1
            # Fill the internal gap if it contains at least one pixel and does not exceed max_gap.
            if 0 < gap_size <= max_gap:
                row_data[start + 1:end] = 1

        # Store the modified row back into the output mask.
        filled_mask[row, :] = row_data

    # Return the mask with all qualifying horizontal gaps filled.
    return filled_mask

def fill_column_gaps(mask, max_gap=10):

    '''Fill vertical gaps of zero values in each column of a binary mask when the gaps are smaller than or equal to a specified maximum size.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
            max_gap - Integer specifying the maximum number of consecutive zero-valued pixels that can be filled in a column.
        Output:
            filled_mask - 2D NumPy array with the same shape and data type as mask, where qualifying gaps between nonzero regions and gaps adjacent to the top or bottom edges have been filled with 1s.
    '''
    # Create a copy of the input mask so that the original array is not modified.
    filled_mask = mask.copy()
    # Determine the number of rows and columns in the binary mask.
    n_rows, n_cols = filled_mask.shape

    # Iterate through every column of the mask.
    for col in range(n_cols):
        # Extract the current column from the mask.
        col_data = filled_mask[:, col]
        # Find the row indices where the current column contains nonzero values.
        ones = np.flatnonzero(col_data)

        # Skip the current column if it contains no nonzero values.
        if len(ones) == 0:
            continue

        # Fill from top edge
        # Fill the pixels between the top edge and the first nonzero pixel if the gap is within the maximum allowed size.
        if ones[0] <= max_gap:
            col_data[:ones[0]] = 1

        # Fill from bottom edge
        # Fill the pixels between the last nonzero pixel and the bottom edge if the gap is within the maximum allowed size.
        if (n_rows - 1 - ones[-1]) <= max_gap:
            col_data[ones[-1] + 1:] = 1

        # Fill internal gaps
        # Iterate through each pair of consecutive nonzero pixels to identify gaps between them.
        for i in range(len(ones) - 1):
            # Get the row indices of the two consecutive nonzero pixels surrounding the potential gap.
            start = ones[i]
            end = ones[i + 1]
            # Calculate the number of zero-valued pixels between the two nonzero pixels.
            gap_size = end - start - 1
            # Fill the gap if it contains at least one pixel and does not exceed the maximum allowed gap size.
            if 0 < gap_size <= max_gap:
                col_data[start + 1:end] = 1

        # Store the modified column back into the copied mask.
        filled_mask[:, col] = col_data

    # Return the modified mask containing the filled vertical gaps.
    return filled_mask

def fill_reference(mask,ref_bbox, min_size = 100, pad = 10, max_gap = 10, show = False):

    '''Crop and process a reference region from a binary mask by removing small objects, adding padding, and filling horizontal and vertical gaps.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
            ref_bbox - Iterable containing four integer coordinates defining the reference bounding box in the order (minr, minc, maxr, maxc), where minr and minc are the starting row and column and maxr and maxc are the ending row and column.
            min_size - Integer specifying the minimum number of pixels required for a connected object to remain in the mask after small-object removal.
            pad - Integer specifying the number of pixels of constant-value padding added to all sides of the cropped reference mask.
            max_gap - Integer specifying the maximum number of consecutive zero-valued pixels that can be filled when filling vertical gaps.
            show - Boolean specifying whether the intermediate and final reference masks should be displayed during processing.
        Output:
            ref_crop - 2D NumPy boolean array containing the processed reference region after cropping, removal of small objects, padding, and filling of horizontal and vertical gaps.
    '''
    # Unpack the bounding box coordinates into minimum and maximum row and column indices.
    minr, minc, maxr, maxc = ref_bbox
    # Crop the mask to the region defined by the reference bounding box and convert the result to a boolean array.
    ref_crop = mask[minr:maxr, minc:maxc].astype(bool)
    # Display the initial cropped reference mask if visualization has been enabled.
    if show == True:
        display_mask(ref_crop,5,5)

    # Remove connected objects containing fewer pixels than the specified minimum size.
    ref_crop = morphology.remove_small_objects(ref_crop, min_size=min_size)
    # Add constant-value padding around all sides of the reference mask.
    ref_crop = np.pad(ref_crop, pad_width = pad, mode='constant')
    # Fill gaps between nonzero regions across each row of the reference mask.
    ref_crop = fill_row_gaps(ref_crop)
    # Fill qualifying gaps between nonzero regions along each column of the reference mask.
    ref_crop = fill_column_gaps(ref_crop, max_gap = max_gap)
    # Display the processed reference mask if visualization has been enabled.
    if show == True:
        display_mask(ref_crop, 5, 5)

    # Return the processed reference mask.
    return ref_crop

def fill_reference3(mask,ref_bbox, min_size = 20, opening_disk_size = 2, pad = 100, closing_disk_size1 = 7,
                    larger_object_size = 50, max_row_gap = 80, max_column_gap = 30, opening_disk_size2 = 3,
                    max_row_gap2 = 5, max_column_gap2 = 4, show = False):
    '''
    Process a cropped reference region from a binary mask using morphological operations and gap filling.
        Input:
            mask - 2D NumPy array containing the original binary mask, where nonzero or True values represent
            regions of interest and zero or False values represent the background.
            ref_bbox - Iterable containing four integer coordinates defining the reference bounding box in the
            order (minr, minc, maxr, maxc), where minr and minc are the starting row and column and maxr and
            maxc are the ending row and column.
            min_size - Integer specifying the minimum size of objects retained during the initial removal of
            small objects.
            opening_disk_size - Integer specifying the radius of the disk-shaped structuring element used for
            the first morphological opening operation.
            pad - Integer specifying the number of zero-valued pixels added around the reference crop before
            subsequent morphological processing.
            closing_disk_size1 - Integer specifying the radius of the disk-shaped structuring element used for
            the first morphological closing operation.
            larger_object_size - Integer specifying the minimum size of objects retained after morphological
            closing.
            max_row_gap - Integer specifying the maximum horizontal gap size that will be filled within each row.
            max_column_gap - Integer specifying the maximum vertical gap size that will be filled within each column.
            opening_disk_size2 - Integer specifying the radius of the disk-shaped structuring element used for
            the second morphological opening operation.
            max_row_gap2 - Integer specifying the maximum horizontal gap size filled during the final row-gap
            filling operation.
            max_column_gap2 - Integer specifying the maximum vertical gap size filled during the final
            column-gap filling operation.
            show - Boolean specifying whether intermediate and final processing results should be displayed.
        Output:
            ref_crop - 2D NumPy boolean array containing the processed reference region after removing small
            objects, applying morphological opening and closing, filling row and column gaps, and removing
            the temporary padding.
    '''
    # Unpack the reference bounding box into minimum and maximum row and column coordinates.
    minr, minc, maxr, maxc = ref_bbox
    # Crop the original mask to the region defined by the reference bounding box and convert it to a boolean array.
    ref_crop = mask[minr:maxr, minc:maxc].astype(bool)
    # Create a copy of the original cropped reference mask for use in optional visualization.
    cropmask = ref_crop.copy()
    # Display the initial cropped reference mask if visualization is enabled.
    if show == True:
        display_mask(ref_crop,5,5)

    # Remove connected objects smaller than the specified minimum size from the reference mask.
    ref_crop = morphology.remove_small_objects(ref_crop, min_size=min_size)
    # print('remove_small_objects')
    # display_mask(ref_crop,6,6)
    # Apply morphological opening using a disk-shaped structuring element to remove small features and smooth boundaries.
    ref_crop = binary_opening(ref_crop, structure = disk(opening_disk_size))
    # print('binary opening')
    # display_mask(ref_crop,6,6)
    # Add a constant zero-valued border around the reference mask to provide space for subsequent morphological operations.
    ref_crop = np.pad(ref_crop, pad_width = pad, mode='constant')
    # print('add pad')
    # display_mask(ref_crop,6,6)
    # Apply morphological closing using a disk-shaped structuring element to close small gaps and connect nearby regions.
    ref_crop = binary_closing(ref_crop, structure = disk(closing_disk_size1))
    # print('binary closing')
    # display_mask(ref_crop,6,6)
    # Remove connected objects smaller than the specified larger-object threshold.
    ref_crop = morphology.remove_small_objects(ref_crop, min_size=larger_object_size)
    # print('remove larger objects')
    # display_mask(ref_crop,6,6)
    # Fill horizontal gaps within rows that are no larger than the specified maximum gap size.
    ref_crop = fill_row_gaps2(ref_crop, max_gap = max_row_gap)
    # print('remove spacing')
    # display_mask(ref_crop,6,6)
    # Fill vertical gaps within columns that are no larger than the specified maximum gap size.
    ref_crop = fill_column_gaps(ref_crop, max_gap = max_column_gap)
    # print('remove spacing')
    # display_mask(ref_crop,6,6)
    # Apply a second morphological opening to further remove small features and smooth the processed mask.
    ref_crop = binary_opening(ref_crop, structure = disk(opening_disk_size2))
    # print('binary opening')
    # display_mask(ref_crop,6,6)
    # Remove the temporary padding from all four sides of the reference mask.
    ref_crop = ref_crop[pad:-pad, pad:-pad]
    # print('remove pad')
    # display_mask(ref_crop,6,6)
    # ref_crop = np.pad(ref_crop, pad_width = pad, mode='constant')
    # print('add pad')
    # display_mask(ref_crop,6,6)
    #ref_crop = fill_small_holes(ref_crop, max_hole_size = hole_size)
    # print('remove holes')
    # display_mask(ref_crop,6,6)
    # ref_crop = binary_closing(ref_crop, structure = disk(closing_disk_size2))
    # smoothed = gaussian_filter(ref_crop.astype(float), sigma=sigma)
    # # Re-threshold after smoothing
    # smoothed_binary = smoothed > smooth_threshold
    # # Apply morphological closing to round inward corners
    # smoothed_binary = morphology.binary_closing(smoothed_binary, morphology.disk(smoothing_disk_size))
    # Fill smaller horizontal gaps within rows during the final refinement of the reference mask.
    ref_crop = fill_row_gaps2(ref_crop, max_gap = max_row_gap2)
    # Fill smaller vertical gaps within columns during the final refinement of the reference mask.
    ref_crop = fill_column_gaps(ref_crop, max_gap = max_column_gap2)

    
    # Display an overlay comparing the processed reference mask with a selected region of the original mask if visualization is enabled.
    if show == True:
        # Create a three-channel floating-point image by duplicating the processed binary mask across RGB channels.
        overlay_img = np.stack([ref_crop]*3, axis=-1).astype(float)
        # Extract a fixed region from the original mask for comparison with the processed reference mask.
        cropmask = mask[52:351,55:355]

        #Overlay in red (R=1, G=0, B=0)
        # Set the red channel to maximum intensity wherever the comparison mask contains nonzero values.
        overlay_img[cropmask.astype(bool), 0] = 1.0  # Red
        # Set the green channel to zero wherever the comparison mask contains nonzero values.
        overlay_img[cropmask.astype(bool), 1] = 0.0  # Green
        # Set the blue channel to zero wherever the comparison mask contains nonzero values.
        overlay_img[cropmask.astype(bool), 2] = 0.0  # Blue

        # Display result
        # Create a figure for displaying the processed reference mask and overlay.
        plt.figure(figsize=(6, 6))
        # Display the overlay image.
        plt.imshow(overlay_img)
        # Add a title describing the overlay.
        plt.title("Overlay Mask in Red")
        # Hide the axis markings around the image.
        plt.axis('off')
        # Display the figure.
        plt.show()

    # Return the fully processed reference mask.
    return ref_crop

def last_data_column_in_rows(arr, row_range):

    '''Find the last column containing any nonzero value within a specified set or range of rows.
        Input:
            arr - 2D NumPy array containing the data to be searched. Nonzero values are treated as data, while zero values are treated as empty.
            row_range - Iterable containing row indices to search, or a tuple containing the start and stop indices defining a range of rows.
        Output:
            Integer - The index of the last column containing at least one nonzero value within the specified rows. Returns -1 if all specified rows contain only zero values.
    '''
    # Check whether row_range was provided as a tuple defining the start and stop of a row range.
    if isinstance(row_range, tuple):  # convert (start, stop) to a proper range
        # Convert the tuple into a range object containing all rows from the start index up to, but not including, the stop index.
        row_range = range(row_range[0], row_range[1])
        
    # Extract the specified rows from the array and retain all columns.
    subarray = arr[np.array(list(row_range)), :]  # valid rows subset
    # Determine which columns contain at least one nonzero value among the selected rows.
    col_sums = np.any(subarray != 0, axis=0)
    # Find the column indices where at least one nonzero value was detected.
    nonzero_cols = np.nonzero(col_sums)[0]
    # Return the index of the last column containing data, or -1 if no nonzero columns were found.
    return nonzero_cols[-1] if nonzero_cols.size > 0 else -1

def last_data_row_in_columns(arr, col_range):

    '''Find the last row containing any nonzero value within a specified set of columns.
        Input:
            arr - 2D NumPy array containing the data to be searched. Nonzero values are treated as data, while zero values are treated as empty.
            col_range - Iterable containing the column indices to search, such as range(50, 150).
        Output:
            Integer - The index of the last row containing at least one nonzero value within the specified columns. Returns -1 if all specified columns contain only zero values.
    '''
    # Extract the specified columns from the array while retaining all rows.
    subarray = arr[:, np.array(list(col_range))]  # select specified columns
    # Determine which rows contain at least one nonzero value among the selected columns.
    row_sums = np.any(subarray != 0, axis=1)      # True if any col in that row is nonzero
    # Find the row indices where at least one nonzero value was detected.
    nonzero_rows = np.nonzero(row_sums)[0]
    # Return the index of the last row containing data, or -1 if no nonzero rows were found.
    return nonzero_rows[-1] if nonzero_rows.size > 0 else -1

def check_last_row_and_column(mask, min_size = 100, last_row = 200, last_column = 200, plot = False):

    '''Determine the last nonzero row and column of a binary mask after removing small objects, with optional plots for visual verification.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
            min_size - Integer specifying the minimum number of pixels required for a connected object to remain in the mask after small-object removal.
            last_row - Integer specifying the number of rows from the top of the mask to search when determining the last data-containing column.
            last_column - Integer specifying the number of columns from the left of the mask to search when determining the last data-containing row.
            plot - Boolean specifying whether plots should be displayed to visually verify the detected last row and column.
        Output:
            last_data - Integer containing the index of the last column containing nonzero data within the specified row range. Returns -1 if no data is found.
            last_y - Integer containing the index of the last row containing nonzero data within the specified column range. Returns -1 if no data is found.
    '''
    # Remove objects smaller than the specified minimum size and find the last data-containing column within the specified top rows.
    last_data = last_data_column_in_rows(morphology.remove_small_objects(mask, min_size=min_size), (0, last_row))
    # Remove objects smaller than the specified minimum size and find the last data-containing row within the specified left columns.
    last_y = last_data_row_in_columns(morphology.remove_small_objects(mask, min_size=min_size), (0, last_column))

    # Remove small objects, crop the mask to the specified top rows and rightmost columns, and convert it to an 8-bit grayscale image.
    mask_img = (morphology.remove_small_objects(mask, min_size=min_size)[0:last_row,-last_row:] * 255).astype(np.uint8)
    # Calculate the horizontal offset between the original mask's right edge and the detected last data-containing column.
    offset = mask.shape[1] - last_data
    # Convert the detected column index from the coordinates of the original mask to the coordinates of the cropped image.
    adjusted_last_data = mask_img.shape[1] - offset
    
    # Display the detected last column and a visual reference line if plotting is enabled.
    if plot == True:
        # Print the dimensions of the original mask for reference.
        print('Mask shape: ', mask.shape)
        # Print a label identifying the column check.
        print('Check last column')
        # Create a Matplotlib figure containing a single set of axes.
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # Display the cropped mask as a grayscale image.
        ax.imshow(mask_img, cmap='gray')
        # Draw a vertical dashed line at the detected last data-containing column.
        ax.axvline(x=adjusted_last_data, color='red', linestyle='--', linewidth=2)
        # Add a title to the displayed mask.
        ax.set_title(f"Binary Mask")
        # Hide the axis markings around the displayed image.
        ax.axis("off")
        # Automatically adjust the spacing around the plot to prevent unnecessary margins or overlap.
        plt.tight_layout()
        # Display the figure containing the detected last column.
        plt.show()
        # Print the detected column index.
        print('Last column: ',last_data)
        # Print a blank line to separate the two checks in the output.
        print('')

    # Remove small objects, crop the mask to the specified bottom rows and leftmost columns, and convert it to an 8-bit grayscale image.
    mask_img = (morphology.remove_small_objects(mask, min_size=min_size)[-last_column:,0:last_column] * 255).astype(np.uint8)
    # Calculate the vertical offset between the original mask's bottom edge and the detected last data-containing row.
    offset = mask.shape[0] - last_y
    # Convert the detected row index from the coordinates of the original mask to the coordinates of the cropped image.
    adjusted_last_y = mask_img.shape[0] - offset

    # Display the detected last row and a visual reference line if plotting is enabled.
    if plot == True:
        # Print a label identifying the row check.
        print('Check last column')
        # Create a Matplotlib figure containing a single set of axes.
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # Display the cropped mask as a grayscale image.
        ax.imshow(mask_img, cmap='gray')
        # Draw a horizontal dashed line at the detected last data-containing row.
        ax.axhline(y=adjusted_last_y, color='red', linestyle='--', linewidth=2)

        # Add a title to the displayed mask.
        ax.set_title(f"Binary Mask")
        # Hide the axis markings around the displayed image.
        ax.axis("off")
        # Automatically adjust the spacing around the plot to prevent unnecessary margins or overlap.
        plt.tight_layout()
        # Display the figure containing the detected last row.
        plt.show()
        # Print the detected row index.
        print('Last row: ',last_y)

    # Return the indices of the last data-containing column and row in the original mask.
    return last_data, last_y

def erode_mask(mask, n=1):

    '''Erode the boundaries of connected regions in a binary mask by a specified number of pixels using morphological erosion.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
            n - Integer specifying the number of morphological erosion iterations to perform, effectively controlling how many layers of pixels are removed from the boundaries of connected regions.
        Output:
            eroded_mask - 2D NumPy uint8 array containing the eroded binary mask, where background pixels have a value of 0 and remaining regions have a value of 1.
    '''
    # Create a 3x3 boolean structuring element that considers all eight neighboring pixels around each pixel.
    structure = np.ones((3, 3), dtype=bool)  # 8-connected structuring element
    # Apply binary erosion to the mask for the specified number of iterations using the 8-connected structuring element.
    eroded_mask = binary_erosion(mask, structure=structure, iterations=n)
    # Convert the resulting boolean mask to an 8-bit unsigned integer array containing 0s and 1s.
    return eroded_mask.astype(np.uint8)

def extract_particles_and_debris(mask, ref_bbox, ref_crop, min_size = 100, pad=10, max_gap = 10, dx_offset = 0, dy_offset = 0,
                                 stagger_x = False, stagger_y = False, stagger_y_frequency = 2, stagger_x_frequency = 2, erode_pixels = 1,
                                 checking = False, check_last_row =  200, check_last_column = 200, num_rows = 25, num_cols = 25):
    '''
    Identify particles matching a manually selected reference mask and separate the identified particles from
    remaining debris using a regular grid of reference locations across the input mask.
        Input:
            mask - 2D NumPy array containing a binary mask of particles and debris, where nonzero or True values
            represent particles or regions of interest and zero or False values represent the background.
            ref_bbox - Tuple containing the bounding box of the reference particle in the format
            (min_row, min_col, max_row, max_col), where the first two values define the row range and the second
            two define the column range.
            ref_crop - 2D NumPy array containing the processed binary mask of the reference particle used to
            identify matching particle regions throughout the input mask.
            min_size - Integer specifying the minimum object size used when determining the extent of valid data
            in the input mask.
            pad - Integer specifying the number of zero-valued pixels added around the reference and each
            extracted particle crop.
            max_gap - Integer specifying the maximum gap size used by the reference-processing workflow. This
            parameter is retained for compatibility but is not directly used within this function.
            dx_offset - Integer offset added to the calculated horizontal spacing between particle locations.
            dy_offset - Integer offset added to the calculated vertical spacing between particle locations.
            stagger_x - Boolean specifying whether alternating particle locations should receive an additional
            horizontal spacing adjustment.
            stagger_y - Boolean specifying whether alternating particle rows should receive an additional
            vertical spacing adjustment.
            stagger_y_frequency - Integer specifying the row frequency at which the additional vertical staggering
            adjustment is applied when stagger_y is enabled.
            stagger_x_frequency - Integer specifying the column frequency at which the additional horizontal
            staggering adjustment is applied when stagger_x is enabled.
            erode_pixels - Integer specifying the number of morphological erosion iterations applied to the
            reference mask before matching. A value of zero disables erosion.
            checking - Boolean specifying whether individual crops and their matched regions should be displayed
            during processing for debugging and verification.
            check_last_row - Integer specifying the number of rows examined near the top of the mask when
            determining the last populated column and row.
            check_last_column - Integer specifying the number of columns examined near the left side of the mask
            when determining the last populated column and row.
            num_rows - Integer specifying the number of particle rows to examine across the mask.
            num_cols - Integer specifying the number of particle columns to examine within each row.
        Output:
            particle_mask - 2D NumPy boolean array with the same spatial dimensions as mask, containing the
            regions identified as particles matching the reference mask.
            debris_mask - 2D NumPy boolean array with the same spatial dimensions as mask, containing regions
            of the original mask that are not included within the identified particle regions.
    '''

    # Unpack the reference bounding box into its minimum and maximum row and column coordinates.
    minr, minc, maxr, maxc = ref_bbox
    # Add zero-valued padding around the supplied reference mask to provide additional space for matching.
    ref_crop = np.pad(ref_crop, pad_width = pad, mode = 'constant')
    # Indicate that the reference mask has been prepared and display it for inspection.
    print('Filled Reference')
    display_mask(ref_crop, 5, 5)

    # Apply morphological erosion to the reference mask when a nonzero number of erosion pixels is specified.
    if erode_pixels != 0:
        # Indicate that erosion is being applied to the reference mask.
        print('Eroded Reference')
        # Erode the reference mask by the specified number of pixels.
        ref_crop = erode_mask(ref_crop, erode_pixels)
        # Display the eroded reference mask for inspection.
        display_mask(ref_crop, 5, 5)

    # Calculate the row and column coordinates of the center of the reference particle.
    ref_center = ((minr + maxr) // 2, (minc + maxc) // 2)
    # Calculate the height of the original reference particle bounding box.
    ref_height = maxr - minr
    # Calculate the width of the original reference particle bounding box.
    ref_width = maxc - minc

    # Determine the last populated column and row of the input mask using the specified search regions.
    last_x, last_y = check_last_row_and_column(mask, min_size = min_size, last_row = check_last_row, last_column = check_last_column, plot = False)
    # Calculate the horizontal distance between the last populated column and the right edge of the mask.
    last_data_distance = mask.shape[1]-last_x
    # Calculate the vertical distance between the last populated row and the bottom edge of the mask.
    last_y_distance = mask.shape[0]-last_y

    # Set dx and dy so that the specified number of particle locations span the usable image dimensions.
    dx = (mask.shape[1]-ref_center[1]-(ref_center[1]+last_data_distance)) // 24
    # Calculate the vertical spacing between particle rows across the usable image dimensions.
    dy = (mask.shape[0]-ref_center[0]-(ref_center[0]+last_y_distance)) // 24
    # Apply the user-specified horizontal spacing offset.
    dx += dx_offset
    # Apply the user-specified vertical spacing offset.
    dy += dy_offset

    # Store the initial horizontal spacing so that it can be reused for each row.
    dx_start = dx
    # Store the initial vertical spacing so that it can be reused for each column.
    dy_start = dy

    # Create an empty boolean mask with the same dimensions as the input mask for storing identified particles.
    particle_mask = np.zeros_like(mask, dtype=bool)

    # Initialize the row center using the vertical coordinate of the reference particle.
    center_r0 = ref_center[0]
    # Iterate through the specified number of particle rows.
    for i in range(num_rows):
        # Apply an additional vertical spacing adjustment at the specified row frequency when staggering is enabled.
        if i % stagger_y_frequency == 0 and stagger_y:
                dy = dy_start + 1
        # Otherwise, use the original vertical spacing.
        else:
            dy = dy_start
        # Reset the horizontal center to the reference particle's column center at the beginning of each row.
        center_c = ref_center[1]
        # Iterate through the specified number of particle columns within the current row.
        for j in range(num_cols):
            # Apply an additional horizontal spacing adjustment at the specified column frequency when staggering is enabled.
            if j % stagger_x_frequency == 0 and stagger_x:
                dx = dx_start + 1
            # Otherwise, use the original horizontal spacing.
            else:
                dx = dx_start
            # Calculate the starting row coordinate of the current particle crop.
            r0 = center_r0 - ref_height // 2
            # Calculate the ending row coordinate of the current particle crop.
            r1 = r0 + ref_height
            # Calculate the starting column coordinate of the current particle crop.
            c0 = center_c - ref_width // 2
            # Calculate the ending column coordinate of the current particle crop.
            c1 = c0 + ref_width
            # Recalculate the horizontal center using the refined crop boundaries.
            center_c = (c0 + c1) // 2  # update x-center using refined result

            # Get crop and pad if necessary
            # Extract the current particle-sized region from the original binary mask.
            crop = mask[r0:r1, c0:c1]
            # Add zero-valued padding around the extracted crop and convert it to a boolean array.
            crop = np.pad(crop, pad_width=pad, mode='constant').astype(bool)

            # Keep only pixels that are present in both the current crop and the reference mask.
            masked_crop = crop & ref_crop
            # Insert the matched particle region into the corresponding location of the global particle mask.
            particle_mask[(r0-pad):(r1+pad), (c0-pad):(c1+pad)] = masked_crop

            # Display intermediate crops and matching results when checking is enabled.
            if checking == True:
                # Print the current row index for tracking the processing location.
                print('row:', i)
                # Print the current column index for tracking the processing location.
                print('column:',j)
                # Display the current particle crop before reference matching.
                display_mask(crop,5,5)
                # Display the portion of the crop retained after applying the reference mask.
                display_mask(masked_crop,5,5)

            # Move the horizontal center to the next expected particle location.
            center_c += dx
        # Move the vertical center to the next expected particle row after completing the current row.
        center_r0 += dy  # move y-center after full row

    # Dilate the identified particle mask to provide a buffer around particle regions when separating debris.
    dilated_particle_mask = binary_dilation(particle_mask, footprint=disk(8))
    # Define debris as regions present in the original mask that do not overlap the dilated particle mask.
    debris_mask = mask & ~dilated_particle_mask
    # Return the identified particle mask and the remaining debris mask.
    return particle_mask, debris_mask

def save_mask(mask, image_path, output_extension):

    '''Save a binary mask as an image file using the input image path to construct the output filename.
        Input:
            mask - 2D NumPy array containing a binary mask, with values represented as either 0 and 1 or False and True.
            image_path - String containing the file path to the original input image. The filename is used to construct the output path for the saved mask.
            output_extension - String specifying the file extension to use for the saved mask image, such as '.png' or '.tif'.
        Output:
            None - The function does not return a value. It saves the binary mask as an image file and prints the resulting file path.
    '''
    # Convert the binary mask into an 8-bit grayscale image where False/0 = 0 (black) and True/1 = 255 (white).
    mask_img = (mask * 255).astype(np.uint8)
    # Construct the output file path by replacing the '.jpg' extension with '_cleaned_mask' followed by the specified output extension.
    save_path = image_path.replace(".jpg","_cleaned_mask"+output_extension)
    # Convert the NumPy image array to a PIL Image and save it to the constructed output path.
    Image.fromarray(mask_img).save(save_path)
    # Print the location where the processed mask was saved.
    print(f"✅ Mask saved to: {save_path}")

def show_original_image(wafer_path, x_size = 20, y_size = 20):

    '''Display an original RGB image using a Matplotlib figure.
        Input:
            wafer_path - String containing the file path to the input image.
            x_size - Numeric value specifying the width of the displayed figure in inches.
            y_size - Numeric value specifying the height of the displayed figure in inches.
        Output:
            None - The function does not return a value. It displays the original image in a Matplotlib figure.
    '''
    # Open the input image from the specified file path and convert it to RGB format.
    original_image = Image.open(wafer_path).convert('RGB')
    #crop_box = (2000, 1800, np.shape(original_image)[0], np.shape(original_image)[1])  # <-- modify these values as needed

    # Create a Matplotlib figure containing a single set of axes with the specified dimensions.
    fig, ax = plt.subplots(1, 1, figsize=(x_size, y_size))
    # Display the original RGB image.
    ax.imshow(original_image)
    # Add a title identifying the displayed image.
    ax.set_title("Original Image")
    # Hide the axis markings around the displayed image.
    ax.axis("off")
    # Automatically adjust the spacing around the plot to prevent unnecessary margins or overlap.
    plt.tight_layout()
    # Display the figure containing the original image.
    plt.show()

    return original_image

def cropped_image_to_mask(cropped_image, brightness=1.0, contrast=1.0, saturation=1.0,temperature=0, R_min=0,
                          G_min=0, B_min=30, V_min=0.1, method="otsu", adaptive_block_size=31, adaptive_offset=0.05):
    
    '''Convert a cropped RGB image into a binary mask using manual RGB/HSV thresholding, Otsu thresholding, adaptive thresholding, or simple grayscale thresholding.
        Input:
            cropped_image - PIL Image object containing the cropped RGB image to be processed.
            brightness - Float controlling the brightness adjustment applied to the image, where 1.0 leaves the brightness unchanged.
            contrast - Float controlling the contrast adjustment applied to the image, where 1.0 leaves the contrast unchanged.
            saturation - Float controlling the color saturation adjustment applied to the image, where 1.0 leaves the saturation unchanged.
            temperature - Numeric value controlling the red/blue temperature shift applied to the image. Positive values increase red and decrease blue.
            R_min - Numeric minimum threshold for the red channel when using the manual thresholding method.
            G_min - Numeric minimum threshold for the green channel when using the manual thresholding method.
            B_min - Numeric minimum threshold for the blue channel when using the manual thresholding method.
            V_min - Float minimum value/brightness threshold in the HSV color space when using the manual thresholding method.
            method - String specifying the thresholding method. Accepted values are 'manual', 'otsu', 'adaptive', or 'simple gray'.
            adaptive_block_size - Integer specifying the size of the local neighborhood used to calculate the mean for adaptive thresholding.
            adaptive_offset - Float added to the local mean when determining the threshold for adaptive thresholding.
        Output:
            mask - 2D NumPy boolean array with the same height and width as the input image. True values represent pixels that pass the selected thresholding criteria, while False values represent pixels that do not.
    '''
    # Adjust the brightness of the cropped image according to the specified brightness factor.
    image = ImageEnhance.Brightness(cropped_image).enhance(brightness)
    # Adjust the contrast of the brightness-adjusted image according to the specified contrast factor.
    image = ImageEnhance.Contrast(image).enhance(contrast)
    # Adjust the color saturation of the contrast-adjusted image according to the specified saturation factor.
    image = ImageEnhance.Color(image).enhance(saturation)

    # Apply temperature shift
    # Convert the PIL image into a NumPy array of 32-bit floating-point values for numerical manipulation.
    image_np = np.array(image).astype(np.float32)
    # Increase the red channel by the temperature value while restricting values to the valid RGB range of 0 to 255.
    image_np[:, :, 0] = np.clip(image_np[:, :, 0] + temperature, 0, 255)  # R
    # Decrease the blue channel by the temperature value while restricting values to the valid RGB range of 0 to 255.
    image_np[:, :, 2] = np.clip(image_np[:, :, 2] - temperature, 0, 255)  # B

    # Grayscale version for Otsu or adaptive methods
    # Convert the RGB image to grayscale using standard luminance weights for the red, green, and blue channels.
    # Divide by 255 to normalize the grayscale intensity values to the range [0, 1].
    gray = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0  # Normalize to [0,1]

    # Select the thresholding method specified by the user.
    if method == "manual":
        # === Manual RGB + HSV thresholding ===
        # Extract the red, green, and blue channels from the image array.
        r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
        # Create an RGB mask containing pixels whose red, green, and blue values all exceed their respective minimum thresholds.
        rgb_mask = (r > R_min) & (g > G_min) & (b > B_min)

        # Normalize the RGB image from the range [0, 255] to [0, 1] for HSV conversion.
        norm = image_np / 255.0
        # Convert the normalized RGB image into HSV color space using the rgb_to_hsv function.
        hsv = rgb_to_hsv(norm)
        # Extract the value/brightness channel from the HSV image.
        v = hsv[:, :, 2]
        # Create an HSV mask containing pixels whose value exceeds the specified minimum threshold.
        hsv_mask = v > V_min

        # Combine the RGB and HSV masks so that a pixel must satisfy both thresholding conditions.
        mask = rgb_mask & hsv_mask

    # Use Otsu's method to automatically determine a global grayscale threshold.
    elif method == "otsu":
        # Calculate the Otsu threshold that best separates the grayscale image into two intensity classes.
        threshold = threshold_otsu(gray)
        # Create a binary mask containing pixels whose grayscale intensity exceeds the Otsu threshold.
        mask = gray > threshold

    # Use adaptive thresholding based on the local mean grayscale intensity.
    elif method == "adaptive":
        # Calculate the mean grayscale intensity within a local neighborhood around each pixel.
        local_mean = uniform_filter(gray, size=adaptive_block_size)
        # Create a binary mask containing pixels brighter than their local mean plus the specified offset.
        mask = gray > (local_mean + adaptive_offset)

    # Use a simple fixed grayscale threshold to identify pixels brighter than 0.25.
    elif method == 'simple gray':
        # Create a mask identifying pixels with grayscale intensity below 0.25.
        inv_mask = gray < 0.25
        # Invert the low-intensity mask so that pixels with grayscale intensity at or above 0.25 are included.
        mask = ~inv_mask
    # Raise an error if the supplied thresholding method is not recognized.
    else:
        raise ValueError("Unknown method. Choose from: 'manual', 'otsu', or 'adaptive'.")

    # Convert the boolean mask into an 8-bit grayscale image where False = 0 (black) and True = 255 (white).
    mask_img = (mask * 255).astype(np.uint8)

    # === Display ===
    # Create a Matplotlib figure containing a single set of axes with dimensions of 10 by 10 inches.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Display the generated binary mask as a grayscale image.
    ax.imshow(mask_img, cmap='gray')
    # Add a title identifying the binary mask and thresholding method used.
    ax.set_title(f"Binary Mask ({method})")
    # Hide the axis markings around the displayed mask.
    ax.axis("off")
    # Automatically adjust the spacing around the plot to prevent unnecessary margins or overlap.
    plt.tight_layout()
    # Display the figure containing the binary mask.
    plt.show()

    # Return the boolean binary mask for use in subsequent image-processing operations.
    return mask

def match_masks(cut_mask, match_mask, modify_rows_left, modify_rows_right, modify_cols_top, modify_cols_bottom):

    '''Modify the dimensions of one binary mask, rescale it to match the width of a reference mask while preserving its aspect ratio, and overlay the modified mask in red on the reference mask.
        Input:
            cut_mask - 2D NumPy array containing the binary mask to be modified and overlaid, with values represented as either 0 and 1 or False and True.
            match_mask - 2D NumPy array containing the binary reference mask whose dimensions determine the final output size.
            modify_rows_left - Integer specifying the number of columns to add to (>0) or remove from (<0) the left side of cut_mask.
            modify_rows_right - Integer specifying the number of columns to add to (>0) or remove from (<0) the right side of cut_mask.
            modify_cols_top - Integer specifying the number of rows to add to (>0) or remove from (<0) the top of cut_mask.
            modify_cols_bottom - Integer specifying the number of rows to add to (>0) or remove from (<0) the bottom of cut_mask.
        Output:
            cut_mask - 2D NumPy boolean array containing the modified and resized cut_mask, with the same height and width as match_mask.
    '''
    # Apply row (vertical) modifications
    # Remove rows from the top of cut_mask when a negative top modification is specified.
    if modify_cols_top < 0:
        cut_mask = cut_mask[-modify_cols_top:, :]
    # Add rows of zeros to the top of cut_mask when a positive top modification is specified.
    elif modify_cols_top > 0:
        cut_mask = np.pad(cut_mask, ((modify_cols_top, 0), (0, 0)), mode='constant')

    # Remove rows from the bottom of cut_mask when a negative bottom modification is specified.
    if modify_cols_bottom < 0:
        cut_mask = cut_mask[:modify_cols_bottom, :]
    # Add rows of zeros to the bottom of cut_mask when a positive bottom modification is specified.
    elif modify_cols_bottom > 0:
        cut_mask = np.pad(cut_mask, ((0, modify_cols_bottom), (0, 0)), mode='constant')

    # Apply column (horizontal) modifications
    # Remove columns from the left of cut_mask when a negative left modification is specified.
    if modify_rows_left < 0:
        cut_mask = cut_mask[:, -modify_rows_left:]
    # Add columns of zeros to the left of cut_mask when a positive left modification is specified.
    elif modify_rows_left > 0:
        cut_mask = np.pad(cut_mask, ((0, 0), (modify_rows_left, 0)), mode='constant')

    # Remove columns from the right of cut_mask when a negative right modification is specified.
    if modify_rows_right < 0:
        cut_mask = cut_mask[:, :modify_rows_right]
    # Add columns of zeros to the right of cut_mask when a positive right modification is specified.
    elif modify_rows_right > 0:
        cut_mask = np.pad(cut_mask, ((0, 0), (0, modify_rows_right)), mode='constant')

    # Resize cut_mask to match match_mask's width, preserving aspect ratio
    # Extract the height and width of the reference mask.
    H, W = match_mask.shape
    # Extract the current height and width of the modified cut mask.
    h, w = cut_mask.shape
    # Calculate the new height required to resize cut_mask to the reference mask's width while preserving its aspect ratio.
    new_height = int(h * (W / w))
    # Resize cut_mask to the calculated dimensions using nearest-neighbor interpolation to preserve its binary structure.
    cut_mask = resize(cut_mask.astype(float), (new_height, W), order=0, preserve_range=True).astype(bool)

    # Crop vertically from bottom if resized cut_mask is too tall
    # If the resized mask is taller than the reference mask, crop excess rows from the bottom.
    if new_height > H:
        cut_mask = cut_mask[:H, :]
    # Otherwise, pad the bottom of the resized mask with zeros until it matches the reference height.
    else:
        # Calculate the number of rows needed to make cut_mask the same height as match_mask.
        pad_rows = H - new_height
        # Add the required number of zero-valued rows to the bottom of cut_mask.
        cut_mask = np.pad(cut_mask, ((0, pad_rows), (0, 0)), mode='constant')

    # Create RGB overlay image from match_mask
    # Replicate the single-channel reference mask three times to create an RGB image.
    overlay = np.stack([match_mask]*3, axis=-1).astype(float)

    # Apply red color to overlayed cut_mask region
    # Set the red channel to full intensity wherever cut_mask contains True values.
    overlay[cut_mask, 0] = 1.0  # Red
    # Set the green channel to zero wherever cut_mask contains True values.
    overlay[cut_mask, 1] = 0.0  # Green
    # Set the blue channel to zero wherever cut_mask contains True values.
    overlay[cut_mask, 2] = 0.0  # Blue

    # Plot the result
    # Create a large Matplotlib figure for displaying the overlay.
    plt.figure(figsize=(20, 20))
    # Display the reference mask with the modified cut mask highlighted in red.
    plt.imshow(overlay)
    # Add a title identifying the overlaid masks.
    plt.title("Top Mask in Red over Base Mask")
    # Hide the axis markings around the displayed image.
    plt.axis('off')
    # Display the overlay figure.
    plt.show()

    # Return the modified cut mask after resizing and padding/cropping to match the reference mask dimensions.
    return cut_mask

def add_rows_to_match(cut_mask, particle_mask):

    '''Pad a binary mask with ones along the bottom and right edges so that its dimensions match a reference particle mask.
        Input:
            cut_mask - 2D NumPy array containing the binary mask to be padded, with values represented as either 0 and 1 or False and True.
            particle_mask - 2D NumPy array containing the reference particle mask whose height and width determine the target dimensions for cut_mask.
        Output:
            cut_mask - 2D NumPy array containing the padded mask with the same height and width as particle_mask. Newly added rows and columns contain values of 1.
    '''
    # Calculate the number of rows that must be added to cut_mask to match the height of particle_mask.
    rows_to_add = particle_mask.shape[0] - cut_mask.shape[0]
    # Calculate the number of columns that must be added to cut_mask to match the width of particle_mask.
    cols_to_add = particle_mask.shape[1] - cut_mask.shape[1]

    # Pad bottom (rows) and right (columns) with 1s
    # Add the required number of rows to the bottom of cut_mask and fill the new rows with 1s.
    cut_mask = np.pad(cut_mask, ((0, rows_to_add), (0, 0)), mode='constant', constant_values=1)
    # Add the required number of columns to the right of cut_mask and fill the new columns with 1s.
    cut_mask = np.pad(cut_mask, ((0, 0), (0, cols_to_add)), mode='constant', constant_values=1)

    # Display the padded mask for visual inspection.
    display_mask(cut_mask, 10, 10)
    # Return the padded mask with dimensions matching particle_mask.
    return cut_mask

def isolate_empty_space(cut_mask, remove_particle_size = 5000000, small_object_size = 2000, max_column_gap1 = 200, max_row_gap = 200, max_filled_col_gap = 500,
                        max_column_gap2 = 50, enhance_large_gap_size = 200000, large_hole_threshold = 2000000, plot = True):    

    '''Process a binary mask to isolate and clean the empty space surrounding particles by removing particles, filling gaps, and removing small holes and objects.
        Input:
            cut_mask - 2D NumPy array containing a binary mask of particles and empty space, with values represented as either 0 and 1 or False and True.
            remove_particle_size - Integer specifying the minimum connected-object size retained during the initial particle-removal step. Objects smaller than this value are removed.
            small_object_size - Integer specifying the minimum connected-object size retained after the mask is inverted to remove small unwanted regions.
            max_column_gap1 - Integer specifying the maximum vertical gap size that can be filled during the first column-gap filling operation.
            max_row_gap - Integer specifying the maximum horizontal gap size that can be filled during the row-gap filling operation.
            max_filled_col_gap - Integer specifying the maximum vertical gap size that can be filled during the second column-gap filling operation after the mask is inverted.
            max_column_gap2 - Integer specifying the maximum vertical gap size that can be filled during the final column-gap filling operation.
            enhance_large_gap_size - Integer specifying the minimum connected-object size retained during the later cleaning step, removing smaller objects to enhance the continuous empty-space region.
            large_hole_threshold - Integer specifying the maximum hole area that can be filled during the final hole-removal operation.
            plot - Boolean specifying whether intermediate masks and processing descriptions should be displayed during processing.
        Output:
            cut_mask - 2D NumPy uint8 array containing the processed mask, where 0 and 1 represent the final isolated empty-space regions.
    '''
    # Convert the input mask to a boolean array so that subsequent morphological operations treat it as a binary mask.
    cut_mask = cut_mask.astype(bool)

    # Remove connected objects smaller than the specified size, primarily eliminating smaller particles from the initial mask.
    cut_mask = morphology.remove_small_objects(cut_mask, min_size=remove_particle_size)
    # Display the mask after removing most particles if plotting is enabled.
    if plot:
        print('Remove most particles from mask')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Invert the mask so that the previously empty regions become the regions of interest.
    cut_mask = ~cut_mask
    # Display the inverted mask if plotting is enabled.
    if plot:
        print('Invert Mask')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Remove connected objects smaller than the specified size to eliminate small isolated regions and image artifacts.
    cut_mask = morphology.remove_small_objects(cut_mask, min_size=small_object_size)
    # Display the mask after removing small unwanted regions if plotting is enabled.
    if plot:
        print('Remove tiny specs')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill vertical gaps within columns that are no larger than the specified maximum gap size.
    cut_mask = fill_column_gaps(cut_mask, max_gap = max_column_gap1)
    # Display the mask after filling the first set of column gaps if plotting is enabled.
    if plot:
        print('Remove gaps in columns')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill horizontal gaps within rows that are no larger than the specified maximum gap size.
    cut_mask = fill_row_gaps2(cut_mask, max_gap = max_row_gap)
    # Display the mask after filling row gaps if plotting is enabled.
    if plot:
        print('Remove gaps in rows')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Invert the mask, fill vertical gaps in the inverted mask, and invert it again to remove excess noise from shadow regions.
    cut_mask = fill_column_gaps(~cut_mask, max_gap = max_filled_col_gap)
    cut_mask = ~cut_mask
    # Display the mask after cleaning excess noise from shadow regions if plotting is enabled.
    if plot:
        print('Remove excess noise from shadows')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill remaining vertical gaps that are no larger than the second specified maximum gap size.
    cut_mask = fill_column_gaps(cut_mask, max_gap=max_column_gap2)
    # Display the mask after filling the remaining column gaps if plotting is enabled.
    if plot:
        print('remove remaining column gaps')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Optional step for removing small holes from the mask. Currently disabled.
    # cut_mask = morphology.remove_small_holes(cut_mask, area_threshold=small_hole_threshold)
    # if plot:
    #     print('Remove leftover holes')
    #     display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Remove connected objects smaller than the specified size to further isolate and enhance the large continuous empty-space region.
    cut_mask = morphology.remove_small_objects(cut_mask, min_size=enhance_large_gap_size)
    # Display the mask after removing smaller excess regions if plotting is enabled.
    if plot:
        print('Enhance blank area by removing excess particles')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill holes smaller than the specified area threshold to eliminate remaining enclosed regions within the isolated empty space.
    cut_mask = morphology.remove_small_holes(cut_mask, area_threshold=large_hole_threshold)
    # Display the final processed mask after removing extraneous holes if plotting is enabled.
    if plot:
        print('Remove the rest of the extraneous holes')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Convert the final boolean mask to an 8-bit unsigned integer array containing 0s and 1s and return it.
    return cut_mask.astype(np.uint8)

def isolate_empty_space2(cut_mask, remove_particle_size = 5000000, small_object_size = 2000, max_column_gap1 = 200, max_row_gap = 200, max_filled_col_gap = 500,
                        max_filled_row_gap = 500, max_column_gap2 = 50, enhance_large_gap_size = 200000, large_hole_threshold = 2000000, plot = True):    
    '''
    Process a binary particle mask to isolate and clean large empty-space regions using morphological filtering,
    gap filling, mask inversion, and hole removal.
        Input:
            cut_mask - 2D NumPy array containing a binary mask, where nonzero or True values represent particles
            or regions of interest and zero or False values represent the background or empty space.
            remove_particle_size - Integer specifying the minimum size of connected particle objects retained
            during the initial removal of small particles.
            small_object_size - Integer specifying the minimum size of connected regions retained after the
            mask is inverted.
            max_column_gap1 - Integer specifying the maximum vertical gap size that will be filled within each
            column during the first column-gap filling operation.
            max_row_gap - Integer specifying the maximum horizontal gap size that will be filled within each
            row during the first row-gap filling operation.
            max_filled_col_gap - Integer specifying the maximum vertical gap size that will be filled within
            columns while processing the inverted mask to remove excess noise from shadows.
            max_filled_row_gap - Integer specifying the maximum horizontal gap size that will be filled within
            rows while processing the inverted mask to remove excess noise from shadows.
            max_column_gap2 - Integer specifying the maximum vertical gap size that will be filled during the
            final column-gap filling operation.
            enhance_large_gap_size - Integer specifying the minimum size of connected regions retained during
            the final removal of smaller regions to enhance the large empty-space area.
            large_hole_threshold - Integer specifying the maximum hole area that will be filled during the
            final hole-removal operation.
            plot - Boolean specifying whether intermediate processing results should be displayed.
        Output:
            cut_mask - 2D NumPy uint8 array containing the processed empty-space mask, where retained regions
            are represented by 1 and the background is represented by 0.
    '''

    # Convert the input mask to a boolean array so that all subsequent morphological operations use a binary mask.
    cut_mask = cut_mask.astype(bool)

    # Remove connected particle objects smaller than the specified size, retaining only sufficiently large particle regions.
    cut_mask = morphology.remove_small_objects(cut_mask, min_size=remove_particle_size)
    # Display the mask after removing smaller particle regions if plotting is enabled.
    if plot:
        print('Remove most particles from mask')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Invert the mask so that the previously identified background or empty-space regions become the foreground.
    cut_mask = ~cut_mask
    # Display the inverted mask if plotting is enabled.
    if plot:
        print('Invert Mask')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Remove small connected regions from the inverted mask to eliminate small isolated background features or artifacts.
    cut_mask = morphology.remove_small_objects(cut_mask, min_size=small_object_size)
    # Display the mask after removing small regions if plotting is enabled.
    if plot:
        print('Remove tiny specs')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill vertical gaps within each column that are no larger than the specified maximum gap size.
    cut_mask = fill_column_gaps(cut_mask, max_gap = max_column_gap1)
    # Display the mask after filling the initial column gaps if plotting is enabled.
    if plot:
        print('Remove gaps in columns')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill horizontal gaps within each row that are no larger than the specified maximum gap size.
    cut_mask = fill_row_gaps2(cut_mask, max_gap = max_row_gap)
    # Display the mask after filling the initial row gaps if plotting is enabled.
    if plot:
        print('Remove gaps in rows')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Invert the mask and fill vertical gaps to remove excess noise associated with shadows or other narrow interruptions.
    cut_mask = fill_column_gaps(~cut_mask, max_gap = max_filled_col_gap)
    # Fill horizontal gaps in the same inverted mask to further remove noise from shadow regions.
    cut_mask = fill_row_gaps2(cut_mask, max_gap = max_filled_row_gap)
    # Invert the mask again to restore the original foreground/background interpretation.
    cut_mask = ~cut_mask
    # Display the mask after removing excess shadow-related noise if plotting is enabled.
    if plot:
        print('Remove excess noise from shadows')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill remaining vertical gaps that are no larger than the specified final column-gap threshold.
    cut_mask = fill_column_gaps(cut_mask, max_gap=max_column_gap2)
    # Display the mask after the final column-gap filling operation if plotting is enabled.
    if plot:
        print('remove remaining column gaps')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # cut_mask = morphology.remove_small_holes(cut_mask, area_threshold=small_hole_threshold)
    # if plot:
    #     print('Remove leftover holes')
    #     display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Remove connected regions smaller than the specified size to retain only large regions corresponding to the desired empty space.
    cut_mask = morphology.remove_small_objects(cut_mask, min_size=enhance_large_gap_size)
    # Display the mask after removing smaller regions if plotting is enabled.
    if plot:
        print('Enhance blank area by removing excess particles')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Fill holes smaller than the specified area threshold to eliminate remaining enclosed extraneous regions.
    cut_mask = morphology.remove_small_holes(cut_mask, area_threshold=large_hole_threshold)
    # Display the final processed empty-space mask if plotting is enabled.
    if plot:
        print('Remove the rest of the extraneous holes')
        display_mask(cut_mask.astype(np.uint8), 10, 10)

    # Convert the final boolean mask to an 8-bit unsigned integer array and return it.
    return cut_mask.astype(np.uint8)

def mask_outline(mask, erosion_iterations=1):
    '''
    Extracts the outline of the foreground regions in a binary mask by
    comparing the original mask to an eroded version of the same mask. Pixels
    that are present in the original mask but removed during erosion are
    identified as the boundary or outline. Increasing the number of erosion
    iterations increases the thickness of the resulting outline.

    Inputs:
        mask (numpy.ndarray): A 2D binary mask containing Boolean values or
            0/1 values, where foreground pixels represent the regions whose
            outline should be extracted.
        erosion_iterations (int): The number of binary erosion iterations
            applied to the input mask. Larger values produce a thicker
            extracted outline.

    Outputs:
        numpy.ndarray: A 2D Boolean array containing the outline of the
            foreground regions. Pixels belonging to the outline are True,
            while all other pixels are False.
    '''

    eroded = binary_erosion(mask, iterations=erosion_iterations) #Erode the foreground regions by the specified number of iterations
    outline = mask & ~eroded #Identify pixels that are present in the original mask but absent from the eroded mask
    return outline #Return the Boolean mask containing only the outline pixels

def row_threshold_mask(mask, threshold):
    '''
    Sets entire rows of a binary mask to 1 when the number of foreground
    pixels in that row exceeds the specified threshold. The function first
    ensures that the input is represented as a binary mask, counts the number
    of foreground pixels in each row, identifies rows whose counts exceed the
    threshold, and fills those entire rows with 1s.

    Inputs:
        mask (numpy.ndarray): A 2D binary array containing Boolean or integer
            values, where values greater than 0 represent foreground pixels.
        threshold (int): The minimum number of foreground pixels that a row
            must contain before the entire row is set to 1.

    Outputs:
        numpy.ndarray: A 2D NumPy array of dtype uint8 containing the modified
            binary mask. Rows containing more than threshold foreground pixels
            are completely filled with 1s, while all other pixels retain their
            original binary values.
    '''

    # Ensure input is binary
    mask = (mask > 0).astype(np.uint8) #Convert the input mask into a binary uint8 array where all positive values become 1 and all other values become 0

    # Count 1s in each row
    row_sums = mask.sum(axis=1) #Count the number of foreground pixels in each row of the binary mask

    # Find rows exceeding threshold
    rows_to_fill = row_sums > threshold #Create a Boolean array identifying rows containing more foreground pixels than the specified threshold

    # Set entire rows to 1
    mask[rows_to_fill, :] = 1 #Set every pixel in each identified row to 1

    return mask #Return the modified binary mask

def column_threshold_mask(mask, threshold):
    '''
    Sets entire columns of a binary mask to 1 when the number of foreground
    pixels in that column exceeds the specified threshold. The function first
    ensures that the input is represented as a binary mask, counts the number
    of foreground pixels in each column, identifies columns whose counts exceed
    the threshold, and fills those entire columns with 1s.

    Inputs:
        mask (numpy.ndarray): A 2D binary array containing Boolean or integer
            values, where values greater than 0 represent foreground pixels.
        threshold (int): The minimum number of foreground pixels that a column
            must contain before the entire column is set to 1.

    Outputs:
        numpy.ndarray: A 2D NumPy array of dtype uint8 containing the modified
            binary mask. Columns containing more than threshold foreground
            pixels are completely filled with 1s, while all other pixels retain
            their original binary values.
    '''

    # Ensure input is binary
    mask = (mask > 0).astype(np.uint8) #Convert the input mask into a binary uint8 array where all positive values become 1 and all other values become 0

    # Count 1s in each row
    col_sums = mask.sum(axis=0) #Count the number of foreground pixels in each column of the binary mask

    # Find rows exceeding threshold
    cols_to_fill = col_sums > threshold #Create a Boolean array identifying columns containing more foreground pixels than the specified threshold

    # Set entire rows to 1
    mask[:, cols_to_fill] = 1 #Set every pixel in each identified column to 1

    return mask #Return the modified binary mask

def fill_small_holes(mask, max_hole_size=10):

    '''Fill small, fully enclosed holes in a binary mask while leaving larger holes and regions connected to the image border unchanged.
        Input:
            mask - 2D NumPy boolean array containing a binary mask, where True values represent foreground regions and False values represent background or holes.
            max_hole_size - Integer specifying the maximum number of pixels that an enclosed hole can contain to be filled.
        Output:
            mask - 2D NumPy boolean array with the same shape as the input mask, where enclosed holes at or below max_hole_size have been filled with True values.
    '''
    # Step 1: Invert the mask to identify holes
    # Invert the binary mask so that background regions become True and can be identified as connected components.
    inverse_mask = ~mask

    # Step 2: Label connected components in the inverted mask (holes)
    # Label each connected region in the inverted mask with a unique integer identifier.
    labeled_holes, num_features = label(inverse_mask)

    # Step 3: Fill only the small, enclosed holes
    # Create an empty boolean array to store holes that meet the size and enclosure criteria.
    small_holes = np.zeros_like(mask, dtype=bool)
    
    # Iterate through each labeled connected component, excluding label 0 which represents the background.
    for i in range(1, num_features + 1):
        # Create a boolean mask identifying the current connected component.
        region = labeled_holes == i
        # Skip regions that touch the border (not enclosed)
        # Check whether the current region touches any edge of the image; regions that do are not considered enclosed holes.
        if np.any(region[0, :]) or np.any(region[-1, :]) or \
           np.any(region[:, 0]) or np.any(region[:, -1]):
            continue
        # Fill hole if it's smaller than the threshold
        # Calculate the number of pixels in the enclosed region and retain it if its size is at or below the specified threshold.
        if np.sum(region) <= max_hole_size:
            small_holes |= region

    # Combine the original mask with all qualifying small holes to fill them while leaving all other regions unchanged.
    return mask | small_holes

def fill_small_holes2(binary_mask, max_hole_size, connectivity=1):
    '''
    Fills regions of background pixels (0s) within a binary mask when the
    connected region is smaller than or equal to the specified maximum hole
    size. The function inverts the binary mask so that background regions can
    be identified and labeled, determines connected regions using either
    4-connectivity or 8-connectivity, and fills qualifying regions by setting
    their pixels to 1.

    Inputs:
        binary_mask (numpy.ndarray): A 2D binary mask containing 0s and 1s,
            where 0 represents background or a hole and 1 represents the
            foreground region.
        max_hole_size (int): The maximum number of pixels that a background
            region can contain to be considered a small hole and filled.
        connectivity (int): Defines the pixel connectivity used when labeling
            connected regions. A value of 1 uses 4-connectivity, while a value
            of 2 uses 8-connectivity.

    Outputs:
        numpy.ndarray: A copy of the input binary mask in which all connected
            background regions containing fewer than or equal to
            max_hole_size pixels have been filled with 1s.
    '''

    # Invert the mask to label holes (0s become 1s)
    inverted_mask = (binary_mask == 0).astype(np.uint8) #Create an inverted binary mask in which the original background pixels become 1 and foreground pixels become 0
    #display_mask(inverted_mask,10,10)



    # Label connected regions of 0s
    labeled_holes, num_features = label(inverted_mask, structure=np.ones((3,3)) if connectivity == 2 else None) #Label each connected background region using 8-connectivity if requested, otherwise use the default 4-connectivity
    #print(num_features)



    # Create output copy
    filled_mask = binary_mask.copy() #Create a copy of the original binary mask so that the input array is not modified directly



    for region_label in range(1, num_features + 1): #Loop through each labeled background region, excluding the background label of 0

        region = (labeled_holes == region_label) #Create a Boolean mask identifying all pixels belonging to the current background region
        #print(np.sum(region))

        if np.sum(region) <= max_hole_size: #Check whether the current background region contains no more pixels than the maximum allowed hole size

            #print(np.sum(region))

            filled_mask[region] = 1 #Fill the current hole by changing all of its pixels from 0 to 1

    return filled_mask #Return the binary mask with all sufficiently small holes filled


##Chain Analysis
def get_distinct_color(i, saturation=1.0, value=1.0, seed=42):

    '''Generate a visually distinct RGBA color by distributing hues around the HSV color space with a small reproducible random variation.
        Input:
            i - Integer index used to determine the base hue of the generated color.
            saturation - Float specifying the saturation component of the HSV color, typically in the range [0, 1], where 1.0 represents full saturation.
            value - Float specifying the brightness component of the HSV color, typically in the range [0, 1], where 1.0 represents full brightness.
            seed - Integer used to initialize the pseudo-random number generator so that the hue variation is reproducible.
        Output:
            tuple - Four-element tuple containing the red, green, blue, and alpha components of the generated color, with each component represented as a float in the range [0, 1].
    '''
    # Golden ratio for hue spacing
    # Use the fractional part of the golden ratio to distribute successive hues around the color wheel with good visual separation.
    golden_ratio_conjugate = 0.61803398875
    
    # Create a pseudo-random but reproducible offset
    # Initialize a random number generator using the seed and color index so that each color receives a consistent variation.
    rng = random.Random(seed + i)
    # Generate a small random adjustment to the hue to prevent the colors from following a perfectly uniform sequence.
    jitter = rng.uniform(-0.03, 0.03)  # Small hue jitter

    # Apply hue spacing and jitter
    # Calculate the hue by combining the index-based golden-ratio spacing with the random jitter and wrapping the result to [0, 1).
    h = (i * golden_ratio_conjugate + jitter) % 1.0
    # Convert the HSV color components into RGB values using the calculated hue, saturation, and brightness.
    r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
    # Return the RGB color components along with an alpha value of 1.0, representing full opacity.
    return (r, g, b, 1.0)  # RGBA

def label_mask_16(filtered_particle_mask, original_image, particle_bounds, disk_size=1, connectivity=1,
                 branch_length_fraction=0.05, global_min_branch_length=5, min_region_size=10, debug_plots=True,
                 prune=True, prune_branch_length=20, max_hole_size=10, vertical_prune_length = 3):

    '''Identify and label individual branches within a binary particle mask using morphological cleaning, watershed segmentation, skeletonization, branch pruning, and skeleton-based traversal.
        Input:
            filtered_particle_mask - 2D NumPy binary array containing the particle mask to be segmented, where nonzero or True values represent particle regions.
            original_image - PIL Image or array-like image containing the original image corresponding to filtered_particle_mask, used for visualization.
            particle_bounds - Iterable containing four integer coordinates in the order (ymin, ymax, xmin, xmax), defining the particle region to crop from the full mask and original image.
            disk_size - Integer specifying the radius of the disk-shaped structuring element used for morphological opening of the particle mask.
            connectivity - Integer specifying the pixel connectivity used when labeling connected regions in the cleaned mask.
            branch_length_fraction - Float specifying the fraction of the total skeleton length used to determine the minimum acceptable branch length during pruning.
            global_min_branch_length - Integer specifying the absolute minimum number of skeleton pixels required for a branch to be retained.
            min_region_size - Integer specifying the minimum number of pixels required for a region to undergo skeleton and branch analysis. Smaller regions are retained as single structures.
            debug_plots - Boolean specifying whether intermediate masks, skeletons, branchpoints, and segmentation results should be displayed.
            prune - Boolean specifying whether short branches should be iteratively removed from each skeleton.
            prune_branch_length - Integer specifying the minimum number of pixels required for a skeleton branch to survive the pruning process.
            max_hole_size - Integer specifying the maximum number of pixels in an enclosed hole that will be filled before skeletonization.
            vertical_prune_length - Integer specifying the minimum vertical segment length that will be removed from the skeleton during vertical-section pruning.
        Output:
            region_counter - Integer representing the number of processed regions, including the final increment after all regions have been analyzed.
            chain_mask - 2D NumPy array containing integer labels for the identified branches, where each nonzero integer corresponds to a distinct labeled branch.
            particle_region_masks - Dictionary mapping each branch label to a 2D NumPy uint16 mask containing that labeled branch.
    '''
    # Unpack the particle bounding box into minimum and maximum row and column coordinates.
    ymin, ymax, xmin, xmax = particle_bounds

    # Extract the particle region from the full binary mask and convert it to a 16-bit unsigned integer array.
    particle = filtered_particle_mask[ymin:ymax, xmin:xmax].astype(np.uint16)

    # Clean the particle mask using morphological opening with a disk-shaped structuring element to remove small protrusions and noise.
    cleaned = morphology.binary_opening(particle, morphology.disk(disk_size))

    # Label connected regions in the cleaned particle mask using the specified pixel connectivity.
    labeled_clean = measure.label(cleaned, connectivity=connectivity)

    # Calculate the Euclidean distance from each foreground pixel in the original particle mask to the nearest background pixel.
    distance = ndi.distance_transform_edt(particle)

    # Apply watershed segmentation to separate connected particle regions using the cleaned connected components as markers.
    resegmented = segmentation.watershed(-distance, markers=labeled_clean, mask=particle)

    # Convert the original image to a NumPy array for image processing and visualization.
    original_array = np.array(original_image)
    # Crop the original image to the same bounding box used for the particle mask.
    cropped_array = original_array[ymin:ymax, xmin:xmax]

    # Plot the cropped binary particle mask and corresponding region of the original image side-by-side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Display the extracted particle mask in grayscale.
    ax1.imshow(particle, cmap='gray')
    # Add a title identifying the displayed particle mask.
    ax1.set_title("Mask of Region")
    # Display the corresponding cropped region from the original image.
    ax2.imshow(cropped_array)
    # Add a title identifying the original image region.
    ax2.set_title("Actual Particle")
    # Hide axis markings on both displayed images.
    for ax in [ax1, ax2]: ax.axis('off')
    # Display the particle mask and original image.
    plt.show()

    # Show the result of watershed segmentation with each labeled region assigned a different color.
    plt.figure(figsize=(5, 5))
    # Convert the integer-labeled watershed result into a color overlay for visualization.
    plt.imshow(color.label2rgb(resegmented, bg_label=0, kind='overlay'))
    # Add a title identifying the watershed segmentation.
    plt.title("Watershed Cluster Segmentation")
    # Hide axis markings.
    plt.axis('off')
    # Display the watershed segmentation.
    plt.show()

    # Initialize outputs
    # Create a dictionary to store a separate labeled mask for each identified branch.
    particle_region_masks = {}  # Dictionary to hold labeled region masks
    # Create a dictionary mapping each branch label to its assigned display color.
    branch_color_lookup = {}    # Mapping from label to color
    # Create an empty integer mask that will contain the final branch labels.
    chain_mask = np.zeros_like(particle)  # Final labeled output mask
    # Initialize the integer label assigned to the first branch.
    branch_label = 1            # Counter for branch labels
    # Initialize the counter used to track the processed watershed regions.
    region_counter = 1          # Counter for regions

    # Process each region in the resegmented mask
    # Iterate through each connected region identified by watershed segmentation.
    for region in measure.regionprops(resegmented):
        # Create a boolean mask containing only the current watershed region.
        region_mask = resegmented == region.label

        # Fill small holes in the current region so that enclosed gaps do not interfere with skeletonization.
        region_mask_holes_filled = fill_small_holes(region_mask, max_hole_size=max_hole_size)

        # Display the original and hole-filled region masks when debugging is enabled.
        if debug_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            # Display the original watershed region.
            ax1.imshow(region_mask, cmap='gray', vmin=0, vmax=1)
            ax1.set_title("mask of region")
            ax1.axis('off')
            # Display the region after filling small enclosed holes.
            ax2.imshow(region_mask_holes_filled, cmap='gray', vmin=0, vmax=1)
            ax2.set_title("filled mask of region")
            ax2.axis('off')
            plt.show()

        # If the region is small, just assign a new label and skip further analysis
        # Check whether the region contains fewer pixels than the minimum region size.
        if np.count_nonzero(region_mask) < min_region_size:
            if debug_plots:
                # Report that the region is below the minimum size and will be retained without further branch analysis.
                print(f'Region {region_counter} of size {np.count_nonzero(region_mask)} is smaller than minimum region size of {min_region_size}')
                print('Add to mask as-is')
            # Convert the small region into a uint16 mask containing its assigned branch label.
            labeled_mask = region_mask.astype(np.uint16) * branch_label
            # Store the labeled region mask in the output dictionary.
            particle_region_masks[branch_label] = labeled_mask
            # Add the branch label to the final chain mask at the region's pixels.
            chain_mask[region_mask] = branch_label
            # Generate and store a distinct color for visualizing this branch.
            branch_color_lookup[branch_label] = get_distinct_color(branch_label)
            # Increment the branch label for the next structure.
            branch_label += 1
            # Increment the processed region counter.
            region_counter += 1
            # Skip the remainder of the branch analysis for this small region.
            continue

        # Skeletonize the region to get centerlines
        # Reduce the filled region to a one-pixel-wide skeleton representing its centerline structure.
        skeleton = morphology.thin(region_mask_holes_filled)

        # Display the skeletonized region when debugging is enabled.
        if debug_plots:
            plt.figure(figsize=(5, 5))
            plt.imshow(skeleton, cmap='gray')
            plt.title(f"Skeleton for Region {region.label}")
            plt.axis('off')
            plt.show()

        # Calculate pruning threshold based on skeleton length
        # Count the number of foreground pixels in the skeleton to estimate its total length.
        total_skel_length = np.count_nonzero(skeleton)
        # Calculate the minimum branch length as a fraction of the total skeleton length.
        min_branch_len = int(branch_length_fraction * total_skel_length)

        # Find endpoints and branchpoints using 3x3 convolution
        # Count the number of neighboring skeleton pixels around each skeleton pixel using a 3x3 convolution.
        neighbors = ndi.convolve(skeleton.astype(np.uint16), np.ones((3, 3)), mode='constant')
        # Identify skeleton pixels with exactly one neighboring skeleton pixel, corresponding to endpoints.
        endpoints = np.logical_and(skeleton, neighbors == 2)
        # Identify skeleton pixels with three or more neighboring skeleton pixels, corresponding to branchpoints.
        branchpoints = np.logical_and(skeleton, neighbors >= 4)
        # Extract the row and column coordinates of all detected endpoints.
        endpoint_coords = np.argwhere(endpoints)
        # Extract the row and column coordinates of all detected branchpoints.
        branchpoint_coords = np.argwhere(branchpoints)

        # Display skeleton and keypoints
        if debug_plots:
            # Create a 3-channel float image from the binary skeleton
            debug_img = np.stack([skeleton]*3, axis=-1).astype(float)

            # Color endpoints red (R=1.0, G=0, B=0)
            # Set endpoint pixels to red in the debug RGB image.
            for y, x in endpoint_coords:
                debug_img[y, x, 0] = 1.0  # Red channel
                debug_img[y, x, 1] = 0.0  # Optional: make sure green and blue are off
                debug_img[y, x, 2] = 0.0

            # Color branchpoints blue (R=0, G=0, B=1.0)
            # Set branchpoint pixels to blue in the debug RGB image.
            for y, x in branchpoint_coords:
                debug_img[y, x, 0] = 0.0
                debug_img[y, x, 1] = 0.0
                debug_img[y, x, 2] = 1.0

            # Display the skeleton with endpoints and branchpoints highlighted.
            plt.figure(figsize=(6, 6))
            plt.imshow(debug_img)
            plt.title(f"Pruned Skeleton with Endpoints (Red) and Branchpoints (Blue) — Region {region.label}")
            # plt.axis('off')  # Optional
            plt.show()

        # Iterative pruning of short branches
        if prune:
            # Continue pruning until an iteration removes no additional branches.
            skeleton_changed = True
            while skeleton_changed:
                skeleton_changed = False
                # Start each pruning iteration with a copy of the current skeleton.
                pruned = skeleton.copy()
                # Track which skeleton pixels have already been visited during branch traversal.
                visited = np.zeros_like(skeleton, dtype=bool)

                # Traverse from each endpoint
                for y0, x0 in endpoint_coords:
                    # Skip this endpoint if it has already been visited or no longer exists in the skeleton.
                    if visited[y0, x0] or not skeleton[y0, x0]: continue
                    # Initialize an array to store the coordinates belonging to the current branch.
                    branch_coords = []
                    # Start a stack containing the current endpoint for branch traversal.
                    queue = [(y0, x0)]

                    # Follow the branch until a branchpoint is reached
                    while queue:
                        # Remove the next pixel coordinate from the traversal stack.
                        y, x = queue.pop()
                        # Skip pixels that have already been visited or are no longer part of the skeleton.
                        if visited[y, x] or not skeleton[y, x]: continue
                        # Stop traversal when a branchpoint other than the starting endpoint is reached.
                        if (y, x) != (y0, x0) and branchpoints[y, x]: break
                        # Mark the current skeleton pixel as visited.
                        visited[y, x] = True
                        # Add the current pixel to the branch coordinate list.
                        branch_coords.append((y, x))
                        # Examine all eight neighboring pixels around the current pixel.
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                # Skip the current pixel itself.
                                if dy == dx == 0: continue
                                # Calculate the neighboring pixel coordinates.
                                ny, nx = y + dy, x + dx
                                # Ensure the neighboring coordinates are inside the skeleton array.
                                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                                    # Add unvisited skeleton neighbors to the traversal stack.
                                    if skeleton[ny, nx] and not visited[ny, nx]:
                                        queue.append((ny, nx))

                    # Remove the branch if it's too short
                    # Remove branches that do not meet either the fractional or absolute minimum length requirement.
                    if len(branch_coords) < min_branch_len or len(branch_coords) < prune_branch_length:
                        for y, x in branch_coords:
                            pruned[y, x] = 0
                        # Record that the skeleton was modified so another pruning iteration will be performed.
                        skeleton_changed = True

                # Clean up and update skeleton
                # Dilate the pruned skeleton and thin it again to reconnect and clean the remaining centerline structure.
                skeleton = morphology.thin(morphology.dilation(pruned, morphology.square(3)))
                # Recalculate the number of neighboring skeleton pixels after pruning.
                neighbors = ndi.convolve(skeleton.astype(np.uint16), np.ones((3, 3)), mode='constant')
                # Recalculate the endpoints of the modified skeleton.
                endpoints = np.logical_and(skeleton, neighbors == 2)
                #branchpoints = np.logical_and(skeleton, neighbors >= 4)
                # Extract the updated endpoint coordinates.
                endpoint_coords = np.argwhere(endpoints)
                #branchpoint_coords = np.argwhere(branchpoints)

                # Plot pruned skeleton
                if debug_plots:
                    # Create a 3-channel float image from the binary skeleton
                    debug_img = np.stack([skeleton]*3, axis=-1).astype(float)

                    # Color endpoints red (R=1.0, G=0, B=0)
                    # Highlight the updated endpoint pixels in red.
                    for y, x in endpoint_coords:
                        debug_img[y, x, 0] = 1.0  # Red channel
                        debug_img[y, x, 1] = 0.0  # Optional: make sure green and blue are off
                        debug_img[y, x, 2] = 0.0

                    # Color branchpoints blue (R=0, G=0, B=1.0)
                    # Highlight the existing branchpoint pixels in blue.
                    for y, x in branchpoint_coords:
                        debug_img[y, x, 0] = 0.0
                        debug_img[y, x, 1] = 0.0
                        debug_img[y, x, 2] = 1.0

                    # Display the updated skeleton after pruning.
                    plt.figure(figsize=(6, 6))
                    plt.imshow(debug_img)
                    plt.title(f"Pruned Skeleton with Endpoints (Red) and Branchpoints (Blue) — Region {region.label}")
                    # plt.axis('off')  # Optional
                    plt.show()

            # Create a mask for vertical skeleton segments that meet the vertical pruning length criterion.
            vertical_mask = np.zeros_like(skeleton, dtype=bool)
            # Iterate over each column of the skeleton to identify vertical segments.
            for x in range(skeleton.shape[1]):  # Iterate over each column
                # Extract the current column of the skeleton.
                col = skeleton[:, x]
                # Track whether the current scan is inside a continuous vertical segment.
                in_segment = False
                # Initialize the starting row of the current vertical segment.
                start = 0

                # Iterate through every row plus one additional position to close a segment at the image boundary.
                for y in range(skeleton.shape[0] + 1):  # One extra iteration to close open segment
                    # Check whether the current row contains a skeleton pixel.
                    if y < skeleton.shape[0] and col[y]:
                        # Start a new vertical segment when entering a skeleton region.
                        if not in_segment:
                            in_segment = True
                            start = y
                    else:
                        # Process the vertical segment when its end is reached.
                        if in_segment:
                            end = y
                            # Calculate the length of the vertical segment.
                            segment_length = end - start
                            # Mark segments meeting the threshold for removal.
                            if segment_length >= vertical_prune_length:
                                vertical_mask[start:end, x] = True  # Mark for removal
                            in_segment = False

            # Only now remove those pixels, after scanning is complete
            # Remove all skeleton pixels belonging to the identified vertical segments.
            skeleton[vertical_mask] = 0
            # Recalculate the number of neighboring skeleton pixels after vertical pruning.
            neighbors = ndi.convolve(skeleton.astype(np.uint16), np.ones((3, 3)), mode='constant')
            # Recalculate the endpoints after vertical pruning.
            endpoints = np.logical_and(skeleton, neighbors == 2)
            # Identify newly created branchpoints after vertical pruning.
            new_branchpoints = np.logical_and(skeleton, neighbors >= 4)
            # Extract the updated endpoint coordinates.
            endpoint_coords = np.argwhere(endpoints)
            # Extract coordinates of branchpoints using the existing branchpoint mask.
            new_branchpoint_coords = np.argwhere(branchpoints)

            # Add the newly identified branchpoints to the existing branchpoint mask.
            branchpoints |= new_branchpoints
            # Convert the original branchpoint coordinates into a set for efficient comparison.
            existing_coords_set = set(map(tuple, branchpoint_coords))
            # Convert the newly identified branchpoint coordinates into a set for efficient comparison.
            new_coords_set = set(map(tuple, new_branchpoint_coords))

            # Find only the new ones not already in the original
            # Determine which branchpoint coordinates are not already present in the original set.
            unique_new_coords = np.array(list(new_coords_set - existing_coords_set))

            # Append the unique new coordinates to the original
            # Add newly identified branchpoint coordinates to the existing branchpoint coordinate array.
            if unique_new_coords.size > 0:
                branchpoint_coords = np.vstack([branchpoint_coords, unique_new_coords])

            # Display the skeleton after vertical pruning when debugging is enabled.
            if debug_plots:
                # Create a 3-channel float image from the binary skeleton
                debug_img = np.stack([skeleton]*3, axis=-1).astype(float)

                # Color endpoints red (R=1.0, G=0, B=0)
                # Highlight endpoint pixels in red.
                for y, x in endpoint_coords:
                    debug_img[y, x, 0] = 1.0  # Red channel
                    debug_img[y, x, 1] = 0.0  # Optional: make sure green and blue are off
                    debug_img[y, x, 2] = 0.0

                # Color branchpoints blue (R=0, G=0, B=1.0)
                # Highlight branchpoint pixels in blue.
                for y, x in branchpoint_coords:
                    debug_img[y, x, 0] = 0.0
                    debug_img[y, x, 1] = 0.0
                    debug_img[y, x, 2] = 1.0

                # Display the vertically pruned skeleton with its keypoints.
                plt.figure(figsize=(6, 6))
                plt.imshow(debug_img)
                plt.title(f"Pruned Skeleton with Vertical sections removed")
                # plt.axis('off')  # Optional
                plt.show()

        # Traverse from endpoints/branchpoints and assign labels to valid branches
        # Track which skeleton pixels have already been assigned to a branch.
        visited = np.zeros_like(skeleton, dtype=bool)
        # Combine endpoint and branchpoint coordinates into a set of starting points for branch traversal.
        visit_coords = np.vstack((endpoint_coords, branchpoint_coords))
        # Sort starting points by their column coordinate to encourage left-to-right traversal.
        visit_coords = visit_coords[np.argsort(visit_coords[:, 1])]
        # Create an integer mask for storing the labels assigned to each skeleton branch.
        skeletons_mask = np.zeros_like(skeleton, dtype=np.int32)
        
        # If no branchpoints were identified, treat the entire skeleton as a single structure.
        if len(branchpoint_coords) == 0:
            if debug_plots:
                print(f"No branchpoints in Region {region.label}. Filling as one structure.")

            # Assign the current branch label to every skeleton pixel.
            skeletons_mask[skeleton] = branch_label
            # Generate and store a distinct visualization color for the branch.
            branch_color_lookup[branch_label] = get_distinct_color(branch_label)
            # Increment the branch label for the next branch.
            branch_label += 1
            if debug_plots:
                # Create an empty RGB overlay for visualizing the labeled branch.
                overlay = np.zeros((*skeleton.shape, 3), dtype=float)
                # Display the skeleton in gray by assigning equal intensity to all three channels.
                overlay[..., 0] = skeleton.astype(float) * 0.5
                overlay[..., 1] = skeleton.astype(float) * 0.5
                overlay[..., 2] = skeleton.astype(float) * 0.5
                # Highlight the current branch in red.
                overlay[branch, 0] = 1.0
                overlay[branch, 1] = 0.0
                overlay[branch, 2] = 0.0
                # Display the branch overlay.
                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.title(f"Branch {branch_label} Overlaid on Skeleton — Region {region.label}")
                plt.axis('off')
                plt.show()
        else:
            # Traverse the skeleton starting from each endpoint or branchpoint.
            for y, x in visit_coords:
                # Skip starting points that are not skeleton pixels or have already been visited unless they are branchpoints.
                if not skeleton[y, x] or (visited[y, x] and not branchpoints[y, x]): continue
                # Initialize an empty mask for the branch currently being traversed.
                branch = np.zeros_like(skeleton, dtype=bool)
                # Initialize the traversal queue with the starting coordinate and no previous direction.
                queue = [((y, x), None)]  # Store previous direction as None initially

                # Continue traversing until no eligible neighboring skeleton pixels remain.
                while queue:
                    # Retrieve the current pixel coordinate and the direction used to reach it.
                    (cy, cx), prev_dir = queue.pop()
                    # Skip the pixel if it is not part of the skeleton or has already been visited.
                    if not skeleton[cy, cx] or visited[cy, cx]: continue
                    # Mark the current skeleton pixel as visited and part of the current branch.
                    visited[cy, cx] = True
                    branch[cy, cx] = True
                    # Initialize a list for valid neighboring skeleton pixels.
                    neighbors = []
                    # Define directions that permit movement to the right.
                    allowed_dirs = [(dy, 1) for dy in [-1, 0, 1]]  # rightward steps
                    # Define vertical directions that may be used under the direction-change rule.
                    exception_dirs = [(0, 1), (0, -1)]  # up/down columns

                    # Examine the allowed neighboring directions around the current skeleton pixel.
                    for dy, dx in allowed_dirs + exception_dirs:
                        # Calculate the neighboring pixel coordinates.
                        ny, nx = cy + dy, cx + dx
                        # Ensure the neighboring pixel is inside the skeleton array.
                        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                            # Ignore neighbors that are not skeleton pixels or have already been visited.
                            if not skeleton[ny, nx] or visited[ny, nx]:
                                continue

                            # Store the direction from the current pixel to the neighboring pixel.
                            current_dir = (dy, dx)

                            # Allow if it's a rightward move or a change in vertical direction
                            if dx == 1:  # Rightward is always allowed
                                # Add rightward neighbors to the traversal queue.
                                neighbors.append(((ny, nx), current_dir))
                            elif dx == 0 and prev_dir != current_dir:  # vertical move allowed if not repeated
                                # Allow a vertical step only when its direction differs from the previous step.
                                neighbors.append(((ny, nx), current_dir))
                                
                            # At a branchpoint, prioritize the neighboring branch with the smallest row coordinate.
                            if branchpoints[cy, cx] and len(neighbors) > 1:
                                neighbors.sort(key=lambda p: p[0][0])  # sort by y value
                                queue.append(neighbors[0])
                            else:
                                # Otherwise, add all eligible neighbors to the traversal queue.
                                queue.extend(neighbors)

                # Display the current branch over the skeleton when debugging is enabled.
                if debug_plots:
                    # Create an empty RGB overlay for visualizing the branch.
                    overlay = np.zeros((*skeleton.shape, 3), dtype=float)
                    # Display the skeleton in gray.
                    overlay[..., 0] = skeleton.astype(float) * 0.5
                    overlay[..., 1] = skeleton.astype(float) * 0.5
                    overlay[..., 2] = skeleton.astype(float) * 0.5
                    # Highlight the current branch in red.
                    overlay[branch, 0] = 1.0
                    overlay[branch, 1] = 0.0
                    overlay[branch, 2] = 0.0
                    # Display the branch overlaid on the skeleton.
                    plt.figure(figsize=(6, 6))
                    plt.imshow(overlay)
                    plt.title(f"Branch {branch_label} Overlaid on Skeleton — Region {region.label}")
                    plt.axis('off')
                    plt.show()

                # Retain the branch only if it meets both the region-specific and global minimum length requirements.
                if np.count_nonzero(branch) >= max(min_branch_len, global_min_branch_length):# and np.count_nonzero(branch) != 0:
                    if debug_plots:
                        print(f'Add branch {branch_label}')
                    # Store the valid branch using its unique integer label.
                    skeletons_mask[branch] = branch_label
                    # Generate and store a distinct display color for the branch.
                    branch_color_lookup[branch_label] = get_distinct_color(branch_label)
                    # Increment the branch label counter.
                    branch_label += 1
                # Report branches that fail the region-specific threshold but exceed the global threshold.
                elif np.count_nonzero(branch) < min_branch_len and np.count_nonzero(branch) > global_min_branch_length and debug_plots:
                    print(f'Branch {branch_label} of length {np.count_nonzero(branch)} smaller than {min_branch_len}')
                # Report branches that fail the global threshold but exceed the region-specific threshold.
                elif np.count_nonzero(branch) < global_min_branch_length and np.count_nonzero(branch) > min_branch_len and debug_plots:
                    print(f'Branch {branch_label} of length {np.count_nonzero(branch)} smaller than global min {global_min_branch_length}')
                # Report branches that fail both minimum length requirements.
                else:
                    if debug_plots:
                        print(f'Branch {branch_label} of length {np.count_nonzero(branch)} smaller than global min {global_min_branch_length} and min {min_branch_len}')

                
        # Expand labeled branches using watershed
        # Calculate the distance from each region pixel to the nearest background pixel for watershed expansion.
        branch_distance = ndi.distance_transform_edt(region_mask)
        # Expand the labeled skeleton branches across the original region using watershed segmentation.
        filled_branch = segmentation.watershed(-branch_distance, markers=skeletons_mask, mask=region_mask)

        # Store each labeled branch mask
        # Iterate through every unique label generated by the watershed expansion.
        for label in np.unique(filled_branch):
            # Ignore the zero label, which represents background pixels.
            if label == 0: continue
            # Store each branch as a separate uint16 mask with its label value.
            particle_region_masks[label] = (filled_branch == label).astype(np.uint16) * label
            # Add the branch label to the final combined chain mask.
            chain_mask[filled_branch == label] = label

        # Show labeled regions on top of original
        if debug_plots:
            # Create an RGBA overlay image for displaying the labeled branches.
            overlay = np.zeros((*chain_mask.shape, 4), dtype=float)
            # Assign each labeled branch its previously generated display color.
            for label in np.unique(chain_mask):
                if label == 0: continue
                overlay[chain_mask == label] = branch_color_lookup[label]
            # Find the outer boundaries between neighboring labeled branches.
            boundaries = segmentation.find_boundaries(chain_mask, mode='outer')
            # Draw the detected boundaries in opaque black.
            overlay[boundaries] = [0, 0, 0, 1]
            # Create side-by-side plots of the labeled branches and the corresponding original image.
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(overlay)
            ax1.set_title('Labeled Branches with Boundaries')
            ax2.imshow(cropped_array)
            ax2.set_title('Original Image')
            # Hide axis markings on both images.
            for ax in [ax1, ax2]: ax.axis('off')
            plt.show()

        # Increment the processed region counter.
        region_counter += 1

    # Get unique branch labels
    # Extract all unique integer labels from the final chain mask.
    unique_labels = np.unique(chain_mask)
    # Remove the zero background label.
    unique_labels = unique_labels[unique_labels != 0]
    # Count the number of distinct labeled branches.
    num_labels = len(unique_labels)

    # Generate N visually distinct colors using HSV space
    # Generate evenly spaced hue values for the number of identified branches.
    hues = np.linspace(0, 1, num_labels, endpoint=False)
    # Seed NumPy's random number generator using a randomly generated integer.
    np.random.seed(np.random.randint(0,100))  # Optional: fix randomness
    # Randomly shuffle the hues so adjacent labels are less likely to have similar colors.
    np.random.shuffle(hues)  # Shuffle to avoid nearby labels looking similar
    # Convert the HSV colors into RGB values using fixed saturation and brightness values.
    colors = hsv_to_rgb(np.stack([hues, np.ones_like(hues)*0.65, np.ones_like(hues)*0.95], axis=1))

    # Create a mapping from label to color
    # Associate each unique branch label with one of the generated RGBA colors.
    label_to_color = {label: np.append(colors[i], 1.0) for i, label in enumerate(unique_labels)}  # RGBA

    # Create the overlay image
    # Initialize an empty RGBA image with the same spatial dimensions as the final chain mask.
    overlay_img = np.zeros((*chain_mask.shape, 4), dtype=float)
    # Assign the corresponding RGBA color to each labeled branch.
    for label, rgba in label_to_color.items():
        overlay_img[chain_mask == label] = rgba

    # Add black boundaries
    # Identify the outer boundaries of the labeled regions.
    boundaries = segmentation.find_boundaries(chain_mask, mode='outer')
    # Draw the boundaries as opaque black pixels.
    overlay_img[boundaries] = [0, 0, 0, 1]

    # Display the result
    # Create side-by-side plots of the globally colorized branch labels and the original particle image.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # Display the final watershed-filled branch segmentation with unique colors and black boundaries.
    ax1.imshow(overlay_img)
    ax1.set_title('Watershed-filled branches (globally unique colors) with black outlines')
    ax1.axis('off')
    # Display the corresponding original particle image.
    ax2.imshow(cropped_array)
    ax2.set_title('Actual Particle')
    ax2.axis('off')
    # Display the final segmentation and original image.
    plt.show()

    # Return the number of processed regions, the combined labeled branch mask, and the individual branch masks.
    return region_counter, chain_mask, particle_region_masks

def get_nonzero_bounding_box(mask):

    '''Determine the smallest rectangular bounding box that contains all nonzero pixels in a 2D mask.
        Input:
            mask - 2D NumPy array containing a binary or numeric mask, where nonzero values represent the region of interest and zero values represent the background.
        Output:
            tuple - Four integer coordinates in the order (ymin, ymax, xmin, xmax), defining the smallest bounding box containing all nonzero pixels. The maximum coordinates are one pixel beyond the final nonzero row or column so that the returned values can be used directly for NumPy slicing.
            None - Returned if the input mask contains no nonzero pixels.
    '''
    # Find the row and column coordinates of every nonzero pixel in the mask.
    nonzero_coords = np.argwhere(mask)
    # Check whether the mask contains any nonzero pixels.
    if nonzero_coords.size == 0:
        # Return None when the mask is completely empty.
        return None  # or raise an error if you prefer

    # Separate the nonzero pixel coordinates into row (y) and column (x) coordinates.
    y_coords, x_coords = nonzero_coords[:, 0], nonzero_coords[:, 1]
    # Determine the minimum and maximum row coordinates, adding 1 to the maximum so it can be used as the exclusive endpoint of a NumPy slice.
    ymin, ymax = y_coords.min(), y_coords.max() + 1  # +1 for slicing
    # Determine the minimum and maximum column coordinates, adding 1 to the maximum so it can be used as the exclusive endpoint of a NumPy slice.
    xmin, xmax = x_coords.min(), x_coords.max() + 1

    # Return the bounding box coordinates in the order required for NumPy array slicing.
    return (ymin, ymax, xmin, xmax)

def identify_branches(region_mask, region_bbox, max_hole_size=20, min_region_size=200, branch_length_fraction=0.007,
                      prune=True, prune_branch_length=5, vertical_prune_length=4, global_min_branch_length=2):

    '''Identify and label individual branches within a binary region mask using hole filling, skeletonization,
        branch pruning, directional skeleton traversal, and watershed segmentation.
        Input:
            region_mask - 2D NumPy array containing a binary mask of a single region, where nonzero or True values
            represent the region of interest and zero or False values represent the background.
            region_bbox - Tuple containing the bounding box of the region in the format (ymin, ymax, xmin, xmax).
            max_hole_size - Integer specifying the maximum number of pixels in an enclosed hole that will be filled.
            min_region_size - Integer specifying the minimum number of pixels required for a region to undergo
            branch identification. Smaller regions are treated as single branches.
            branch_length_fraction - Float specifying the fraction of the total skeleton length used to determine
            the minimum acceptable branch length.
            prune - Boolean specifying whether short branches should be iteratively removed from the skeleton.
            prune_branch_length - Integer specifying the minimum absolute branch length allowed during pruning.
            vertical_prune_length - Integer specifying the minimum length of a vertical skeleton segment that will
            be removed during the vertical pruning step.
            global_min_branch_length - Integer specifying the minimum absolute branch length required for a branch
            to be retained in the final labeled mask.
        Output:
            chain_mask - 2D NumPy integer array with the same shape as region_mask, where each identified branch
            is assigned a unique positive integer label and the background is zero.
            region_bbox - The original bounding box tuple passed into the function, returned unchanged.
    '''
    # Initialize an empty integer mask that will store the final unique label assigned to each branch.
    chain_mask = np.zeros_like(region_mask)  # Final labeled output mask
    # Initialize the region counter used to track the current region.
    region_counter = 1
    # Initialize the branch label counter used to assign unique integer labels to branches.
    branch_label = 1
    # Fill enclosed holes smaller than the specified threshold before skeletonization.
    region_mask_holes_filled = fill_small_holes(region_mask, max_hole_size=max_hole_size)

    # If the region is smaller than the minimum allowed size, treat the entire region as a single branch.
    if np.count_nonzero(region_mask) < min_region_size:
        # Assign the current branch label to every pixel belonging to the region.
        chain_mask[region_mask] = branch_label
        # Increment the branch label counter for subsequent branches.
        branch_label += 1
        # Increment the region counter to indicate that this region has been processed.
        region_counter += 1
        # Return the labeled region and its original bounding box without further branch analysis.
        return chain_mask, region_bbox

    # Thin the filled binary region to produce a one-pixel-wide skeleton representing its structure.
    skeleton = morphology.thin(region_mask_holes_filled)
    # Delete the filled mask because it is no longer needed after skeletonization.
    del region_mask_holes_filled  # no longer needed

    # Count the number of nonzero pixels in the skeleton as an approximation of its total length.
    total_skel_length = np.count_nonzero(skeleton)
    # Calculate the minimum branch length as a fraction of the total skeleton length.
    min_branch_len = int(branch_length_fraction * total_skel_length)

    # Count the number of neighboring skeleton pixels around each pixel using a 3x3 convolution.
    neighbors = ndi.convolve(skeleton.astype(np.uint16), np.ones((3, 3)), mode='constant')
    # Identify skeleton pixels with exactly one neighboring skeleton pixel, corresponding to endpoints.
    endpoints = np.logical_and(skeleton, neighbors == 2)
    # Identify skeleton pixels with three or more neighboring skeleton pixels, corresponding to branchpoints.
    branchpoints = np.logical_and(skeleton, neighbors >= 4)
    # Store the row and column coordinates of all identified endpoints.
    endpoint_coords = np.argwhere(endpoints)
    # Store the row and column coordinates of all identified branchpoints.
    branchpoint_coords = np.argwhere(branchpoints)
    # Delete the full endpoint and neighbor masks because only their coordinate arrays are needed from now on.
    del endpoints, neighbors  # only coords are used from now on

    # Optionally remove short branches from the skeleton before branch labeling.
    if prune:
        # Continue pruning until an iteration produces no changes to the skeleton.
        skeleton_changed = True
        while skeleton_changed:
            # Assume that no branches will be removed during the current iteration.
            skeleton_changed = False
            # Create a working copy of the current skeleton that can be modified during pruning.
            pruned = skeleton.copy()
            # Track which skeleton pixels have already been visited during branch traversal.
            visited = np.zeros_like(skeleton, dtype=bool)

            # Examine each endpoint as a possible starting point for a short branch.
            for y0, x0 in endpoint_coords:
                # Skip endpoints that have already been visited or are no longer part of the skeleton.
                if visited[y0, x0] or not skeleton[y0, x0]:
                    continue
                # Initialize a list to store the coordinates belonging to the branch being examined.
                branch_coords = []
                # Initialize a stack containing the current endpoint for traversal.
                queue = [(y0, x0)]

                # Traverse the skeleton outward from the endpoint until a branchpoint or termination is reached.
                while queue:
                    # Remove the next pixel coordinate from the traversal stack.
                    y, x = queue.pop()
                    # Skip pixels that have already been visited or are no longer part of the skeleton.
                    if visited[y, x] or not skeleton[y, x]:
                        continue
                    # Stop traversing when a branchpoint other than the starting endpoint is reached.
                    if (y, x) != (y0, x0) and branchpoints[y, x]:
                        break
                    # Mark the current skeleton pixel as visited.
                    visited[y, x] = True
                    # Add the current pixel to the branch being evaluated.
                    branch_coords.append((y, x))
                    # Examine all neighboring pixels in the 3x3 neighborhood around the current pixel.
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            # Skip the current pixel itself.
                            if dy == dx == 0:
                                continue
                            # Calculate the coordinates of the neighboring pixel.
                            ny, nx = y + dy, x + dx
                            # Ensure that the neighboring pixel is within the skeleton image boundaries.
                            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                                # Add unvisited skeleton neighbors to the traversal stack.
                                if skeleton[ny, nx] and not visited[ny, nx]:
                                    queue.append((ny, nx))

                # Remove the branch if it is shorter than either the relative or absolute pruning threshold.
                if len(branch_coords) < min_branch_len or len(branch_coords) < prune_branch_length:
                    # Set every pixel belonging to the short branch to zero in the working skeleton.
                    for y, x in branch_coords:
                        pruned[y, x] = 0
                    # Indicate that the skeleton changed and another pruning iteration is required.
                    skeleton_changed = True

            # Reconnect and re-thin the remaining skeleton after removing short branches.
            skeleton = morphology.thin(morphology.dilation(pruned, morphology.square(3)))
            # Delete temporary arrays that are no longer needed.
            del pruned, visited  # no longer needed

            # Recalculate the number of neighboring skeleton pixels after pruning.
            neighbors = ndi.convolve(skeleton.astype(np.uint16), np.ones((3, 3)), mode='constant')
            # Recalculate the skeleton endpoints after pruning.
            endpoints = np.logical_and(skeleton, neighbors == 2)
            # Store the coordinates of the newly identified endpoints.
            endpoint_coords = np.argwhere(endpoints)
            # Delete the temporary neighbor array.
            del neighbors

        # Create a boolean mask identifying vertical skeleton segments that should be removed.
        vertical_mask = np.zeros_like(skeleton, dtype=bool)
        # Examine each column of the skeleton independently.
        for x in range(skeleton.shape[1]):
            # Extract the current column of the skeleton.
            col = skeleton[:, x]
            # Track whether the current scan is inside a continuous vertical skeleton segment.
            in_segment = False
            # Initialize the starting row of the current vertical segment.
            start = 0

            # Scan one position beyond the bottom of the column so that segments ending at the image boundary are closed.
            for y in range(skeleton.shape[0] + 1):
                # Check whether the current position is within the image and contains a skeleton pixel.
                if y < skeleton.shape[0] and col[y]:
                    # Start a new vertical segment when entering a run of skeleton pixels.
                    if not in_segment:
                        in_segment = True
                        start = y
                else:
                    # Process the segment when a zero pixel or the end of the column is reached.
                    if in_segment:
                        # Record the row where the vertical segment ends.
                        end = y
                        # Mark vertical segments meeting the minimum length for removal.
                        if end - start >= vertical_prune_length:
                            vertical_mask[start:end, x] = True
                        # Reset the segment-tracking flag.
                        in_segment = False
        # Remove the temporary column variable.
        del col

        # Remove all vertical skeleton segments identified by the vertical pruning mask.
        skeleton[vertical_mask] = 0
        # Delete the vertical pruning mask because it is no longer needed.
        del vertical_mask  # mask no longer needed

        # Recalculate the number of neighboring skeleton pixels after vertical pruning.
        neighbors = ndi.convolve(skeleton.astype(np.uint16), np.ones((3, 3)), mode='constant')
        # Recalculate the skeleton endpoints after vertical pruning.
        endpoints = np.logical_and(skeleton, neighbors == 2)
        # Identify any new branchpoints created by the pruning operation.
        new_branchpoints = np.logical_and(skeleton, neighbors >= 4)
        # Store the coordinates of the updated endpoints.
        endpoint_coords = np.argwhere(endpoints)
        # Retrieve the coordinates currently stored in the original branchpoint mask.
        new_branchpoint_coords = np.argwhere(branchpoints)
        # Delete temporary arrays for which only the coordinate information is needed.
        del neighbors, endpoints  # only coords needed

        # Add newly detected branchpoints to the existing branchpoint mask.
        branchpoints |= new_branchpoints
        # Convert the existing branchpoint coordinates into a set for efficient comparison.
        existing_coords_set = set(map(tuple, branchpoint_coords))
        # Convert the updated branchpoint coordinates into a set for efficient comparison.
        new_coords_set = set(map(tuple, new_branchpoint_coords))
        # Delete the temporary arrays containing branchpoint masks and coordinates.
        del new_branchpoints, new_branchpoint_coords

        # Identify branchpoints that were not already present in the original branchpoint coordinates.
        unique_new_coords = np.array(list(new_coords_set - existing_coords_set))
        # Add newly identified branchpoints to the existing coordinate array.
        if unique_new_coords.size > 0:
            branchpoint_coords = np.vstack([branchpoint_coords, unique_new_coords])
        # Delete temporary coordinate sets and arrays.
        del existing_coords_set, new_coords_set, unique_new_coords

    # Track which skeleton pixels have already been assigned to a branch during traversal.
    visited = np.zeros_like(skeleton, dtype=bool)
    # Combine endpoint and branchpoint coordinates into a single set of traversal starting points.
    visit_coords = np.vstack((endpoint_coords, branchpoint_coords))
    # Sort traversal starting points by their column coordinate.
    visit_coords = visit_coords[np.argsort(visit_coords[:, 1])]
    # Delete the separate endpoint and branchpoint coordinate arrays because they have been combined.
    del endpoint_coords, branchpoint_coords  # now merged

    # Initialize an integer mask that will store the skeleton pixels assigned to each branch.
    skeletons_mask = np.zeros_like(skeleton, dtype=np.int32)

    # If no endpoints or branchpoints remain, assign the entire skeleton to one branch.
    if len(visit_coords) == 0:
        # Assign the current branch label to all remaining skeleton pixels.
        skeletons_mask[skeleton] = branch_label
        # Increment the branch label for the next branch.
        branch_label += 1
    else:
        # Traverse the skeleton starting from each endpoint or branchpoint.
        for y, x in visit_coords:
            # Skip coordinates that are not part of the skeleton or have already been visited unless they are branchpoints.
            if not skeleton[y, x] or (visited[y, x] and not branchpoints[y, x]):
                continue
            # Initialize an empty mask for the branch currently being traversed.
            branch = np.zeros_like(skeleton, dtype=bool)
            # Initialize the traversal stack and store the previous direction for directional filtering.
            queue = [((y, x), None)]  # Store previous direction as None initially

            # Initialize a temporary list for neighboring skeleton pixels.
            neighbors = []
            # Continue traversing until all connected allowable branch pixels have been processed.
            while queue:
                # Remove the next pixel and its previous movement direction from the stack.
                (cy, cx), prev_dir = queue.pop()
                # Skip pixels that are not part of the skeleton or have already been visited.
                if not skeleton[cy, cx] or visited[cy, cx]:
                    continue
                # Mark the current pixel as visited so it is not assigned repeatedly.
                visited[cy, cx] = True
                # Add the current pixel to the branch mask.
                branch[cy, cx] = True
                # Reset the temporary neighbor list for the current pixel.
                neighbors = []
                # Define allowed diagonal or horizontal movements that progress to the right.
                allowed_dirs = [(dy, 1) for dy in [-1, 0, 1]]
                # Define vertical movement directions that can be used as exceptions.
                exception_dirs = [(0, 1), (0, -1)]

                # Examine the allowed neighboring directions around the current skeleton pixel.
                for dy, dx in allowed_dirs + exception_dirs:
                    # Calculate the coordinates of the candidate neighboring pixel.
                    ny, nx = cy + dy, cx + dx
                    # Ensure the candidate pixel lies within the skeleton boundaries.
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        # Skip candidate pixels that are not skeleton pixels or have already been visited.
                        if not skeleton[ny, nx] or visited[ny, nx]:
                            continue
                        # Store the current movement direction.
                        current_dir = (dy, dx)
                        # Always permit movement to the right.
                        if dx == 1:
                            neighbors.append(((ny, nx), current_dir))
                        # Permit a vertical movement only when it changes from the previous vertical direction.
                        elif dx == 0 and prev_dir != current_dir:
                            neighbors.append(((ny, nx), current_dir))

                # At a branchpoint, choose the neighboring branch with the smallest row coordinate when multiple paths exist.
                if branchpoints[cy, cx] and len(neighbors) > 1:
                    neighbors.sort(key=lambda p: p[0][0])
                    queue.append(neighbors[0])
                else:
                    # Otherwise, continue traversing along all allowable neighboring pixels.
                    queue.extend(neighbors)

            # Retain the branch only if it meets both the relative and global minimum length requirements.
            if np.count_nonzero(branch) >= max(min_branch_len, global_min_branch_length):
                # Assign the current branch label to the branch skeleton pixels.
                skeletons_mask[branch] = branch_label
                # Increment the branch label for the next branch.
                branch_label += 1
            # Delete temporary arrays and lists associated with the completed branch traversal.
            del branch, queue, neighbors  # cleanup inner loop

    # Delete traversal arrays and the skeleton because they are no longer required after branch labeling.
    del visit_coords, visited, skeleton, branchpoints

    # Calculate the Euclidean distance from each pixel in the region to the nearest background pixel.
    branch_distance = ndi.distance_transform_edt(region_mask)
    # Use the labeled branch skeletons as watershed markers to expand each branch into the original region.
    filled_branch = segmentation.watershed(-branch_distance, markers=skeletons_mask, mask=region_mask)
    # Delete intermediate arrays that are no longer needed after watershed segmentation.
    del branch_distance, skeletons_mask

    # Iterate through each unique watershed label in the segmented region.
    for label in np.unique(filled_branch):
        # Skip label zero, which represents the background.
        if label == 0:
            continue
        # Assign each watershed region its corresponding branch label in the final output mask.
        chain_mask[filled_branch == label] = label
    # Delete the watershed result after transferring its labels to the final mask.
    del filled_branch

    # Increment the region counter after completing branch identification for the current region.
    region_counter += 1
    # Return the final labeled branch mask and the original region bounding box.
    return chain_mask, region_bbox

def label_mask_parallel(filtered_particle_mask, original_image, particle_bounds, disk_size=1, connectivity=1,
                 branch_length_fraction=0.05, global_min_branch_length=5, min_region_size=10, debug_plots=False,
                 prune=True, prune_branch_length=20, max_hole_size=10, vertical_prune_length = 3):

    '''Identify and label individual branches within a segmented particle mask using parallel processing, skeletonization,
        branch pruning, and watershed segmentation.
        Input:
            filtered_particle_mask - 2D NumPy array containing a binary mask of particles, where nonzero or True values
            represent particles or regions of interest and zero or False values represent the background.
            original_image - PIL Image or image-like object containing the original image corresponding to
            filtered_particle_mask. Used only for optional visualization of the processed regions.
            particle_bounds - Tuple containing the bounding box of the particle region in the format
            (ymin, ymax, xmin, xmax), where the first two values define the row range and the second two
            define the column range.
            disk_size - Integer specifying the radius of the disk-shaped structuring element used for
            morphological opening before watershed segmentation.
            connectivity - Integer specifying the pixel connectivity used when labeling connected regions.
            branch_length_fraction - Float specifying the fraction of the total skeleton length used to determine
            the minimum acceptable branch length.
            global_min_branch_length - Integer specifying the minimum absolute branch length required for a branch
            to be retained.
            min_region_size - Integer specifying the minimum number of pixels required for a region to undergo
            branch identification. Smaller regions are treated as single structures.
            debug_plots - Boolean specifying whether intermediate and final processing results should be displayed.
            prune - Boolean specifying whether short branches should be iteratively removed from the skeleton.
            prune_branch_length - Integer specifying the minimum absolute branch length allowed during pruning.
            max_hole_size - Integer specifying the maximum enclosed hole size that will be filled before skeletonization.
            vertical_prune_length - Integer specifying the minimum length of vertical skeleton segments that will be
            removed during the pruning process.
        Output:
            chain_mask - 2D NumPy array with the same spatial dimensions as the cropped particle mask, where each
            identified branch is assigned a unique positive integer label and the background is zero.
    '''
    # Unpack the particle bounding box into minimum and maximum row and column coordinates.
    ymin, ymax, xmin, xmax = particle_bounds
    # Delete the original bounding box variable because its individual coordinates are now stored separately.
    del particle_bounds

    # Extract the particle region from the full mask and convert it to a 16-bit unsigned integer array.
    particle = filtered_particle_mask[ymin:ymax, xmin:xmax].astype(np.uint16)
    #_global_particle = None

    # Clean the particle mask using morphological opening to remove small features and smooth boundaries.
    cleaned = morphology.binary_opening(particle, morphology.disk(disk_size))

    # Label connected regions in the cleaned particle mask using the specified connectivity.
    labeled_clean = measure.label(cleaned, connectivity=connectivity)
    # Delete the cleaned mask because only its labeled regions are needed for watershed segmentation.
    del cleaned

    # Calculate the Euclidean distance from each particle pixel to the nearest background pixel.
    distance = ndi.distance_transform_edt(particle)

    # Apply watershed segmentation to separate connected particle regions using the cleaned labels as markers.
    resegmented = segmentation.watershed(-distance, markers=labeled_clean, mask=particle)

    # Delete intermediate arrays that are no longer needed after watershed segmentation.
    del distance, labeled_clean

    # Convert the original image into a NumPy array for visualization and crop it to the particle bounding box.
    original_array = np.array(original_image)
    cropped_array = original_array[ymin:ymax, xmin:xmax]

    # Delete the full original image array because only the cropped region is needed for visualization.
    del original_array

    # Display the cropped binary mask, original image, and watershed segmentation when debugging is enabled.
    if debug_plots:
        # Create a figure showing the extracted particle mask and corresponding original image side-by-side.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Display the cropped binary particle mask.
        ax1.imshow(particle, cmap='gray')
        # Add a title describing the binary mask.
        ax1.set_title("Mask of Region")
        # Display the corresponding region from the original image.
        ax2.imshow(cropped_array)
        # Add a title describing the original image.
        ax2.set_title("Actual Particle")
        # Hide axis markings for both displayed images.
        for ax in [ax1, ax2]: ax.axis('off')
        # Display the figure.
        plt.show()

        # Show the result of watershed segmentation using different colors for each labeled region.
        plt.figure(figsize=(5, 5))
        # Convert the integer watershed labels into a color visualization.
        plt.imshow(color.label2rgb(resegmented, bg_label=0, kind='overlay'))
        # Add a title describing the watershed segmentation.
        plt.title("Watershed Cluster Segmentation")
        # Hide the axis markings.
        plt.axis('off')
        # Display the watershed segmentation figure.
        plt.show()

    # Determine the number of available CPU cores to use for parallel branch processing.
    num_workers = mp.cpu_count() #Determine the number of available cpu cores
    # Create a multiprocessing worker pool using all available CPU cores.
    pool = mp.Pool(processes=num_workers) #Instantiate the worker pool
    # Initialize a list for storing the asynchronous results from each worker.
    results = []
    # Initialize an empty mask with the same dimensions as the cropped particle mask for the final globally labeled branches.
    chain_mask = np.zeros_like(particle)

    # Process each region in the watershed-segmented particle mask.
    count = 0
    for region in measure.regionprops(resegmented):

        # Create a binary mask containing only the current watershed region.
        full_region_mask = resegmented == region.label
        # Determine the smallest bounding box containing the current region.
        region_bbox = get_nonzero_bounding_box(full_region_mask)
        # Crop the full region mask to its nonzero bounding box to reduce the amount of data sent to the worker process.
        region_mask = full_region_mask[region_bbox[0]:region_bbox[1],region_bbox[2]:region_bbox[3]]
        # Submit branch identification for the current region to the multiprocessing pool without blocking execution.
        result = pool.apply_async(identify_branches,(region_mask, region_bbox, max_hole_size, min_region_size,
                                                     branch_length_fraction, prune, prune_branch_length, vertical_prune_length,
                                                     global_min_branch_length))

        # Store the asynchronous result so it can be retrieved after all regions have been submitted.
        results.append(result) #append the results of each instance of the process pool
        # Increment the count of regions submitted for processing.
        count += 1

    # Retrieve the completed results from all multiprocessing workers.
    processed_results = [result.get() for result in results] #process the multiprocessing results
    # Initialize the offset used to ensure branch labels from different regions remain globally unique.
    keystart = 0

    # Loop through all processed region results and combine them into the global chain mask.
    for i in range(len(processed_results)): #loop through all processed results
        # Extract the locally labeled branch mask and its corresponding bounding box.
        particle_chain_mask, region_bbox = processed_results[i] #record outputs from each simulation

        # Identify all nonzero branch labels present in the current region.
        old_vals = np.unique(particle_chain_mask[particle_chain_mask != 0])
        # Shift the local branch labels by the current global offset so they do not overlap with labels from previous regions.
        new_vals = old_vals + keystart
        # Build a dictionary mapping each local branch label to its new globally unique label.
        relabel_dict = dict(zip(old_vals, new_vals))
        # Create a vectorized function that replaces local labels with their corresponding global labels.
        relabel_fn = np.vectorize(lambda x: relabel_dict.get(x, 0))
        # Apply the global relabeling to the current region mask.
        particle_chain_mask = relabel_fn(particle_chain_mask)
        # Insert the relabeled branches into the corresponding location of the global chain mask.
        chain_mask[region_bbox[0]:region_bbox[1],region_bbox[2]:region_bbox[3]][particle_chain_mask != 0] = particle_chain_mask[particle_chain_mask != 0]
        # Delete the current region mask because its labels have been transferred to the global mask.
        del particle_chain_mask
        # Update the global label offset by the number of branches found in the current region.
        keystart += len(old_vals)

    # Close the multiprocessing pool so that no additional tasks can be submitted.
    pool.close()
    # Wait for all worker processes to finish before continuing.
    pool.join()

    # Get the unique branch labels present in the final globally labeled mask.
    unique_labels = np.unique(chain_mask)
    # Remove the background label of zero.
    unique_labels = unique_labels[unique_labels != 0]
    # Count the total number of unique branches identified.
    num_labels = len(unique_labels)

    # Generate evenly spaced hue values for visually distinguishing the identified branches.
    hues = np.linspace(0, 1, num_labels, endpoint=False)
    # Set a random seed using a randomly generated integer to control the subsequent hue shuffling.
    np.random.seed(np.random.randint(0,100))  # Optional: fix randomness
    # Shuffle the hues so neighboring branch labels are less likely to have similar colors.
    np.random.shuffle(hues)  # Shuffle to avoid nearby labels looking similar
    # Convert the HSV hue values into RGB colors with fixed saturation and brightness.
    colors = hsv_to_rgb(np.stack([hues, np.ones_like(hues)*0.65, np.ones_like(hues)*0.95], axis=1))

    # Create a lookup dictionary mapping each branch label to an RGBA color.
    label_to_color = {label: np.append(colors[i], 1.0) for i, label in enumerate(unique_labels)}  # RGBA

    # Initialize a four-channel RGBA image for visualizing the labeled branches.
    overlay_img = np.zeros((*chain_mask.shape, 4), dtype=float)
    # Assign each branch its corresponding RGBA color in the visualization image.
    for label, rgba in label_to_color.items():
        overlay_img[chain_mask == label] = rgba

    # Identify the outer boundaries between different labeled branches.
    boundaries = segmentation.find_boundaries(chain_mask, mode='outer')
    # Set all branch boundaries to opaque black in the visualization.
    overlay_img[boundaries] = [0, 0, 0, 1]

    # Display the final globally labeled branch mask and corresponding original particle image when debugging is enabled.
    if debug_plots:
        # Create a figure showing the colored watershed-filled branches and original particle side-by-side.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        # Display the globally colored branch segmentation.
        ax1.imshow(overlay_img)
        # Add a title describing the labeled branch visualization.
        ax1.set_title('Watershed-filled branches (globally unique colors) with black outlines')
        # Hide the axis markings around the branch visualization.
        ax1.axis('off')
        # Display the corresponding original particle image.
        ax2.imshow(cropped_array)
        # Add a title describing the original particle.
        ax2.set_title('Actual Particle')
        # Hide the axis markings around the original image.
        ax2.axis('off')
        # Display the final comparison figure.
        plt.show()

    # Return the final globally labeled branch mask.
    return chain_mask

def relabel_region(region_mask, region_bbox, small_hole_size = 2, min_region_size=200, branch_length_fraction=0.025, prune=True,
                   prune_branch_length=0, vertical_prune_length=4, global_min_branch_length=2, vertical_prune_length2 = 3):

    '''Relabel the nonzero regions of a labeled particle mask using a specified label offset.
        Input:
            particle_chain_mask - 2D NumPy array containing a labeled particle mask, where zero values represent
            the background and positive integer values represent individual branches or structures.
            keystart - Integer offset added to each existing nonzero label to ensure that labels remain unique
            when combining multiple independently processed regions.
        Output:
            particle_chain_mask - 2D NumPy array containing the relabeled particle mask, where each nonzero
            branch label has been increased by the specified offset and the background remains zero.
    '''

    # Create an 8-connected neighborhood kernel by setting all neighboring
    # pixels to 1 while excluding the center pixel itself.
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # exclude the center pixel

    # Initialize the final mask that will contain uniquely labeled branches.
    labelled_chain_mask = np.zeros_like(region_mask, dtype=np.uint16)
    # Start assigning branch labels at 1; 0 is reserved for the background.
    branch_label = 1
    # Track the number of regions processed.
    region_counter = 1

    # If the region is smaller than the minimum allowed size, skip skeletonization
    # and simply assign the entire region a single branch label.
    if np.count_nonzero(region_mask) < min_region_size:
        labelled_chain_mask[region_mask] = branch_label
        branch_label += 1
        region_counter += 1
        return labelled_chain_mask, region_bbox

    # Skeletonize the filled region to obtain its medial axis.
    skeleton = morphology.thin(region_mask)
    # Preserve the original skeleton because the skeleton will later be modified
    # during pruning, while the original version is needed for watershed filling.
    original_skeleton = copy.deepcopy(skeleton)

    # Compute the total number of skeleton pixels.
    total_skel_length = np.count_nonzero(skeleton)

    # Compute minimum allowed branch length as a fraction of total skeleton length.
    min_branch_len = int(branch_length_fraction * total_skel_length)

    # Perform the first pass of vertical pruning if enabled.
    if vertical_prune_length != 0:
        # Create a mask identifying vertical skeleton segments that should be removed.
        vertical_mask = np.zeros_like(skeleton, dtype=bool)
        # Pad the skeleton so horizontal-neighbor checks can be performed safely
        # near the image boundaries.
        padded = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant')

        # Examine the skeleton one column at a time.
        for x in range(1, padded.shape[1] - 1):
            col = padded[:, x]
            in_segment = False
            start = 0
            segment_coords = []

            # Traverse each pixel in the current column.
            for y in range(1, padded.shape[0] - 1):
                if col[y]:
                    if not in_segment:
                        # Start tracking a new continuous vertical segment.
                        in_segment = True
                        segment_coords = [(y, x)]
                    else:
                        # Continue adding pixels to the current vertical segment.
                        segment_coords.append((y, x))
                else:
                    # End of a continuous segment.
                    if in_segment:
                        in_segment = False
                        segment_length = len(segment_coords)

                        # Only consider vertical segments at least as long as
                        # the specified pruning threshold.
                        if segment_length >= vertical_prune_length:
                            # Mark all for pruning except those with horizontal neighbors.
                            for (yy, xx) in segment_coords:
                                has_horizontal_neighbor = padded[yy, xx - 1] or padded[yy, xx + 1]
                                if not has_horizontal_neighbor:
                                    # Account for the one-pixel padding when
                                    # transferring coordinates back to the original mask.
                                    vertical_mask[yy - 1, xx - 1] = True

            # Handle bottom-edge continuation.
            # This catches a vertical segment that continues to the bottom
            # of the padded image without encountering a zero.
            if in_segment and len(segment_coords) >= vertical_prune_length:
                for (yy, xx) in segment_coords:
                    has_horizontal_neighbor = padded[yy, xx - 1] or padded[yy, xx + 1]
                    if not has_horizontal_neighbor:
                        vertical_mask[yy - 1, xx - 1] = True

        # Apply pruning by removing the pixels identified as isolated
        # vertical sections.
        skeleton[vertical_mask] = 0
        # Optional cleanup
        skeleton = binary_dilation(skeleton, np.ones((small_hole_size, small_hole_size)))
        skeleton = morphology.thin(skeleton)

    #print('vertical again')
    # Perform a second vertical-pruning pass using a potentially different threshold.
    if vertical_prune_length2 != 0:
        vertical_mask = np.zeros_like(skeleton, dtype=bool)
        padded = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant')

        # Again scan the skeleton one column at a time.
        for x in range(1, padded.shape[1] - 1):
            col = padded[:, x]
            in_segment = False
            start = 0
            segment_coords = []

            # Identify continuous vertical segments.
            for y in range(1, padded.shape[0] - 1):
                if col[y]:
                    if not in_segment:
                        in_segment = True
                        segment_coords = [(y, x)]
                    else:
                        segment_coords.append((y, x))
                else:
                    # End of a continuous segment.
                    if in_segment:
                        in_segment = False
                        segment_length = len(segment_coords)

                        # Mark sufficiently long vertical segments for removal,
                        # except for pixels that connect horizontally to another
                        # portion of the skeleton.
                        if segment_length >= vertical_prune_length2:
                            # Mark all for pruning except those with horizontal neighbors.
                            for (yy, xx) in segment_coords:
                                has_horizontal_neighbor = padded[yy, xx - 1] or padded[yy, xx + 1]
                                if not has_horizontal_neighbor:
                                    vertical_mask[yy - 1, xx - 1] = True

            # Handle vertical segments that continue to the bottom edge.
            if in_segment and len(segment_coords) >= vertical_prune_length2:
                for (yy, xx) in segment_coords:
                    has_horizontal_neighbor = padded[yy, xx - 1] or padded[yy, xx + 1]
                    if not has_horizontal_neighbor:
                        vertical_mask[yy - 1, xx - 1] = True

        # Remove the identified vertical sections.
        skeleton[vertical_mask] = 0
        # Dilate slightly to reconnect nearby skeleton pixels after pruning.
        # Then thin again to restore a one-pixel-wide skeleton.
        skeleton = binary_dilation(skeleton, np.ones((small_hole_size, small_hole_size)))
        skeleton = morphology.thin(skeleton)

    # Count the number of neighboring skeleton pixels around each skeleton pixel.
    # The center pixel is excluded by the kernel defined above.
    neighbor_count = ndi.convolve(skeleton.astype(np.uint8), kernel, mode='constant')

    # Endpoints have exactly one neighboring skeleton pixel.
    endpoints = np.logical_and(skeleton, neighbor_count == 1)
    endpoint_coords = np.argwhere(endpoints)

    # Branchpoints have three or more neighboring skeleton pixels.
    branchpoints = np.logical_and(skeleton, neighbor_count >= 3)
    # Group adjacent branchpoint pixels into connected clusters.
    labeled_bp, num_bp = ndi.label(branchpoints)
    # Keep only one representative pixel per cluster (the centroid or first pixel).
    filtered_branchpoints = np.zeros_like(branchpoints)
    for i in range(1, num_bp + 1):
        coords = np.argwhere(labeled_bp == i)
        # pick the central one (median) or just the first one
        y, x = np.median(coords, axis=0).astype(int)
        filtered_branchpoints[y, x] = True
    branchpoints = filtered_branchpoints

    # These intermediate arrays are no longer needed after their coordinates
    # have been extracted.
    del neighbor_count, endpoints, labeled_bp, num_bp, filtered_branchpoints

    # If pruning is enabled, remove short or unwanted branches.
    if prune:
        skeleton_changed = True

        # Continue pruning until an iteration produces no further changes.
        while skeleton_changed:
            skeleton_changed = False

            # Copy current skeleton so branches can be removed without
            # modifying the skeleton being traversed.
            pruned = skeleton.copy()

            # Track skeleton pixels that have already been visited during traversal.
            visited = np.zeros_like(skeleton, dtype=bool)

            # Iterate over all endpoint coordinates.
            for y0, x0 in endpoint_coords:
                # Skip endpoints that were already processed or removed.
                if visited[y0, x0] or not skeleton[y0, x0]:
                    continue

                # Store the coordinates belonging to the branch being followed.
                branch_coords = []

                # Queue for depth-first traversal.
                queue = [(y0, x0)]

                # Perform traversal starting from the endpoint.
                while queue:
                    y, x = queue.pop()

                    # Skip already visited or non-skeleton pixels.
                    if visited[y, x] or not skeleton[y, x]:
                        continue

                    # Stop following this branch when a branchpoint is reached.
                    if (y, x) != (y0, x0) and branchpoints[y, x]:
                        break

                    # Mark pixel as visited and record it as part of the branch.
                    visited[y, x] = True
                    branch_coords.append((y, x))

                    # Explore the full 8-connected neighborhood.
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == dx == 0:
                                continue
                            ny, nx = y + dy, x + dx

                            # Check that the neighboring pixel lies inside the mask.
                            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                                if skeleton[ny, nx] and not visited[ny, nx]:
                                    queue.append((ny, nx))

                # Remove branch if shorter than either the fractional minimum
                # branch length or the absolute pruning threshold.
                if len(branch_coords) < min_branch_len or len(branch_coords) < prune_branch_length:
                    for y, x in branch_coords:
                        pruned[y, x] = 0
                    skeleton_changed = True
                
                

            #Re-thin after pruning to clean up artifacts.
            skeleton = pruned

            # Recompute neighbors and endpoints after pruning.
            neighbor_count = ndi.convolve(skeleton.astype(np.uint8), kernel, mode='constant')

            endpoints = np.logical_and(skeleton, neighbor_count == 1)
            endpoint_coords = np.argwhere(endpoints)

            # Recalculate branchpoints because pruning may have changed
            # the connectivity of the skeleton.
            branchpoints = np.logical_and(skeleton, neighbor_count >= 3)
            labeled_bp, num_bp = ndi.label(branchpoints)

            # Keep only one representative pixel for each connected
            # branchpoint cluster.
            filtered_branchpoints = np.zeros_like(branchpoints)
            for i in range(1, num_bp + 1):
                coords = np.argwhere(labeled_bp == i)
                # pick the central one (median) or just the first one
                y, x = np.median(coords, axis=0).astype(int)
                filtered_branchpoints[y, x] = True
            branchpoints = filtered_branchpoints
            branchpoint_coords = np.argwhere(branchpoints)

        # Free temporary arrays from the pruning process.
        del pruned, visited, neighbor_count, endpoints, labeled_bp, num_bp, filtered_branchpoints, queue

    # Initialize visit tracker for skeleton traversal.
    visited = np.zeros_like(skeleton, dtype=bool)

    # Combine endpoints and branchpoints into one list for traversal.
    visit_coords = endpoint_coords

    # Sort by x-coordinate to prioritize rightmost traversal.
    visit_coords = visit_coords[np.argsort(visit_coords[:, 1])[::-1]]

    # Free separated coordinate arrays.
    del endpoint_coords

    # Initialize integer mask to hold labeled skeleton branches.
    skeletons_mask = np.zeros_like(skeleton, dtype=np.int32)

    # If there are no branchpoints or endpoints, label entire skeleton as one branch.
    if len(visit_coords) == 0:
        skeletons_mask[skeleton] = branch_label
        branch_label += 1
    else:
        # Traverse each endpoint or branchpoint to label distinct branches.
        for y, x in visit_coords:
            # Ignore pixels that are not part of the skeleton or have already
            # been assigned to another branch.
            if not skeleton[y, x] or (visited[y, x] and not branchpoints[y, x]):
                continue

            # Initialize a blank branch mask and queue for traversal.
            branch = np.zeros_like(skeleton, dtype=bool)
            queue = [((y, x), None)]

            neighbors = []
            while queue:
                (cy, cx), prev_dir = queue.pop()

                # Skip non-skeleton pixels or pixels already assigned to a branch.
                if not skeleton[cy, cx] or visited[cy, cx]:
                    continue

                visited[cy, cx] = True
                branch[cy, cx] = True
                neighbors = []

                # Define possible traversal directions.
                # Rightward movement can include vertical offsets, allowing
                # the branch to follow slightly angled structures.
                allowed_dirs = [(dy, 1) for dy in [-1, 0, 1]]
                # Horizontal movement to the left/right is handled separately.
                exception_dirs = [(0, 1), (0, -1)]

                # Explore neighbor pixels based on directionality.
                for dy, dx in allowed_dirs + exception_dirs:
                    ny, nx = cy + dy, cx + dx

                    # Ignore coordinates outside the skeleton.
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if not skeleton[ny, nx] or visited[ny, nx]:
                            continue

                        current_dir = (dy, dx)

                        # Rightward movement is always allowed.
                        if dx == 1:
                            neighbors.append(((ny, nx), current_dir))
                        # Vertical movement is allowed only when the direction
                        # changes, preventing repeated vertical backtracking.
                        elif dx == 0 and prev_dir != current_dir:
                            neighbors.append(((ny, nx), current_dir))

                # If multiple neighbors are encountered at a branchpoint,
                # prioritize the path with the smallest y-coordinate.
                if branchpoints[cy, cx] and len(neighbors) > 1:
                    neighbors.sort(key=lambda p: p[0][0])
                    queue.append(neighbors[0])
                else:
                    queue.extend(neighbors)

            # Label branch if its length exceeds the required thresholds.
            if np.count_nonzero(branch) >= max(min_branch_len, global_min_branch_length):
                skeletons_mask[branch] = branch_label
                branch_label += 1
                
        # Find branchpoints that were not reached during the endpoint traversal.
        remaining_branchpoints = [(y, x) for y, x in branchpoint_coords if skeleton[y, x] and not visited[y, x]]

        # Traverse these remaining branchpoints separately.
        for y, x in remaining_branchpoints:
            if not skeleton[y, x] or (visited[y, x]):
                #print('ignore coordinate: ', (y,x))
                continue

            # Initialize a blank branch mask and queue for traversal.
            branch = np.zeros_like(skeleton, dtype=bool)
            queue = [((y, x), None)]

            neighbors = []
            while queue:
                (cy, cx), prev_dir = queue.pop()

                # Skip non-skeleton pixels or pixels already assigned to a branch.
                if not skeleton[cy, cx] or visited[cy, cx]:
                    continue

                visited[cy, cx] = True
                branch[cy, cx] = True
                neighbors = []

                # Define possible traversal directions.
                allowed_dirs = [(dy, 1) for dy in [-1, 0, 1]]
                exception_dirs = [(0, 1), (0, -1)]

                # Explore neighbor pixels based on directionality.
                for dy, dx in allowed_dirs + exception_dirs:
                    ny, nx = cy + dy, cx + dx

                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if not skeleton[ny, nx] or visited[ny, nx]:
                            continue

                        current_dir = (dy, dx)

                        if dx == 1:
                            neighbors.append(((ny, nx), current_dir))
                        elif dx == 0 and prev_dir != current_dir:
                            neighbors.append(((ny, nx), current_dir))

                # If multiple neighbors are encountered at a branchpoint,
                # prioritize the path with the smallest y-coordinate.
                if branchpoints[cy, cx] and len(neighbors) > 1:
                    neighbors.sort(key=lambda p: p[0][0])
                    queue.append(neighbors[0])
                else:
                    queue.extend(neighbors)

            # Label branch if its length exceeds the required thresholds.
            if np.count_nonzero(branch) >= max(min_branch_len, global_min_branch_length):
                skeletons_mask[branch] = branch_label
                branch_label += 1

                #show_branch(skeleton, branch, endpoint_coords, branchpoint_coords, title=f"Branch {branch_label-1}")

        # Free skeleton arrays no longer needed.
        del visit_coords, visited, skeleton, branchpoints, branchpoint_coords, remaining_branchpoints, branch, queue, neighbors

    #skeleton = morphology.thin(morphology.dilation(skeleton, morphology.square(2)))

    # Thicken the original skeleton before watershed segmentation.
    # This provides a larger set of pixels that can be assigned to the
    # labeled branch seeds.
    thick_skeleton = morphology.dilation(original_skeleton, morphology.square(2))

    # Calculate the distance from each pixel to the nearest background pixel.
    branch_distance = ndi.distance_transform_edt(thick_skeleton)

    # Use the labeled skeleton branches as markers to divide the thickened
    # skeleton into separate branch regions.
    filled_skeleton = segmentation.watershed(-branch_distance, markers=skeletons_mask, mask=thick_skeleton)

    # Calculate the distance transform of the complete original region.
    region_distance = ndi.distance_transform_edt(region_mask)

    # Expand the skeleton-based labels across the original region using
    # watershed segmentation. This assigns every part of the region to
    # the nearest labeled branch.
    filled_branch = segmentation.watershed(-region_distance, markers=filled_skeleton,mask=region_mask)

    # Free distance and skeleton label maps.
    del branch_distance, skeletons_mask, thick_skeleton, original_skeleton, filled_skeleton, region_distance

    # Transfer watershed results into the final labeled mask.
    for label in np.unique(filled_branch):
        if label == 0:
            continue
        labelled_chain_mask[filled_branch == label] = label
    del filled_branch

    # Increment region counter.
    region_counter += 1

    return labelled_chain_mask, region_bbox

def relabel_mask_parallel(chain_mask, small_hole_size = 2, min_region_size=200, branch_length_fraction=0.025, prune=True,
                          prune_branch_length=0, vertical_prune_length=4, global_min_branch_length=2, vertical_prune_length2 = 3, debug_plots = False):
    '''
    Relabels the regions in a labeled particle mask by processing each particle
    independently in parallel and then combining the relabeled regions into a
    single global mask with unique labels.

    Inputs:
        chain_mask (numpy.ndarray): A 2D labeled mask in which background pixels
            are represented by 0 and each particle/region is represented by a
            unique integer label.
        small_hole_size (int): Maximum size of holes to fill during region
            processing.
        min_region_size (int): Minimum region size used during region
            processing.
        branch_length_fraction (float): Fraction of the region size used to
            determine the branch length for skeleton pruning.
        prune (bool): Determines whether skeleton branches should be pruned.
        prune_branch_length (int or float): Minimum branch length used for
            pruning.
        vertical_prune_length (int or float): Length threshold used for
            vertical branch pruning.
        global_min_branch_length (int or float): Global minimum branch length
            used during branch processing.
        vertical_prune_length2 (int or float): Second length threshold used
            for vertical branch pruning.
        debug_plots (bool): Determines whether debugging plots are generated
            during region processing.

    Outputs:
        chain_mask (numpy.ndarray): A 2D labeled mask in which the processed
            regions have been relabeled so that each resulting particle chain
            has a unique integer label across the entire image.
    '''

    num_workers = mp.cpu_count() #Determine the number of available cpu cores
    pool = mp.Pool(processes=num_workers) #Instantiate the worker pool
    results = [] #Create an empty list to store the asynchronous processing results

    # Process each region in the resegmented mask
    count = 0 #Initialize a counter for the number of regions being processed
    for region in np.unique(chain_mask)[1:]: #Loop through each unique non-background region label in the mask

        full_region_mask = chain_mask == region #Create a binary mask containing only the current region
        region_bbox = get_nonzero_bounding_box(full_region_mask) #Determine the bounding box containing the current region
        region_mask = full_region_mask[region_bbox[0]:region_bbox[1],region_bbox[2]:region_bbox[3]] #Crop the binary region mask to its bounding box
        result = pool.apply_async(relabel_region,(region_mask, region_bbox, small_hole_size, min_region_size, branch_length_fraction, #Submit the current region to the multiprocessing pool for asynchronous processing
                                                prune, prune_branch_length, vertical_prune_length, global_min_branch_length, vertical_prune_length2)) #Pass the remaining processing parameters to relabel_region


        results.append(result) #append the results of each instance of the process pool
        count += 1 #Increment the number of regions submitted for processing

    processed_results = [result.get() for result in results] #process the multiprocessing results
    keystart = 0 #Initialize the offset used to ensure labels remain unique between regions

    for i in range(len(processed_results)): #loop through all processed results
        particle_chain_mask, region_bbox = processed_results[i] #record outputs from each simulation

        old_vals = np.unique(particle_chain_mask[particle_chain_mask != 0]) #Find all unique nonzero labels produced for the current region
        new_vals = old_vals + keystart #Shift the labels by the current global label offset
        # Build a lookup dictionary
        relabel_dict = dict(zip(old_vals, new_vals)) #Create a dictionary mapping each original label to its new globally unique label
        # Vectorized relabeling using np.vectorize
        relabel_fn = np.vectorize(lambda x: relabel_dict.get(x, 0)) #Create a vectorized function that replaces each label using the lookup dictionary and assigns 0 to unmapped values
        particle_chain_mask = relabel_fn(particle_chain_mask) #Apply the relabeling function to the entire processed region mask
        # Combine into global mask
        chain_mask[region_bbox[0]:region_bbox[1],region_bbox[2]:region_bbox[3]][particle_chain_mask != 0] = particle_chain_mask[particle_chain_mask != 0] #Insert the relabeled nonzero pixels into their corresponding location in the global mask
        del particle_chain_mask #Delete the temporary processed mask to free memory
        keystart += len(old_vals) #Advance the label offset by the number of labels generated in the current region

    pool.close() #Close the multiprocessing pool so no additional tasks can be submitted
    pool.join() #Wait for all worker processes to finish and terminate cleanly

    return chain_mask #Return the globally relabeled chain mask

def relabel_disconnected_regions(mask):
    """
    Relabels disconnected regions with unique labels, even if they originally shared a label.

    Parameters:
        mask (np.ndarray): 2D labeled mask where each integer represents a region.

    Returns:
        new_mask (np.ndarray): 2D mask with uniquely labeled connected components.
    """

    # Initialize a new mask of zeros (same shape as input), dtype int32 to hold new labels
    new_mask = np.zeros_like(mask, dtype=np.int32)

    # Counter for assigning new unique labels
    current_label = 1

    # Get all unique labels in the input mask
    unique_labels = np.unique(mask)
    
    # Loop through each unique label
    for lbl in unique_labels:
        if lbl == 0:
            # Skip background (assumed to be 0)
            continue

        # Optional progress print every 500 labels
        if lbl % 10 == 0:
            print(f"{lbl/len(unique_labels)}%")

        # Find coordinates of all pixels belonging to the current label
        coords = np.argwhere(mask == lbl)
        if coords.size == 0:
            # Skip if no pixels found for this label (just in case)
            continue

        # Determine the bounding box for the current label to minimize processing
        min_row, min_col = coords.min(axis=0)      # top-left corner
        max_row, max_col = coords.max(axis=0) + 1  # bottom-right corner (+1 because slicing is exclusive)

        # Create a slice object for extracting this region
        region_slice = (slice(min_row, max_row), slice(min_col, max_col))

        # Make a boolean mask for this region (True where pixels equal current label)
        region_mask = (mask[region_slice] == lbl)

        # Label connected components within this region
        labeled_region, num_features = label(region_mask)
        # labeled_region has values 0 (background) and 1..num_features for connected components

        # Assign each connected component a new unique label in the new_mask
        for i in range(1, num_features + 1):
            # Set pixels in new_mask corresponding to this component to current_label
            new_mask[region_slice][labeled_region == i] = current_label
            # Increment the label counter for next component
            current_label += 1

    # Return the fully relabeled mask
    return new_mask

def box_counting_fractal_dim(points, box_sizes=[2, 4, 8, 16, 32]):
    '''Calculates the fractal dimension of a set of points using the box-counting
    method. The points are analyzed at multiple box sizes, and the slope of
    the log-log relationship between box size and the number of occupied boxes
    is used as the estimated fractal dimension.

    Inputs:
        points (numpy.ndarray): A 2D array of point coordinates with shape
            (N, D), where N is the number of points and D is the number of
            spatial dimensions.
        box_sizes (list of int or float): A list of box side lengths at which
            the point distribution is evaluated.

    Outputs:
        float: The estimated box-counting fractal dimension of the point
            distribution. Returns np.nan if fewer than two points are
            provided or if fewer than two valid box sizes are available.
    '''

    if len(points) < 2: #Check whether there are enough points to calculate a fractal dimension
        return np.nan #Return NaN because a fractal dimension cannot be calculated from fewer than two points

    min_vals = points.min(axis=0) #Calculate the minimum coordinate value along each spatial dimension
    max_vals = points.max(axis=0) #Calculate the maximum coordinate value along each spatial dimension
    side_lengths = max_vals - min_vals #Calculate the total extent of the point distribution along each spatial dimension

    counts = [] #Initialize a list to store the number of occupied boxes at each valid scale
    valid_sizes = [] #Initialize a list to store the box sizes that can be used for the calculation

    for size in box_sizes: #Loop through each specified box size
        bins = np.ceil(side_lengths / size).astype(int) #Calculate the number of histogram bins required along each dimension for the current box size

        # Skip if any bin is less than 1 (too small for histogramdd)
        if np.any(bins < 1): #Check whether any dimension would have fewer than one histogram bin
            continue #Skip this box size if it cannot produce a valid histogram

        hist = np.histogramdd(points, bins=bins)[0] #Create a multidimensional histogram that counts points within each box
        counts.append(np.sum(hist > 0)) #Count and store the number of boxes containing at least one point
        valid_sizes.append(size) #Store the current box size because it produced a valid measurement

    # If too few valid scales, return NaN
    if len(counts) < 2: #Check whether at least two valid box sizes were available for the linear fit
        return np.nan #Return NaN because a slope cannot be reliably calculated from fewer than two scales

    logsizes = np.log(1 / np.array(valid_sizes)) #Calculate the logarithm of the inverse box sizes for the log-log analysis
    logcounts = np.log(counts) #Calculate the logarithm of the number of occupied boxes at each scale

    # Linear fit to get slope = fractal dimension
    slope, _ = np.polyfit(logsizes, logcounts, 1) #Fit a linear relationship between log inverse box size and log occupied-box count and extract the slope
    return slope #Return the fitted slope as the estimated fractal dimension

def analyze_clusters(labeled_mask, pixel_micron, angle_offset = None, fixed_angle = None, plot=False, print_statement=False):

    '''Analyzes the geometry, orientation, width, and fractal dimension of each
    labeled cluster in a 2D binary or labeled mask. Each cluster is analyzed
    independently using principal component analysis (PCA) to determine its
    major axis, minor-axis scatter, orientation, and maximum width. The
    function optionally plots and prints the calculated properties, converts
    pixel-based measurements into physical units, and returns the results as
    a pandas DataFrame.

    Inputs:
        labeled_mask (numpy.ndarray): A 2D labeled mask in which background
            pixels are represented by 0 and each cluster is represented by a
            unique positive integer label.
        pixel_micron (int or float): The physical size of one pixel in
            micrometers. Used to convert pixel-based lengths, widths, and areas
            into physical units.
        angle_offset (int, float, or None): Optional angular offset in degrees
            to apply to the calculated cluster orientations. Defaults to None.
        fixed_angle (int, float, or None): Optional fixed reference angle in
            degrees. If provided, cluster orientations are measured relative to
            this angle instead of the area-weighted average cluster orientation.
            Defaults to None.
        plot (bool): Determines whether a plot showing each cluster's points,
            major axis, minor-axis scatter, and center is generated. Defaults
            to False.
        print_statement (bool): Determines whether calculated properties for
            each cluster are printed to the console. Defaults to False.

    Outputs:
        pandas.DataFrame: A DataFrame containing one row for each analyzed
            cluster. The DataFrame includes the cluster label, physical area,
            major-axis length, minor-axis scatter, orientation, maximum width,
            fractal dimension, total number of clusters, fractional area,
            scatter ratio, relative orientation, and order parameter.
    '''

    num_features = len(np.unique(labeled_mask)) #Determine the total number of unique labels in the mask, including the background label
    label_list = list(np.unique(labeled_mask))[1:] #Create a list of cluster labels while excluding the first label, which is assumed to be the background label
    cluster_data = [] #Initialize an empty list to store the calculated properties of each cluster

    for label_id in label_list:  # Limit to the first 10 clusters
        points = np.argwhere(labeled_mask == label_id)[:, ::-1] #Find the pixel coordinates belonging to the current cluster and reverse the coordinate order to use x,y convention
        area = len(points) #Calculate the area of the cluster in pixels by counting the number of pixels belonging to it

        if area < 2: #Check whether the cluster contains fewer than two pixels
            continue #Skip clusters that are too small for covariance and principal-axis calculations

        centered = points - points.mean(axis=0) #Center the cluster coordinates around their mean position
        cov = np.cov(centered, rowvar=False) #Calculate the covariance matrix of the centered cluster coordinates
        eigvals, eigvecs = np.linalg.eigh(cov) #Calculate the eigenvalues and eigenvectors of the covariance matrix
        order = np.argsort(eigvals)[::-1] #Determine the indices that sort the eigenvalues from largest to smallest
        eigvals = eigvals[order] #Reorder the eigenvalues so the largest variance is first
        eigvecs = eigvecs[:, order] #Reorder the corresponding eigenvectors to match the sorted eigenvalues

        major_axis_length = 2 * np.sqrt(eigvals[0]) #Calculate the characteristic major-axis length from the standard deviation along the principal axis
        minor_axis_std = np.sqrt(eigvals[1]) #Calculate the standard deviation of the cluster perpendicular to the major axis
        major_axis_vector = eigvecs[:, 0] #Extract the eigenvector corresponding to the major axis
        angle_rad = np.arctan2(major_axis_vector[1], major_axis_vector[0]) #Calculate the major-axis orientation angle in radians
        
        # Constrain orientation to between -90 and 90 degrees
        angle_deg = np.degrees(angle_rad) #Convert the major-axis orientation from radians to degrees
        if angle_deg > 90: #Check whether the orientation is greater than the upper allowed limit
            angle_deg -= 180 #Subtract 180 degrees to bring the orientation into the -90 to 90 degree range
        elif angle_deg < -90: #Check whether the orientation is less than the lower allowed limit
            angle_deg += 180 #Add 180 degrees to bring the orientation into the -90 to 90 degree range
        
        #plt.scatter(points[:, 0], points[:, 1], s=50, label='Cluster Points')
        #plt.show()

        projections = centered @ eigvecs[:, 1] #Project the centered cluster coordinates onto the minor-axis eigenvector
        max_width = projections.max() - projections.min() #Calculate the maximum cluster width perpendicular to the major axis
        fractal_dim = box_counting_fractal_dim(points) #Calculate the box-counting fractal dimension of the cluster point distribution

        # Optional plotting
        if plot: #Check whether cluster visualization has been requested
            center = points.mean(axis=0) #Calculate the center coordinates of the cluster
            dx, dy = major_axis_vector * major_axis_length * 1.5 #Calculate the endpoints of an extended major-axis line for visualization
            minor_vector = eigvecs[:, 1] #Extract the eigenvector corresponding to the minor axis
            dx_minor, dy_minor = minor_vector * minor_axis_std * 2 #Calculate the endpoints of the minor-axis line representing two standard deviations

            plt.figure(figsize=(8, 8)) #Create a new square figure for plotting the cluster
            plt.scatter(points[:, 0], points[:, 1], s=50, label='Cluster Points') #Plot the individual pixels belonging to the cluster
            plt.plot([center[0] - dx, center[0] + dx],
                    [center[1] - dy, center[1] + dy],
                    color='red', linewidth=2, label='Longest Axis') #Plot the major axis through the cluster center
            plt.plot([center[0] - dx_minor, center[0] + dx_minor],
                    [center[1] - dy_minor, center[1] + dy_minor],
                    color='blue', linewidth=2, linestyle='--', label='2σ Scatter') #Plot the minor-axis scatter corresponding to two standard deviations
            plt.scatter(*center, color='black', marker='x', label='Center') #Mark the center of the cluster
            plt.title(f'Cluster {label_id}') #Set the plot title to identify the current cluster
            plt.axis('square') #Force equal scaling of the x and y axes
            plt.legend() #Display the plot legend
            plt.show() #Render the cluster plot

        if print_statement: #Check whether calculated cluster properties should be printed
            print(f"Cluster {label_id}:") #Print the current cluster label
            print(f"  Area: {area}") #Print the cluster area in pixels
            print(f"  Major axis length: {major_axis_length:.2f}") #Print the calculated major-axis length
            print(f"  Scatter orthogonal to main axis (std dev): {minor_axis_std*2:.2f}") #Print twice the minor-axis standard deviation as the orthogonal scatter width
            print(f"  Orientation (deg): {angle_deg:.2f}") #Print the major-axis orientation in degrees
            print(f"  Max width: {max_width:.2f}") #Print the maximum width perpendicular to the major axis
            print(f"  Fractal dimension: {fractal_dim:.2f}") #Print the calculated fractal dimension
            print() #Print a blank line to visually separate the output for different clusters

        cluster_data.append({ #Add the calculated properties of the current cluster to the cluster data list
            'label id': label_id, #Store the original label identifying the cluster
            'area': area/(pixel_micron**2), #Convert the cluster area from pixels squared to physical area in square micrometers
            'length': major_axis_length/pixel_micron, #Convert the major-axis length from pixels to micrometers
            'width scatter': (minor_axis_std/pixel_micron)*2, #Convert the two-standard-deviation minor-axis scatter from pixels to micrometers
            'orientation (deg)': angle_deg, #Store the major-axis orientation in degrees
            'max width': max_width/pixel_micron, #Convert the maximum cluster width from pixels to micrometers
            'fractal dim': fractal_dim, #Store the calculated box-counting fractal dimension
            'total clusters': num_features #Store the total number of unique labels in the original mask
        })

    df = pd.DataFrame(cluster_data) #Convert the list of cluster property dictionaries into a pandas DataFrame
    total_area = df['area'].sum() #Calculate the total physical area of all analyzed clusters
    df['fractional area'] = df['area'] / total_area #Calculate the fraction of total cluster area represented by each cluster
    df['scatter ratio'] = df['width scatter'] / df['length'] #Calculate the ratio of minor-axis scatter to major-axis length for each cluster
    average_angle = np.average(df['orientation (deg)'],weights = df['fractional area']) #Calculate the area-weighted average orientation of all clusters
    if fixed_angle is None and not angle_offset: #Check whether neither a fixed reference angle nor a nonzero angle offset was provided
        df['relative orientation (deg)'] = df['orientation (deg)'] - average_angle #Calculate each cluster's orientation relative to the area-weighted average orientation
    elif fixed_angle is not None: #Check whether a fixed reference angle was provided
        df['relative orientation (deg)'] = df['orientation (deg)'] - fixed_angle #Calculate each cluster's orientation relative to the fixed reference angle
    elif angle_offset is not None and fixed_angle is None: #Check whether an angle offset was provided without a fixed reference angle
        df['relative orientation (deg)'] = df['orientation (deg)'] - average_angle + angle_offset #Calculate each cluster's orientation relative to the average orientation and then apply the angular offset
    else: #Handle the remaining case where both an angle offset and fixed angle are provided
        df['relative orientation (deg)'] = df['orientation (deg)'] - fixed_angle + angle_offset #Calculate each cluster's orientation relative to the fixed angle and apply the angular offset
    # Normalize the angle to be within -180 to 180 degrees
    df['relative orientation (deg)'] = ((df['relative orientation (deg)'] + 90) % 180) - 90 #Normalize relative orientations to the -90 to 90 degree range
    df['order parameter'] = np.cos(np.radians(df['relative orientation (deg)'])) #Calculate the cosine-based orientation order parameter from the relative orientation
    #df['particle'] = jpg_path.replace(".jpg", "")

    return df #Return the DataFrame containing the calculated properties for all analyzed clusters

def analyze_debris(debris_mask, pixel_micron, angle_offset = None, fixed_angle = None, plot=False, print_statement=False):
    '''Identifies and analyzes individual debris particles within a binary debris
    mask. Each connected debris region is labeled and analyzed independently
    using principal component analysis (PCA) to determine its major-axis
    length, minor-axis scatter, orientation, maximum width, and fractal
    dimension. Debris regions outside the specified size range are excluded.
    The function optionally plots and prints the calculated properties,
    converts pixel-based measurements into physical units, and returns the
    results as a pandas DataFrame.

    Inputs:
        debris_mask (numpy.ndarray): A 2D binary mask in which foreground
            pixels represent debris and background pixels are represented by 0.
        pixel_micron (int or float): The physical size of one pixel in
            micrometers. Used to convert pixel-based lengths, widths, and areas
            into physical units.
        angle_offset (int, float, or None): Optional angular offset in degrees
            to apply when calculating relative orientations. Defaults to None.
        fixed_angle (int, float, or None): Optional fixed reference angle in
            degrees. If provided, debris orientations are measured relative to
            this angle instead of the average debris orientation. Defaults to
            None.
        plot (bool): Determines whether a plot showing each debris cluster's
            points, major axis, minor-axis scatter, and center is generated.
            Defaults to False.
        print_statement (bool): Determines whether calculated properties for
            each debris cluster are printed to the console. Defaults to False.

    Outputs:
        pandas.DataFrame: A DataFrame containing one row for each analyzed
            debris cluster. The DataFrame includes the cluster label, physical
            area, major-axis length, minor-axis scatter, orientation, maximum
            width, fractal dimension, total number of clusters, fractional
            area, scatter ratio, relative orientation, and order parameter.
    '''

    labeled_mask, num_features = label(debris_mask) #Label each connected debris region in the binary mask and record the total number of identified regions
    y_coords, x_coords = np.nonzero(labeled_mask) #Find the y and x coordinates of every non-background pixel in the labeled mask
    labels = labeled_mask[y_coords, x_coords] #Retrieve the label associated with each non-background pixel
    coords = np.column_stack((x_coords, y_coords, labels)) #Combine the x coordinate, y coordinate, and region label into a single array

    cluster_data = [] #Initialize an empty list to store the calculated properties of each debris cluster

    for label_id in range(1, num_features + 1):  # Limit to the first 10 clusters
        points = coords[coords[:, 2] == label_id][:, :2] #Extract the x and y coordinates belonging to the current debris cluster
        area = len(points) #Calculate the area of the debris cluster in pixels by counting its pixels

        if area < 2  or area > 100: #Check whether the debris cluster is too small or too large to be included in the analysis
            continue #Skip the current debris cluster if it falls outside the accepted area range

        centered = points - points.mean(axis=0) #Center the cluster coordinates around their mean position
        cov = np.cov(centered, rowvar=False) #Calculate the covariance matrix of the centered cluster coordinates
        eigvals, eigvecs = np.linalg.eigh(cov) #Calculate the eigenvalues and eigenvectors of the covariance matrix
        order = np.argsort(eigvals)[::-1] #Determine the indices that sort the eigenvalues from largest to smallest
        eigvals = eigvals[order] #Reorder the eigenvalues so the largest variance is first
        eigvecs = eigvecs[:, order] #Reorder the corresponding eigenvectors to match the sorted eigenvalues

        major_axis_length = 2 * np.sqrt(eigvals[0]) #Calculate the characteristic major-axis length from the standard deviation along the major principal axis
        minor_axis_std = np.sqrt(eigvals[1]) #Calculate the standard deviation of the cluster perpendicular to the major axis
        major_axis_vector = eigvecs[:, 0] #Extract the eigenvector corresponding to the major axis
        angle_rad = np.arctan2(major_axis_vector[1], major_axis_vector[0]) #Calculate the major-axis orientation angle in radians
        
        # Constrain orientation to between -90 and 90 degrees
        angle_deg = np.degrees(angle_rad) #Convert the major-axis orientation from radians to degrees
        if angle_deg > 90: #Check whether the orientation is greater than the upper allowed limit
            angle_deg -= 180 #Subtract 180 degrees to bring the orientation into the -90 to 90 degree range
        elif angle_deg < -90: #Check whether the orientation is less than the lower allowed limit
            angle_deg += 180 #Add 180 degrees to bring the orientation into the -90 to 90 degree range

        projections = centered @ eigvecs[:, 1] #Project the centered cluster coordinates onto the minor-axis eigenvector
        max_width = projections.max() - projections.min() #Calculate the maximum width of the debris cluster perpendicular to its major axis
        fractal_dim = box_counting_fractal_dim(points) #Calculate the box-counting fractal dimension of the debris cluster

        # Optional plotting
        if plot: #Check whether visualization of the current debris cluster has been requested
            center = points.mean(axis=0) #Calculate the center coordinates of the debris cluster
            dx, dy = major_axis_vector * major_axis_length * 1.5 #Calculate the endpoints of an extended major-axis line for visualization
            minor_vector = eigvecs[:, 1] #Extract the eigenvector corresponding to the minor axis
            dx_minor, dy_minor = minor_vector * minor_axis_std * 2 #Calculate the endpoints of the minor-axis line representing two standard deviations

            plt.figure(figsize=(8, 8)) #Create a new square figure for plotting the debris cluster
            plt.scatter(points[:, 0], points[:, 1], s=50, label='Cluster Points') #Plot the individual pixels belonging to the debris cluster
            plt.plot([center[0] - dx, center[0] + dx],
                    [center[1] - dy, center[1] + dy],
                    color='red', linewidth=2, label='Longest Axis') #Plot the major axis through the debris cluster center
            plt.plot([center[0] - dx_minor, center[0] + dx_minor],
                    [center[1] - dy_minor, center[1] + dy_minor],
                    color='blue', linewidth=2, linestyle='--', label='2σ Scatter') #Plot the minor-axis scatter corresponding to two standard deviations
            plt.scatter(*center, color='black', marker='x', label='Center') #Mark the center of the debris cluster
            plt.title(f'Cluster {label_id}') #Set the plot title to identify the current debris cluster
            plt.axis('square') #Force equal scaling of the x and y axes
            plt.legend() #Display the plot legend
            plt.show() #Render the debris cluster plot

        if print_statement: #Check whether calculated debris properties should be printed
            print(f"Cluster {label_id}:") #Print the current debris cluster label
            print(f"  Area: {area}") #Print the debris cluster area in pixels
            print(f"  Major axis length: {major_axis_length:.2f}") #Print the calculated major-axis length
            print(f"  Scatter orthogonal to main axis (std dev): {minor_axis_std*2:.2f}") #Print twice the minor-axis standard deviation as the orthogonal scatter width
            print(f"  Orientation (deg): {angle_deg:.2f}") #Print the major-axis orientation in degrees
            print(f"  Max width: {max_width:.2f}") #Print the maximum width perpendicular to the major axis
            print(f"  Fractal dimension: {fractal_dim:.2f}") #Print the calculated fractal dimension
            print() #Print a blank line to separate the output for different debris clusters

        cluster_data.append({ #Add the calculated properties of the current debris cluster to the cluster data list
            'label id': label_id, #Store the label identifying the debris cluster
            'area': area/(pixel_micron**2), #Convert the debris cluster area from pixels squared to physical area in square micrometers
            'length': major_axis_length/pixel_micron, #Convert the major-axis length from pixels to micrometers
            'width scatter': (minor_axis_std/pixel_micron)*2, #Convert the two-standard-deviation minor-axis scatter from pixels to micrometers
            'orientation (deg)': angle_deg, #Store the major-axis orientation in degrees
            'max width': max_width/pixel_micron, #Convert the maximum cluster width from pixels to micrometers
            'fractal dim': fractal_dim, #Store the calculated box-counting fractal dimension
            'total clusters': num_features #Store the total number of connected debris regions identified in the mask
        })

    df = pd.DataFrame(cluster_data) #Convert the list of debris property dictionaries into a pandas DataFrame
    total_area = df['area'].sum() #Calculate the total physical area of all analyzed debris clusters
    df['fractional area'] = df['area'] / total_area #Calculate the fraction of the total debris area represented by each cluster
    df['scatter ratio'] = df['width scatter'] / df['length'] #Calculate the ratio of minor-axis scatter to major-axis length for each cluster
    average_angle = np.average(df['orientation (deg)'],weights = df['fractional area']) #Calculate the area-weighted average orientation of all analyzed debris clusters
    if fixed_angle is None and not angle_offset: #Check whether neither a fixed reference angle nor a nonzero angle offset was provided
        df['relative orientation (deg)'] = df['orientation (deg)'] - average_angle #Calculate each cluster's orientation relative to the area-weighted average orientation
    elif fixed_angle is not None: #Check whether a fixed reference angle was provided
        df['relative orientation (deg)'] = df['orientation (deg)'] - fixed_angle #Calculate each cluster's orientation relative to the fixed reference angle
    else: #Handle the remaining case where an angle offset is provided without a fixed reference angle
        df['relative orientation (deg)'] = df['orientation (deg)'] - average_angle + angle_offset #Calculate each cluster's orientation relative to the average orientation and then apply the angular offset
    # Normalize the angle to be within -180 to 180 degrees
    df['relative orientation (deg)'] = ((df['relative orientation (deg)'] + 90) % 180) - 90 #Normalize relative orientations to the -90 to 90 degree range
    df['order parameter'] = np.cos(np.radians(df['relative orientation (deg)'])) #Calculate the cosine of each relative orientation to produce the orientation order parameter

    return df #Return the DataFrame containing the calculated properties for all analyzed debris clusters

def process_region(args):
    """
    Worker function to relabel connected components in a small region of the mask.

    Parameters:
        args (tuple): (region_array, region_slice, original_label, starting_label)

    Returns:
        List of (global_coords, new_labels)
    """
    region_array, region_slice, original_label, starting_label = args

    # Identify binary mask of this label in the region
    binary_region = (region_array == original_label)

    labeled_region, num_features = label(binary_region)
    output = []

    for i in range(1, num_features + 1):
        component_coords = np.argwhere(labeled_region == i)
        global_coords = component_coords + [region_slice[0].start, region_slice[1].start]
        labels = np.full(len(global_coords), starting_label, dtype=np.int32)
        output.append((global_coords, labels))
        starting_label += 1

    return output

def relabel_disconnected_regions_parallel(mask, max_label=None, n_processes=None):
    """
    Relabel disconnected regions in parallel.

    Parameters:
        mask (np.ndarray): Labeled 2D mask.
        max_label (int, optional): Max label to consider (default: np.max(mask))
        n_processes (int, optional): Number of parallel workers.

    Returns:
        new_mask (np.ndarray): Relabeled mask.
    """
    if max_label is None:
        unique_labels = np.unique(mask)
    else:
        unique_labels = np.arange(1, max_label + 1)

    tasks = []
    label_counter = 1  # Start labeling from 1

    unique_labels = np.unique(mask)
    unique_labels = range(0,504)
    for lbl in unique_labels:
        if lbl == 0:
            continue

        coords = np.argwhere(mask == lbl)
        if coords.size == 0:
            continue

        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0) + 1
        region_slice = (slice(min_row, max_row), slice(min_col, max_col))
        region_array = mask[region_slice]

        tasks.append((region_array.copy(), region_slice, lbl, label_counter))
        # Estimate how many labels we might need to reserve
        label_counter += 100  # Overallocate a block to prevent overlaps

    new_mask = np.zeros_like(mask, dtype=np.int32)

    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_region, tasks)

    label_counter = 1
    for region_result in results:
        for global_coords, labels in region_result:
            rows, cols = global_coords[:, 0], global_coords[:, 1]
            new_mask[rows, cols] = labels
            label_counter += 1

    return new_mask

def crop_and_relabel_mask(mask, target_value, pad=150):
    """
    Crops and relabels a mask:
    
    - 0: background (original 0s)
    - 1: everything nonzero except the region of interest
    - 2: the region of interest

    Parameters:
        mask (ndarray): 2D input mask array
        target_value (int): value corresponding to the region of interest
        pad (int): number of pixels to pad around bounding box

    Returns:
        cropped_relabelled (ndarray): cropped and relabelled mask
    """
    # Make boolean mask for target region
    roi_mask = mask == target_value

    coords = np.argwhere(roi_mask)  # (row, col) pairs
    #print(f"Coordinates of region {target_value} (in original array):")
    #for y, x in coords:
    #    print(f"({y}, {x})")

    if coords.size == 0:
        raise ValueError("Target value not found in the mask.")

    # Find bounding box of the region of interest
    row_min, row_max = coords[:, 0].min(), coords[:, 0].max()
    col_min, col_max = coords[:, 1].min(), coords[:, 1].max()

    # Apply padding
    row_min_p = max(row_min - pad, 0)
    row_max_p = min(row_max + pad, mask.shape[0] - 1)
    col_min_p = max(col_min - pad, 0)
    col_max_p = min(col_max + pad, mask.shape[1] - 1)

    # Crop the mask
    cropped = mask[row_min_p:row_max_p+1, col_min_p:col_max_p+1]

    # Initialize new mask
    new_mask = np.zeros_like(cropped, dtype=np.uint8)

    # Set 2 for region of interest
    new_mask[cropped == target_value] = 2

    # Set 1 for all other nonzero entries
    new_mask[(cropped != 0) & (cropped != target_value)] = 1

    return new_mask

def recalculate_chain_params(all_chain_df):
    '''
    Recalculates orientation, nematic order parameter, and area-weighted
    structural properties for a collection of chains. The function extracts
    the relevant chain-level parameters from an input DataFrame, calculates
    the area-weighted average orientation, uses that orientation to determine
    each chain's relative orientation and nematic order parameter, and then
    calculates area-weighted averages of the geometric and structural
    properties across all chains.

    Inputs:
        all_chain_df (pandas.DataFrame): A DataFrame containing chain-level
            measurements and calculated properties. The DataFrame must contain
            columns including 'label id', 'area', 'length', 'width scatter',
            'orientation (deg)', 'max width', 'fractal dim', and
            'scatter ratio'. The 'area' column is used as the weighting factor
            for the area-weighted calculations.

    Outputs:
        chain_df (pandas.DataFrame): A DataFrame containing the selected
            chain-level properties along with recalculated relative
            orientations, nematic order parameters, and area-weighted average
            structural properties.
    '''

    chain_df = all_chain_df.loc[:,'label id':'scatter ratio'] #Select the columns from 'label id' through 'scatter ratio' from the input DataFrame

    av_orientation = np.average(chain_df.loc[:,'orientation (deg)'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average orientation of all chains
    chain_df['relative orientation'] = chain_df['orientation (deg)'].to_numpy() - av_orientation #Calculate each chain's orientation relative to the area-weighted average orientation

    nem = lambda θ: (2)*(np.cos(np.deg2rad(θ)))**2 - 1 #Define a function that calculates the nematic order parameter from an orientation angle
    chain_df['order parameter'] = nem(chain_df['relative orientation'].to_numpy()) #Calculate the nematic order parameter for each chain using its relative orientation

    chain_df['weighted average length'] = np.average(chain_df.loc[:,'length'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average chain length
    chain_df['weighted average width scatter'] = np.average(chain_df.loc[:,'width scatter'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average width scatter
    chain_df['weighted average orientation'] = np.average(chain_df.loc[:,'orientation (deg)'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average absolute orientation
    chain_df['weighted average relative orientation'] = np.average(chain_df.loc[:,'relative orientation'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average relative orientation
    chain_df['weighted average max width'] = np.average(chain_df.loc[:,'max width'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average maximum chain width
    chain_df['weighted average fractal dim'] = (np.average(chain_df.loc[chain_df['fractal dim'].notna(), 'fractal dim'],weights=chain_df.loc[chain_df['fractal dim'].notna(), 'area'])) #Calculate the area-weighted average fractal dimension while excluding chains with missing fractal dimension values
    chain_df['weighted average scatter ratio'] = np.average(chain_df.loc[:,'scatter ratio'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average ratio of width scatter to chain length
    chain_df['weighted average order parameter'] = np.average(chain_df.loc[:,'order parameter'].to_numpy(), weights = chain_df.loc[:,'area'].to_numpy()) #Calculate the area-weighted average nematic order parameter

    return chain_df #Return the DataFrame containing the recalculated chain properties and area-weighted averages

# Magnetometry
def load_data(sample, angles, max_angle):
    '''
    Loads and processes magnetic measurement data from Excel files organized
    within a sample- and angle-specific directory structure. The function
    identifies files corresponding to the requested sample, extracts raw
    magnetic moment and applied-field data, processes smoothed magnetic moment
    data, and stores the measurements according to their corresponding sample
    angle.

    Inputs:
        sample (str): The name of the sample directory containing the magnetic
            measurement data. The first three characters are also used to
            identify the weight-percent portion of the sample name when
            matching Excel filenames.
        angles (iterable): A collection of measurement angles, typically in
            degrees, used to initialize dictionaries for storing the processed
            and raw data.
        max_angle (str): The name of the subdirectory within the sample
            directory containing the Excel measurement files.

    Outputs:
        raw_data (dict): A dictionary mapping each measurement angle to its
            corresponding raw magnetic moment data as a NumPy array.
        data (dict): A dictionary mapping each measurement angle to its
            corresponding processed or smoothed magnetic moment data as a
            NumPy array.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values from the most recently processed matching
            Excel file.
    '''

    path = os.path.join(os.getcwd(), sample, max_angle) #Construct the path to the directory containing the Excel measurement files
    excel_files = [f for f in os.listdir(path) if f.endswith('.xlsx')] #Create a list containing the names of all Excel files in the directory
    data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for processed data
    raw_data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for raw magnetic moment data
    wt_percent = sample[:3] #Extract the first three characters of the sample name to use when identifying matching measurement files

    if sample == '0.2_20mT': #Check whether the sample is the specific 0.2 wt% sample measured at 20 mT
        start = 2 #Set the starting row index to 2 for this sample
    else: #Handle all other sample types
        start = 5 #Set the starting row index to 5 for all other samples

    # Loop through each Excel file
    for filename in excel_files: #Loop through every Excel measurement file found in the sample directory
        file_path = os.path.join(path, filename) #Construct the complete file path for the current Excel file
        match = re.search(rf'{wt_percent}-(\d+)deg', filename) #Search the filename for the sample weight percentage followed by an angle in degrees

        if match: #Continue processing only if the filename matches the expected naming pattern
            angle = int(match.group(1)) #Extract the measurement angle from the filename and convert it from a string to an integer
            df = pd.read_excel(file_path) #Load the current Excel file into a pandas DataFrame
            moment = df.loc[start:,'Magnetic Moment'].dropna().to_numpy() #Extract the non-empty magnetic moment measurements beginning at the specified starting row and convert them to a NumPy array
            max_moment = np.max(moment) #Determine the maximum measured magnetic moment for the current dataset
            min_moment = np.min(moment) #Determine the minimum measured magnetic moment for the current dataset
            smoothed_upper_norm = df.loc[start:,'Smoothed Y1'].dropna().to_numpy() #Extract the non-empty normalized upper smoothed magnetic moment data and convert it to a NumPy array
            smoothed_lower_norm = df.loc[start:,'Smoothed Y2'].dropna().to_numpy() #Extract the non-empty normalized lower smoothed magnetic moment data and convert it to a NumPy array
            smoothed_norm = np.hstack((smoothed_upper_norm,smoothed_lower_norm)) #Concatenate the upper and lower smoothed datasets into a single array
            if sample == '0.2_20mT': #Check whether the current sample requires conversion of normalized smoothed data back to magnetic moment units
                smoothed_moment = ((smoothed_norm+1)/2)*(max_moment-min_moment)+min_moment #Rescale the normalized smoothed data from the range [-1,1] back to the measured magnetic moment range
            else: #Handle all other samples where the smoothed data is already in the desired units
                smoothed_moment = smoothed_norm #Use the smoothed data without additional rescaling
            raw_data[angle] = moment #Store the raw magnetic moment measurements in the dictionary under their corresponding measurement angle
            data[angle] = smoothed_moment #Store the processed smoothed magnetic moment measurements in the dictionary under their corresponding measurement angle
            applied_field = df.loc[start:,'Applied_Field'].dropna().to_numpy() #Extract the non-empty applied magnetic field measurements and convert them to a NumPy array

    return raw_data, data, applied_field #Return the dictionaries of raw and processed magnetic moment data along with the applied magnetic field array

def load_x_y_data(sample, angles, max_angle):
    '''
    Loads X-direction, Y-direction, raw magnetic moment, and smoothed magnetic
    moment data from Excel files containing measurements collected at different
    sample angles. The function searches for files matching the specified
    sample and angle naming convention, extracts the relevant measurement
    columns, processes the smoothed magnetic moment data, and stores the
    resulting datasets in dictionaries indexed by their corresponding angles.

    Inputs:
        sample (str): The name of the sample directory containing the magnetic
            measurement data. The first three characters are used to identify
            the weight-percent portion of the sample name when matching Excel
            filenames.
        angles (iterable): A collection of measurement angles, typically in
            degrees, used to initialize dictionaries for storing the X-direction,
            Y-direction, raw magnetic moment, and smoothed magnetic moment data.
        max_angle (str): The name of the subdirectory within the sample
            directory containing the Excel measurement files.

    Outputs:
        x_data (dict): A dictionary mapping each measurement angle to a NumPy
            array containing the corresponding magnetic signal measured in the
            X direction.
        y_data (dict): A dictionary mapping each measurement angle to a NumPy
            array containing the corresponding magnetic signal measured in the
            Y direction.
        data (dict): A dictionary mapping each measurement angle to a NumPy
            array containing the corresponding processed or smoothed magnetic
            moment measurements.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values from the most recently processed matching
            Excel file.
    '''

    path = os.path.join(os.getcwd(), sample, max_angle) #Construct the path to the directory containing the Excel measurement files
    excel_files = [f for f in os.listdir(path) if f.endswith('.xlsx')] #Create a list containing the names of all Excel files in the directory that have an .xlsx extension
    data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for processed magnetic moment data
    raw_data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for raw magnetic moment data
    x_data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for X-direction magnetic signal data
    y_data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for Y-direction magnetic signal data
    wt_percent = sample[:3] #Extract the first three characters of the sample name to use when identifying matching measurement files


    if sample == '0.2_20mT': #Check whether the sample is the specific 0.2 wt% sample measured at 20 mT
        start = 2 #Set the starting row index to 2 for this sample
    else: #Handle all other sample types
        start = 5 #Set the starting row index to 5 for all other samples

    # Loop through each Excel file
    for filename in excel_files: #Loop through every Excel measurement file found in the sample directory
        file_path = os.path.join(path, filename) #Construct the complete file path for the current Excel file
        match = re.search(rf'{wt_percent}-(\d+)deg', filename) #Search the filename for the expected sample weight percentage and measurement angle pattern

        if match: #Continue processing only if the filename matches the expected naming pattern
            angle = int(match.group(1)) #Extract the measurement angle from the filename and convert it from a string to an integer
            df = pd.read_excel(file_path) #Load the current Excel file into a pandas DataFrame
            x_data_i = df.loc[start:,'Signal_X_direction'].dropna().to_numpy() #Extract the non-empty X-direction magnetic signal data and convert it to a NumPy array
            y_data_i = df.loc[start:,'Signal_Y_direction'].dropna().to_numpy() #Extract the non-empty Y-direction magnetic signal data and convert it to a NumPy array


            moment = df.loc[start:,'Magnetic Moment'].dropna().to_numpy() #Extract the non-empty raw magnetic moment measurements and convert them to a NumPy array
            max_moment = np.max(moment) #Determine the maximum magnetic moment in the current dataset
            min_moment = np.min(moment) #Determine the minimum magnetic moment in the current dataset
            smoothed_upper_norm = df.loc[start:,'Smoothed Y1'].dropna().to_numpy() #Extract the non-empty normalized upper smoothed magnetic moment data and convert it to a NumPy array
            smoothed_lower_norm = df.loc[start:,'Smoothed Y2'].dropna().to_numpy() #Extract the non-empty normalized lower smoothed magnetic moment data and convert it to a NumPy array
            smoothed_norm = np.hstack((smoothed_upper_norm,smoothed_lower_norm)) #Concatenate the upper and lower smoothed datasets into a single array
            if sample == '0.2_20mT': #Check whether the current sample requires conversion of normalized smoothed data back to magnetic moment units
                smoothed_moment = ((smoothed_norm+1)/2)*(max_moment-min_moment)+min_moment #Rescale the normalized smoothed data from the range [-1,1] back to the measured magnetic moment range
            else: #Handle all other samples where the smoothed data is already in the desired units
                smoothed_moment = smoothed_norm #Use the smoothed data without additional rescaling
            raw_data[angle] = moment #Store the raw magnetic moment measurements in the dictionary under their corresponding measurement angle
            data[angle] = smoothed_moment #Store the processed smoothed magnetic moment measurements in the dictionary under their corresponding measurement angle
            x_data[angle] = x_data_i #Store the X-direction magnetic signal data in the dictionary under its corresponding measurement angle
            y_data[angle] = y_data_i #Store the Y-direction magnetic signal data in the dictionary under its corresponding measurement angle
            applied_field = df.loc[start:,'Applied_Field'].dropna().to_numpy() #Extract the non-empty applied magnetic field values and convert them to a NumPy array

    return x_data, y_data, data, applied_field #Return the X-direction data, Y-direction data, processed magnetic moment data, and applied magnetic field data

def plot_hysteresis(data,applied_field, cmap = viridis):
    '''
    Plots smoothed magnetic hysteresis data as a function of applied magnetic
    field for multiple measurement angles. Each dataset is plotted using a
    color determined by its corresponding angle, with the colors normalized
    across the range of available angle values.

    Inputs:
        data (dict): A dictionary in which each key represents a measurement
            angle and each value is a NumPy array containing the corresponding
            smoothed magnetic moment measurements.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values corresponding to the magnetic moment data.
        cmap (matplotlib.colors.Colormap): A Matplotlib colormap used to assign
            a distinct color to each measurement angle. Defaults to the
            `viridis` colormap.

    Outputs:
        None: The function does not return a value. It generates and displays
            a Matplotlib figure containing the hysteresis curves.
    '''

    keys = sorted(data.keys()) #Sort the measurement-angle keys from smallest to largest
    norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization function that maps the minimum and maximum angles to the bounds of the colormap
    cmap = cmap #Assign the provided colormap to the local cmap variable

    plt.figure(figsize = (10,8)) #Create a new Matplotlib figure with a width of 10 inches and height of 8 inches
    for key in data.keys(): #Loop through each measurement angle in the data dictionary
        color = cmap(norm(key)) #Convert the current angle into a normalized value and use it to select a color from the colormap
        y = data[key] #Retrieve the smoothed magnetic moment data corresponding to the current angle
        plt.plot(applied_field, y, label = str(key), color = color) #Plot the magnetic moment against applied magnetic field using the angle as the curve label
        plt.legend(loc = 'lower right') #Display the legend in the lower-right corner of the plot
        plt.xlabel('Applied Field') #Set the x-axis label to indicate the applied magnetic field
        plt.ylabel('Smoothed Magnetic Moment') #Set the y-axis label to indicate the smoothed magnetic moment
        plt.title('Moment vs. Applied Field') #Set the title of the hysteresis plot
    plt.show() #Display the completed hysteresis plot

def load_x_y_data360(sample, angles, max_angle):
    '''
    Loads X- and Y-direction magnetic signal data from Excel files containing
    measurements collected at different sample angles. The function searches
    for files matching the specified sample and angle naming convention,
    extracts the X-direction signal, Y-direction signal, and applied magnetic
    field data, and stores the X and Y measurements in dictionaries indexed by
    their corresponding angles.

    Inputs:
        sample (str): The name of the sample directory containing the magnetic
            measurement data. The first three characters are used to identify
            the weight-percent portion of the sample name when matching Excel
            filenames.
        angles (iterable): A collection of measurement angles, typically in
            degrees, used to initialize dictionaries for storing the X- and
            Y-direction measurements.
        max_angle (str): The name of the subdirectory within the sample
            directory containing the Excel measurement files.

    Outputs:
        x_data (dict): A dictionary mapping each measurement angle to a NumPy
            array containing the corresponding magnetic signal measured in the
            X direction.
        y_data (dict): A dictionary mapping each measurement angle to a NumPy
            array containing the corresponding magnetic signal measured in the
            Y direction.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values from the most recently processed matching
            Excel file.
    '''

    path = os.path.join(os.getcwd(), sample, max_angle) #Construct the path to the directory containing the Excel measurement files
    excel_files = [f for f in os.listdir(path) if f.endswith('.xlsx')] #Create a list containing the names of all Excel files in the directory that have an .xlsx extension
    x_data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for X-direction data
    y_data = {i:None for i in angles} #Initialize a dictionary with each requested angle as a key and None as its initial value for Y-direction data
    wt_percent = sample[:3] #Extract the first three characters of the sample name to use when identifying matching measurement files

    start = 0 #Set the starting row index used when extracting data from each Excel file

    # Loop through each Excel file
    for filename in excel_files: #Loop through every Excel measurement file found in the sample directory
        file_path = os.path.join(path, filename) #Construct the complete file path for the current Excel file
        match = re.search(rf'manip_{wt_percent}-(\d+)deg', filename) #Search the filename for the expected sample and angle naming pattern

        if match: #Continue processing only if the filename matches the expected pattern
            angle = int(match.group(1)) #Extract the measurement angle from the filename and convert it from a string to an integer
            df = pd.read_excel(file_path) #Load the current Excel file into a pandas DataFrame
            x_data_i = df.loc[start:,'Signal_X_direction'].dropna().to_numpy() #Extract the non-empty X-direction magnetic signal data and convert it to a NumPy array
            y_data_i = df.loc[start:,'Signal_Y_direction'].dropna().to_numpy() #Extract the non-empty Y-direction magnetic signal data and convert it to a NumPy array
            x_data[angle] = x_data_i #Store the X-direction signal data in the dictionary under its corresponding measurement angle
            y_data[angle] = y_data_i #Store the Y-direction signal data in the dictionary under its corresponding measurement angle
            applied_field = df.loc[start:,'Applied_Field'].dropna().to_numpy() #Extract the non-empty applied magnetic field values and convert them to a NumPy array

    return x_data, y_data, applied_field #Return the X-direction data dictionary, Y-direction data dictionary, and applied magnetic field array

def plot_hysteresis3(data, applied_field, lower_bound=None, upper_bound=None, ylims = None, cmap = viridis):
    '''
    Plots smoothed magnetic hysteresis data as a function of applied magnetic
    field for multiple measurement angles. Each dataset is assigned a color
    based on its angle, and optional lower and upper applied-field bounds can
    be used to restrict the plotted data. An optional y-axis limit can also
    be specified.

    Inputs:
        data (dict): A dictionary in which each key represents a measurement
            angle and each value is a NumPy array containing the corresponding
            smoothed magnetic moment measurements.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values corresponding to the magnetic moment data.
        lower_bound (int, float, or None): Optional lower bound for the applied
            magnetic field. Only values greater than this bound are plotted.
            Defaults to None.
        upper_bound (int, float, or None): Optional upper bound for the applied
            magnetic field. Only values less than this bound are plotted.
            Defaults to None.
        ylims (tuple, list, or None): Optional pair of values defining the
            lower and upper limits of the y-axis. Defaults to None.
        cmap (matplotlib.colors.Colormap): A Matplotlib colormap used to assign
            colors to the curves based on their measurement angles. Defaults
            to the `viridis` colormap.

    Outputs:
        None: The function does not return a value. It generates and displays
            a Matplotlib figure containing the filtered hysteresis curves.
    '''

    keys = sorted(data.keys()) #Sort the measurement-angle keys from smallest to largest
    norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization function that maps the minimum and maximum angles to the bounds of the colormap
    cmap = cmap #Assign the provided colormap to the local cmap variable

    #plt.figure(figsize=(10, 13)) #Create a new Matplotlib figure with a width of 10 inches and height of 13 inches
    plt.figure(figsize=(10, 10)) #Create a new Matplotlib figure with a width of 10 inches and height of 13 inches

    for key in keys: #Loop through each measurement angle in sorted order
        color = cmap(norm(key)) #Convert the current angle into a normalized value and use it to select a color from the colormap
        y = data[key] #Retrieve the smoothed magnetic moment data corresponding to the current angle
        ap_field = applied_field #Create a local reference to the applied magnetic field array

        # Apply mask for bounds
        mask = np.ones_like(ap_field, dtype=bool) #Create a Boolean mask initially set to True for every applied-field measurement
        if lower_bound is not None: #Check whether a lower applied-field bound has been specified
            mask &= ap_field > lower_bound #Keep only measurements whose applied field is greater than the lower bound
        if upper_bound is not None: #Check whether an upper applied-field bound has been specified
            mask &= ap_field < upper_bound #Keep only measurements whose applied field is less than the upper bound

        y = y[mask] #Apply the Boolean mask to retain only magnetic moment values within the specified field bounds
        ap_field = ap_field[mask] #Apply the same Boolean mask to retain only applied-field values within the specified bounds

        plt.plot(ap_field, y, label=str(key), color=color, linewidth = 2.5) #Plot the filtered magnetic moment against applied field using the angle as the curve label

    plt.legend(loc='lower right') #Display the legend in the lower-right corner of the plot
    plt.xlabel('Applied Field') #Set the x-axis label to indicate the applied magnetic field
    plt.xlim([lower_bound, upper_bound]) #Set the x-axis limits to the specified lower and upper applied-field bounds
    plt.ylabel('Smoothed Magnetic Moment') #Set the y-axis label to indicate the smoothed magnetic moment
    plt.title('Moment vs. Applied Field') #Set the title of the hysteresis plot
    if ylims is not None: #Check whether custom y-axis limits have been provided
        plt.ylim(ylims) #Apply the specified lower and upper y-axis limits
    plt.show() #Display the completed hysteresis plot

def truncate_data(data,applied_field,lower_bound = None, upper_bound = None):
    '''
    Filters magnetic measurement data and the corresponding applied-field
    values to retain only measurements within optional lower and upper field
    bounds. The same Boolean mask is applied to the applied-field array and
    every dataset stored in the input data dictionary, ensuring that all
    measurements remain aligned.

    Inputs:
        data (dict): A dictionary in which each key identifies a measurement
            condition, such as an angle, and each value is a NumPy array
            containing the corresponding measurement data. Each array must
            have the same length as applied_field.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values corresponding to the measurements in data.
        lower_bound (int, float, or None): Optional lower bound for the applied
            magnetic field. If provided, only values greater than this bound
            are retained. Defaults to None.
        upper_bound (int, float, or None): Optional upper bound for the applied
            magnetic field. If provided, only values less than this bound are
            retained. Defaults to None.

    Outputs:
        filtered_data (dict): A dictionary with the same keys as data, where
            each NumPy array has been filtered using the same field-based mask.
        filtered_field (numpy.ndarray): The applied-field array after removing
            measurements outside the specified bounds.
    '''

    keys = sorted(data.keys()) #Sort the keys in the data dictionary from smallest to largest

    # Build masks separately
    mask_raw = np.ones_like(applied_field, dtype=bool) #Create a Boolean mask initially set to True for every applied-field measurement
    if lower_bound is not None: #Check whether a lower applied-field bound has been specified
        mask_raw &= applied_field > lower_bound #Keep only measurements whose applied field is greater than the lower bound
    if upper_bound is not None: #Check whether an upper applied-field bound has been specified
        mask_raw &= applied_field < upper_bound #Keep only measurements whose applied field is less than the upper bound
    filtered_field= applied_field[mask_raw] #Apply the Boolean mask to the applied-field array to retain only measurements within the specified bounds

    # Prepare outputs
    filtered_data = {key: data[key][mask_raw] for key in data.keys()} #Apply the same Boolean mask to every dataset in the data dictionary

    return filtered_data, filtered_field #Return the filtered measurement data and corresponding applied-field values

def langevin(x, a, b, c):
    '''
    Calculate the Langevin function for an input array using a numerically stable
    approximation near zero and the standard Langevin expression elsewhere.

    Input:
        x - NumPy array or array-like input containing the independent variable values
        at which the Langevin function is evaluated.
        a - Float scaling parameter that controls the magnitude of the Langevin response.
        b - Float scaling parameter that multiplies the input x and controls the argument
        of the Langevin function.
        c - Float offset parameter added to the calculated Langevin response.

    Output:
        out - NumPy array containing the calculated Langevin function values at each
        element of x. A series expansion is used for values of b*x sufficiently close
        to zero to avoid numerical instability, while the standard Langevin expression
        is used for all other values.
    '''

    bx = b * x

    # Create an empty floating-point array with the same shape as bx
    # to store the calculated Langevin function values.
    out = np.empty_like(bx, dtype=float)
    
    # Identify values of bx that are sufficiently close to zero,
    # where direct evaluation of the Langevin expression can become
    # numerically unstable because of division by a very small number.
    small = np.isclose(bx, 0.0)

    # Identify values of bx that are not sufficiently close to zero
    # and can therefore be evaluated using the standard Langevin expression.
    large = ~small
    
    # Use the first-order series expansion of the Langevin function,
    # L(bx) ≈ bx/3, for values sufficiently close to zero.
    # The result is scaled by a and shifted by the offset c.
    out[small] = a * (bx[small] / 3.0) + c
    
    # Evaluate the standard Langevin function,
    # L(bx) = coth(bx) - 1/(bx),
    # for values sufficiently far from zero.
    # Since coth(z) = 1/tanh(z), the expression is evaluated using np.tanh.
    # The result is scaled by a and shifted by the offset c.
    out[large] = a * (1/np.tanh(bx[large]) - 1/(bx[large])) + c
    
    # Return the calculated Langevin function values.
    return out

def langevin_jacobian(x, a, b, c):
    '''
    Calculate the Jacobian matrix of the Langevin function with respect to the
    parameters a, b, and c, using a series expansion for values near bx = 0
    to maintain numerical stability.

    Input:
        x - NumPy array or array-like input containing the independent variable
        values at which the Langevin function is evaluated.
        a - Float scaling parameter controlling the magnitude of the Langevin response.
        b - Float scaling parameter multiplying x and defining the argument of
        the Langevin function.
        c - Float offset parameter added to the Langevin function output.

    Output:
        J - 2D NumPy array with shape (len(x), 3) containing the Jacobian of the
        Langevin function with respect to a, b, and c. The first column contains
        df/da, the second column contains df/db, and the third column contains df/dc.
        Series expansions are used near bx = 0 to avoid numerical instability.
    '''
    # Convert x to a NumPy array of floating-point values to ensure
    # consistent numerical operations throughout the calculation.
    x = np.asarray(x, dtype=float)

    # Calculate the argument of the Langevin function, bx = b*x.
    bx = b * x

    # Initialize the Jacobian as a two-dimensional floating-point array
    # with one row for each x value and three columns for the parameters a, b, and c.
    J = np.zeros((len(x), 3), dtype=float)

    # df/da = g(bx) = coth(bx) - 1/(bx).
    # The equivalent expression 1/tanh(bx) - 1/(bx) is used for coth(bx).
    # Separate small and large bx values to avoid numerical instability near zero.
    small = np.isclose(bx, 0.0)
    large = ~small

    # Initialize an array to store the Langevin function g(bx)
    # without the scaling parameter a or offset c.
    g = np.empty_like(bx)

    # Use the small-argument series approximation g(bx) ≈ bx/3
    # when bx is sufficiently close to zero.
    g[small] = bx[small] / 3.0

    # Evaluate the standard Langevin expression for values sufficiently
    # far from zero.
    g[large] = 1.0/np.tanh(bx[large]) - 1.0/(bx[large])

    # Store df/da in the first column of the Jacobian.
    J[:, 0] = g

    # df/dc = 1 because c is an additive constant in the Langevin function.
    J[:, 2] = 1.0

    # df/db = a*x*g'(bx), where
    # g'(z) = d/dz[coth(z) - 1/z] = -csch^2(z) + 1/z^2.
    # A separate treatment is used near bx = 0 for numerical stability.
    # For small z, g'(z) ≈ 1/3 - z^2/15 + ...
    dg_dz = np.empty_like(bx)

    # Evaluate the derivative of the Langevin function for values
    # sufficiently far from zero using the standard expression.
    # Since csch(z) = 1/sinh(z), csch^2(z) is calculated as 1/sinh(z)^2.
    dg_dz[large] = -1.0/(np.sinh(bx[large])**2) + 1.0/(bx[large]**2)

    # Use the leading term of the series expansion for g'(z) near zero.
    # This is the derivative of the approximation g(z) ≈ z/3.
    dg_dz[small] = 1.0/3.0

    # Store df/db in the second column of the Jacobian using
    # the chain rule: df/db = a*x*g'(bx).
    J[:, 1] = a * x * dg_dz

    # Return the completed Jacobian matrix.
    return J

def langevin_fit_with_yerr(data, applied_field, datapoints=None, sigma=1e-6, absolute_sigma=True, plot=False):
    """
    Fit a Langevin function to each dataset in a dictionary and calculate the
    propagated 1-sigma uncertainty of the fitted curve using the parameter covariance matrix.

    Input:
        data - Dictionary containing the measured y-values to be fit, where each key identifies
        a dataset and each corresponding value is an array of y-values.

        applied_field - 1D array-like object containing the x-values corresponding to the
        measurements in each dataset in data.

        datapoints - Optional 1D array-like object specifying the x-values at which the fitted
        Langevin curves and their uncertainties should be evaluated. If None, the original
        applied_field values are used.

        sigma - Measurement uncertainty used by curve_fit. If None, curve_fit estimates the
        parameter covariance from the residuals. If a scalar float is provided, the same
        uncertainty is applied to every data point for every dataset. If an array-like object
        is provided, it specifies the uncertainty for each x-value and is applied to all datasets.
        If a dictionary is provided, each key should correspond to a key in data and contain
        the uncertainty array for that dataset.

        absolute_sigma - Boolean specifying whether sigma represents absolute measurement
        uncertainties. If True, the supplied sigma values are treated as absolute uncertainties
        when calculating the parameter covariance matrix.

        plot - Boolean specifying whether the experimental data, Langevin fit, and propagated
        1-sigma uncertainty band should be plotted for each dataset.

    Output:
        results - Dictionary containing the fitting results for each key in data. Each key
        contains a nested dictionary with:
            params - Optimized Langevin parameters [a, b, c].
            pcov - Covariance matrix of the fitted parameters.
            param_err - 1-sigma uncertainty of each fitted parameter.
            y_fit - Langevin fit evaluated at x_out.
            y_err - Propagated 1-sigma uncertainty of the fitted Langevin curve at x_out.

        x_out - 1D NumPy array containing the x-values at which the fitted Langevin curves
        and their uncertainties were evaluated.
    """

    # Initialize an empty dictionary to store the fitting results for each dataset.
    results = {}

    # Determine the maximum measured moment for each dataset.
    # The value is used as the initial guess for the Langevin parameter a.
    max_moment_dict = {key: data[key][np.argmax(applied_field)] for key in data.keys()}

    # If no custom evaluation points are supplied, use the original applied field values.
    if datapoints is None:
        x_out = np.array(applied_field, dtype=float)

    # Otherwise, convert the supplied evaluation points to a NumPy array of floats.
    else:
        x_out = np.asarray(datapoints, dtype=float)

    # Convert the applied field values to a NumPy array of floats for fitting.
    applied_field = np.array(applied_field, dtype=float)

    # Iterate through each dataset in the input dictionary.
    for key in data.keys():

        # Convert the current dataset to a NumPy array of floating-point values.
        y = np.array(data[key], dtype=float)

        # Determine how the measurement uncertainty sigma should be assigned to this dataset.
        # If sigma is a dictionary, retrieve the uncertainty array corresponding to the current key.
        if isinstance(sigma, dict):
            sigma_key = np.asarray(sigma.get(key, None))

        # If sigma is a scalar float, create an array containing the same uncertainty
        # for every measurement in the current dataset.
        elif isinstance(sigma, float):
            sigma_key = np.full_like(y, sigma)  # same length as y

        # Otherwise, convert the supplied uncertainty array to a NumPy array.
        # If sigma is None, preserve it as None so curve_fit can estimate the covariance.
        else:
            sigma_key = np.asarray(sigma) if sigma is not None else None

        # Set the initial guesses for the three Langevin parameters.
        # a is initialized using the maximum measured moment,
        # b is initialized to a small positive value,
        # and c is initialized to zero.
        p0 = [max_moment_dict[key], 1e-4, 0.0]

        # Fit the Langevin function without supplied measurement uncertainties.
        if sigma_key is None:
            popt, pcov = curve_fit(langevin, applied_field, y, p0=p0, maxfev=10000)

        # Fit the Langevin function using the supplied measurement uncertainties.
        else:
            popt, pcov = curve_fit(langevin, applied_field, y, p0=p0, sigma=sigma_key, absolute_sigma=absolute_sigma, maxfev=10000)

        # Calculate the 1-sigma uncertainty of each fitted parameter from the diagonal
        # elements of the parameter covariance matrix.
        perr = np.sqrt(np.diag(pcov))

        # Evaluate the fitted Langevin function at all requested output x-values.
        y_fit = langevin(x_out, *popt)

        # Evaluate the Jacobian of the Langevin function with respect to the fitted parameters
        # at every x-value in x_out. The resulting array has shape (N_x, 3).
        J = langevin_jacobian(x_out, *popt)   # shape (N_x, 3)

        # Propagate the parameter covariance matrix into uncertainty in the fitted y-values.
        # For each x-value, calculate:
        # var_y = J @ pcov @ J.T
        # The einsum expression performs this calculation for every x-value simultaneously.
        var_y = np.einsum('ij,jk,ik->i', J, pcov, J)   # length N_x

        # Take the square root of the propagated variance to obtain the 1-sigma uncertainty.
        # Maximum with zero prevents numerical roundoff from producing invalid negative values.
        y_err = np.sqrt(np.maximum(var_y, 0.0))        # numerical safety

        # Store all fitting results for the current dataset.
        results[key] = {
            'params': popt,
            'pcov': pcov,
            'param_err': perr,
            'y_fit': y_fit,
            'y_err': y_err
        }

        # Optionally display the experimental data, fitted Langevin curve,
        # and propagated 1-sigma uncertainty band.
        if plot:

            # Import matplotlib only when plotting is requested.
            import matplotlib.pyplot as plt

            # Create a new figure for the current dataset.
            plt.figure(figsize=(8,6))

            # Plot the experimental data as individual points.
            plt.scatter(applied_field, y, s=15, color='blue', label='Data')

            # Plot the fitted Langevin curve.
            plt.plot(x_out, y_fit, color='red', lw=2, label='Langevin Fit')

            # Shade the region corresponding to ±1 standard deviation around the fitted curve.
            plt.fill_between(x_out, y_fit - y_err, y_fit + y_err, color='red', alpha=0.2, label='±1σ')

            # Label the x-axis with the applied magnetic field.
            plt.xlabel("Applied Field")

            # Label the y-axis with the corrected measured signal.
            plt.ylabel("Corrected Signal")

            # Display the legend identifying the data, fit, and uncertainty region.
            plt.legend()

            # Add a grid to the plot.
            plt.grid()

            # Display the completed plot.
            plt.show()

            # Unpack the optimized Langevin parameters for convenient display.
            a_fit, b_fit, c_fit = popt

            # Print the fitted parameters and their corresponding 1-sigma uncertainties.
            print(f"{key}: a={a_fit:.4g} ± {perr[0]:.4g}, b={b_fit:.4g} ± {perr[1]:.4g}, c={c_fit:.4g} ± {perr[2]:.4g}")

    # Return the fitting results for all datasets and the x-values used to evaluate the fits.
    return results, x_out

def plot_hysteresis_oneplot(data, applied_field, corrected_data=None, smoothed_field=None,
                             lower_bound=None, upper_bound=None, correction_type='corrected',
                             plots=True, set_xlim = False, ylims = None):
    '''
    Plot hysteresis curves for each key on a single figure, optionally overlay corrected or smoothed data,
    and return the field and data arrays after applying the specified bounds.

    Input:
        data - Dictionary containing magnetic moment data, where each key identifies a dataset or measurement
        and each corresponding value is an array of magnetic moment values.

        applied_field - NumPy array containing the applied magnetic field values corresponding to the
        measurements in data. The same field array is assumed to apply to all keys.

        corrected_data - Optional dictionary containing corrected or smoothed magnetic moment data.
        Keys should correspond to keys in data and values should have the same length as the associated
        field array.

        smoothed_field - Optional NumPy array containing smoothed applied field values. If provided,
        these values are used when plotting corrected_data and when filtering corrected data.

        lower_bound - Optional lower bound on applied field values. Only field values greater than
        this value are retained.

        upper_bound - Optional upper bound on applied field values. Only field values less than
        this value are retained.

        correction_type - String used as the legend label for the corrected or smoothed data.

        plots - Boolean specifying whether the hysteresis curves should be plotted. If False, the
        function only filters and returns the data.

        set_xlim - Boolean specifying whether the x-axis limits should be explicitly set using
        lower_bound and upper_bound.

        ylims - Optional tuple specifying the lower and upper limits of the y-axis.

    Output:
        filtered_field_raw - NumPy array containing the applied field values after applying the
        specified lower and upper bounds.

        filtered_data - Dictionary containing the raw magnetic moment data after applying the same
        field bounds to each dataset.

        filtered_corrected_data - Dictionary containing the corrected or smoothed magnetic moment
        data after applying the appropriate field bounds, or None if corrected_data was not provided.
    '''

    # Sort the dictionary keys so that datasets are processed in a consistent order
    keys = sorted(data.keys())

    # Determine the number of datasets being plotted
    n_keys = len(keys)

    # Build masks separately
    # Start with a mask that includes every applied field value
    mask_raw = np.ones_like(applied_field, dtype=bool)

    # If a lower bound is provided, exclude field values below or equal to it
    if lower_bound is not None:
        mask_raw &= applied_field > lower_bound

    # If an upper bound is provided, exclude field values above or equal to it
    if upper_bound is not None:
        mask_raw &= applied_field < upper_bound

    # Apply the raw-data mask to obtain the filtered applied field
    filtered_field_raw = applied_field[mask_raw]

    # Check whether a separate smoothed field was provided
    if smoothed_field is not None:
        # Create a mask initially including every smoothed field value
        mask_smoothed = np.ones_like(smoothed_field, dtype=bool)

        # If a lower bound is provided, exclude smoothed field values below or equal to it
        if lower_bound is not None:
            mask_smoothed &= smoothed_field > lower_bound

        # If an upper bound is provided, exclude smoothed field values above or equal to it
        if upper_bound is not None:
            mask_smoothed &= smoothed_field < upper_bound

        # Apply the smoothed-field mask to obtain the filtered smoothed field
        filtered_field_smoothed = smoothed_field[mask_smoothed]
    else:
        # If no smoothed field was provided, indicate that no separate filtered field exists
        filtered_field_smoothed = None

    # Prepare outputs
    # Initialize the dictionary that will store the filtered raw data
    filtered_data = {}

    # Initialize the corrected-data dictionary only if corrected data were provided
    filtered_corrected_data = {} if corrected_data is not None else None

    # Process each dataset individually
    for key in keys:
        # Raw data
        # Apply the raw field mask to the magnetic moment data for the current key
        y = data[key][mask_raw]

        # Store the filtered raw data using the same key
        filtered_data[key] = y

        # Corrected/smoothed data
        # Only process corrected data if it was provided and contains the current key
        if corrected_data is not None and key in corrected_data:
            # If a separate smoothed field was provided, use its corresponding mask
            if filtered_field_smoothed is not None:
                y_corr = corrected_data[key][mask_smoothed]
            # Otherwise, use the raw-field mask
            else:
                y_corr = corrected_data[key][mask_raw]

            # Store the filtered corrected data using the same key
            filtered_corrected_data[key] = y_corr

    # Only plot if requested
    if plots:

        # Color mapping
        # Create a normalization object based on the range of dataset keys
        norm = Normalize(vmin=min(keys), vmax=max(keys))

        # Override the normalization range so that colors correspond to a fixed 0–180 range
        norm = Normalize(vmin = 0, vmax = 180)

        # Use the viridis colormap to assign colors to the different datasets
        cmap = viridis
        
        # Create the figure used to display all hysteresis curves
        plt.figure(figsize = (10,8))

        # Plot each dataset
        for idx, key in enumerate(keys):

            # Determine the color associated with the current key
            color = cmap(norm(key))

            # Plot the raw magnetic moment data against the filtered applied field
            plt.plot(filtered_field_raw, filtered_data[key], label="Raw", color=color, zorder = 0)

            # Plot corrected data if it was provided and contains the current key
            if corrected_data is not None and key in filtered_corrected_data:
                # Use the smoothed field if one was provided
                if filtered_field_smoothed is not None:
                    plt.plot(filtered_field_smoothed, filtered_corrected_data[key],
                            label=correction_type, color="black", linestyle="--", linewidth = 3, zorder = 1)
                # Otherwise, plot corrected data against the raw applied field
                else:
                    plt.plot(filtered_field_raw, filtered_corrected_data[key],
                            label=correction_type, color="black", linestyle="--", linewidth = 3, zorder = 1)

            # Add a horizontal reference line at zero magnetic moment
            plt.axhline(0, color="red", linestyle=":", linewidth=1)

            # Add a vertical reference line at zero applied field
            plt.axvline(0, color="red", linestyle=":", linewidth=1)

            #plt.set_title(f"Angle = {key}˚")

            # Label the x-axis with the applied magnetic field
            plt.xlabel("Applied Field (mT)")

            # Label the y-axis with the magnetic moment
            plt.ylabel("Magnetic Moment (emu)")

            # Display the legend in the lower-right corner
            plt.legend(loc="lower right")

            # Set the x-axis limits if requested
            if set_xlim:
                plt.xlim(lower_bound, upper_bound)

            # Set the y-axis limits if they were provided
            if ylims is not None:
                plt.ylim(ylims)

        # Display the completed hysteresis plot
        plt.show()

    # Return the filtered field and data arrays
    return filtered_field_raw, filtered_data, filtered_corrected_data

def plot_specific_moment_err360(data, applied_field, error, field_value, ylabel = None, plot = True, cmap = viridis):
    '''
    Extracts the magnetic moment and corresponding measurement error at a
    specified applied-field value for each measurement angle. The function
    identifies the array index corresponding to the requested field value,
    stores the magnetic moment and error for each angle in dictionaries, and
    optionally plots the resulting values with error bars as a function of
    measurement angle.

    Inputs:
        data (dict): A dictionary in which each key represents a measurement
            angle and each value is a NumPy array containing the corresponding
            magnetic moment data.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values corresponding to the measurements in data.
            The array is assumed to be sorted for use with np.searchsorted.
        error (dict): A dictionary with the same keys as data, where each value
            is a NumPy array containing the measurement uncertainty associated
            with the corresponding magnetic moment data.
        field_value (int or float): The applied magnetic field value at which
            the magnetic moment and error should be extracted.
        ylabel (str or None): Optional label for the y-axis. This parameter is
            currently not used because the corresponding plotting code is
            commented out. Defaults to None.
        plot (bool): Determines whether the extracted magnetic moments and
            uncertainties are plotted. Defaults to True.
        cmap (matplotlib.colors.Colormap): A Matplotlib colormap used to assign
            colors to the data points based on their measurement angles.
            Defaults to the `viridis` colormap.

    Outputs:
        moment_dict (dict): A dictionary mapping each measurement angle to the
            magnetic moment measured at the applied field closest to
            field_value.
        moment_err_dict (dict): A dictionary mapping each measurement angle to
            the corresponding measurement uncertainty at the selected applied
            field.
    '''

    keys = sorted(data.keys()) #Sort the measurement-angle keys from smallest to largest

    norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization function that maps the minimum and maximum angles to the bounds of the colormap

    cmap = cmap #Assign the provided colormap to the local cmap variable

    moment_dict = {} #Initialize an empty dictionary to store the magnetic moment at the selected field for each angle

    moment_err_dict = {} #Initialize an empty dictionary to store the magnetic moment uncertainty at the selected field for each angle



    i = 0 #Initialize a counter for the number of measurement angles processed

    if plot: #Check whether plotting has been requested

        plt.figure(figsize = (10,8)) #Create a new Matplotlib figure with a width of 10 inches and height of 8 inches

    for key in data.keys(): #Loop through each measurement angle in the data dictionary

        color = cmap(norm(key)) #Convert the current measurement angle into a normalized value and use it to select a color from the colormap

        idx = np.searchsorted(applied_field, field_value, side='left') #Find the index of the first applied-field value that is greater than or equal to the requested field value

        if idx >= len(applied_field):  # handle case where field_value is beyond range
            idx = len(applied_field) - 1 #Set the index to the final available measurement if the requested field is beyond the available field range

        mom = data[key][idx] #Extract the magnetic moment for the current angle at the selected applied-field index

        mom_err = error[key][idx] #Extract the magnetic moment uncertainty for the current angle at the selected applied-field index

        if plot: #Check whether the extracted value should be plotted

            #plt.scatter(key, mom, label = str(key), color = color, linewidths=2, edgecolors='black', s  = 100)

            plt.errorbar(key, mom, yerr = mom_err, label = str(key), fmt = 'o', markersize = 8, linewidth = 3, color = color,
                        elinewidth = 2, capsize = 4, ecolor = 'black', markeredgecolor='black', markeredgewidth=1) #Plot the magnetic moment at the current angle with its associated uncertainty as an error bar

            #plt.legend(loc = 'best')

            #plt.xlabel('Measured Angle')

            plt.xticks(np.linspace(0,360, 13)) #Set the x-axis tick locations to 13 evenly spaced angles from 0 to 360 degrees

            # if not ylabel:

            #     plt.ylabel('Magnetic Moment (emu)')

            # else:

            #     plt.ylabel(ylabel)

            #plt.title(f'Moment at {field_value} mT')

        moment_dict[key] = mom #Store the extracted magnetic moment in the dictionary using the measurement angle as the key

        moment_err_dict[key] = mom_err #Store the extracted magnetic moment uncertainty in the dictionary using the measurement angle as the key

        i += 1 #Increment the counter after processing the current measurement angle

    if plot: #Check whether plotting has been requested

        #plt.plot(moment_dict.keys(),moment_dict.values())

        plt.show() #Display the completed magnetic moment versus angle plot



    return moment_dict, moment_err_dict #Return dictionaries containing the magnetic moments and their corresponding uncertainties at the selected field

def langevin_y_data360(y_data, applied_field, lower_transition=-100, higher_transition=100, lang_datapoints=None, y_error = 1e-6):
    '''
    Processes Y-direction magnetic moment data for multiple measurement angles
    and fits the data within a specified low-field transition region using a
    Langevin model. The function first reverses the measurement arrays to
    maintain a consistent field ordering, truncates the data to the specified
    field range, performs Langevin fitting with measurement uncertainty, and
    then sorts and removes duplicate field values before returning the fitted
    data and associated uncertainties.

    Inputs:
        y_data (dict): A dictionary in which each key represents a measurement
            angle and each value is a NumPy array containing Y-direction
            magnetic moment data corresponding to applied_field.
        applied_field (numpy.ndarray): A NumPy array containing the applied
            magnetic field values corresponding to the measurements in y_data.
        lower_transition (int or float): Lower applied-field boundary defining
            the region used for the Langevin fit. Defaults to -100.
        higher_transition (int or float): Upper applied-field boundary defining
            the region used for the Langevin fit. Defaults to 100.
        lang_datapoints (numpy.ndarray or None): Optional array of applied-field
            values at which the Langevin fit is evaluated. If None, the
            truncated low-field applied-field values are used. Defaults to None.
        y_error (int or float): Measurement uncertainty supplied to the
            Langevin fitting function. Defaults to 1e-6.

    Outputs:
        unique_ys (dict): A dictionary mapping each measurement angle to its
            fitted Y-direction magnetic moment values evaluated at the unique
            applied-field points.
        unique_fields (numpy.ndarray): A sorted NumPy array containing the
            unique applied-field values associated with the fitted data.
        unique_errs (dict): A dictionary mapping each measurement angle to the
            uncertainty in the corresponding fitted Y-direction magnetic
            moment values.
    '''

    # Reverse data for consistency
    for key in y_data.keys(): #Loop through each measurement angle in the Y-direction data dictionary
        y_data[key] = y_data[key][::-1] #Reverse the Y-direction data array for the current angle so that its ordering matches the reversed applied-field array
    applied_field = applied_field[::-1] #Reverse the applied-field array so that its ordering matches the reversed Y-direction data

    print(f'Smoothing Moments between B={lower_transition} and B={higher_transition}') #Print the applied-field range over which the Langevin smoothing will be performed
    truncated_y_lowfield, truncated_lowfield = truncate_data(y_data, applied_field, lower_bound=lower_transition, upper_bound=higher_transition) #Restrict the Y-direction data and applied-field values to the specified low-field transition region

    # Use fallback: applied_lowfield if lang_datapoints not provided
    if lang_datapoints is None: #Check whether specific applied-field values for the Langevin fit have been provided
        lang_datapoints = truncated_lowfield #Use the truncated low-field measurements as the Langevin evaluation points if none were provided

    y_lang_params_lowfield, applied_lowfield = langevin_fit_with_yerr(truncated_y_lowfield, truncated_lowfield, datapoints=lang_datapoints, sigma=y_error, absolute_sigma=True, plot=False) #Fit the truncated Y-direction data using the Langevin model and calculate the fitted values and uncertainties
    y_lang_lowfield_data = {key:y_lang_params_lowfield[key]['y_fit'] for key in y_lang_params_lowfield.keys()} #Extract the fitted Y-direction magnetic moment values for each measurement angle
    y_lang_lowfield_err = {key:y_lang_params_lowfield[key]['y_err'] for key in y_lang_params_lowfield.keys()} #Extract the uncertainties associated with the fitted Y-direction magnetic moment values for each measurement angle

    combined_dict = {} #Initialize an empty dictionary to store the sorted Langevin-fitted Y-direction data
    err_dict = {} #Initialize an empty dictionary to store the sorted Langevin-fit uncertainties

    sort_idx = np.argsort(lang_datapoints) #Determine the indices that sort the Langevin evaluation field values from smallest to largest
    combined_fields = lang_datapoints[sort_idx] #Sort the Langevin evaluation field values using the calculated sorting indices
    for key in y_lang_lowfield_data: #Loop through each measurement angle in the fitted Y-direction data
        combined_dict[key] = y_lang_lowfield_data[key][sort_idx] #Reorder the fitted Y-direction data using the same field-sorting indices
        err_dict[key] = y_lang_lowfield_err[key][sort_idx] #Reorder the fitted uncertainties using the same field-sorting indices

    unique_ys = {} #Initialize an empty dictionary to store fitted Y-direction data with duplicate field values removed
    unique_errs = {} #Initialize an empty dictionary to store fitted uncertainties with duplicate field values removed
    for key, vec in combined_dict.items(): #Loop through each measurement angle and its sorted fitted Y-direction data
        _, idx = np.unique(combined_fields, return_index=True) #Find the indices corresponding to the first occurrence of each unique applied-field value
        unique_ys[key] = vec[idx] #Use the unique-field indices to remove duplicate field measurements from the fitted Y-direction data
        unique_errs[key] = err_dict[key][idx] #Use the same unique-field indices to remove duplicate field measurements from the fitted uncertainties
    _, idx = np.unique(combined_fields, return_index=True) #Find the indices corresponding to the first occurrence of each unique applied-field value
    unique_fields = combined_fields[idx] #Create the final applied-field array containing only unique field values

    return unique_ys, unique_fields, unique_errs #Return the unique fitted Y-direction data, unique applied-field values, and corresponding uncertainties

def saturation_and_mass(x_data, last_ind = 2, plot = False, applied_field = None):
    '''
    Calculates the average saturation value from the upper and lower extrema
    of X-direction magnetic moment data across multiple measurement angles,
    and uses the resulting saturation value to estimate magnetic mass. The
    function optionally plots the sorted magnetic moment data and highlights
    the points used to calculate the upper and lower saturation values.

    Inputs:
        x_data (dict): A dictionary in which each key represents a measurement
            angle and each value is a NumPy array containing X-direction
            magnetic moment measurements.
        last_ind (int): Number of measurements taken from each end of the
            sorted data to use when calculating the upper and lower average
            saturation values. Defaults to 2.
        plot (bool): Determines whether a plot is generated for the dataset
            corresponding to angle 0. Defaults to False.
        applied_field (numpy.ndarray or None): A NumPy array containing the
            applied magnetic field values corresponding to the X-direction
            measurements. Required when plot is True. Defaults to None.

    Outputs:
        average_satval (float): The average saturation magnetic moment,
            calculated as the mean of the average upper saturation value and
            the absolute value of the average lower saturation value.
        mass (float): An estimated magnetic mass calculated by dividing the
            average saturation value by 63.
    '''

    top_av_list = [] #Initialize an empty list to store the average upper saturation value for each measurement angle
    bottom_av_list = [] #Initialize an empty list to store the average lower saturation value for each measurement angle
    for i in x_data.keys(): #Loop through each measurement angle in the X-direction data dictionary
        sort_arr = np.sort(x_data[i]) #Sort the X-direction magnetic moment values for the current angle from smallest to largest
        top_values = sort_arr[-last_ind:] #Extract the specified number of largest magnetic moment values
        bottom_values = sort_arr[:last_ind] #Extract the specified number of smallest magnetic moment values
        top_av_list.append(np.mean(top_values)) #Calculate and store the mean of the largest magnetic moment values
        bottom_av_list.append(np.mean(bottom_values)) #Calculate and store the mean of the smallest magnetic moment values
        if plot: #Check whether plotting has been requested
            if i == 0: #Only generate the diagnostic plot for the dataset corresponding to angle 0
                sorts = np.argsort(x_data[i]) #Determine the indices that sort the X-direction magnetic moment values from smallest to largest
                sorted_fields = applied_field[sorts] #Reorder the applied-field values using the same sorting indices as the magnetic moment data
                plt.scatter(sorted_fields,sort_arr, color = 'blue') #Plot all sorted magnetic moment values against their corresponding applied-field values
                plt.scatter(sorted_fields[-last_ind:], top_values, color = 'orange') #Highlight the upper saturation points used in the calculation
                plt.scatter(sorted_fields[:last_ind], bottom_values, color = 'orange') #Highlight the lower saturation points used in the calculation
                plt.hlines([np.mean(top_values),np.mean(bottom_values)],xmin = np.min(sorted_fields), xmax = np.max(sorted_fields)) #Draw horizontal lines representing the average upper and lower saturation values
                plt.show() #Display the diagnostic saturation plot
    top_val = np.mean(top_av_list) #Calculate the mean upper saturation value across all measurement angles
    bottom_val = np.mean(bottom_av_list) #Calculate the mean lower saturation value across all measurement angles
    average_satval = np.mean([top_val, np.abs(bottom_val)]) #Calculate the average saturation magnitude from the upper saturation value and absolute lower saturation value
    mass = average_satval/63 #Estimate the magnetic mass by dividing the average saturation value by 63
    return average_satval, mass #Return the average saturation value and estimated magnetic mass

def plot_max_Torque(x_fit, y_fit, fields, smoothed_fields, x_err, y_err, field_min = -100, field_max = 100, plots = False, title = None, torque_plots = False):
    '''
    Calculates the maximum torque and associated uncertainty at each specified
    applied magnetic field using fitted X- and Y-direction magnetic moment data.
    Optionally plots the maximum torque as a function of applied magnetic field.

    Inputs:
        x_fit (dict): Dictionary containing the fitted X-direction magnetic
            moment data for each measurement angle.
        y_fit (dict): Dictionary containing the fitted Y-direction magnetic
            moment data for each measurement angle.
        fields (iterable): Collection of applied magnetic field values at which
            the maximum torque should be calculated.
        smoothed_fields (numpy.ndarray): Applied magnetic field values
            corresponding to the fitted magnetic moment data.
        x_err (dict): Dictionary containing the uncertainties associated with
            the fitted X-direction magnetic moment data.
        y_err (dict): Dictionary containing the uncertainties associated with
            the fitted Y-direction magnetic moment data.
        field_min (float): Lower magnetic field boundary used when calculating
            the maximum torque.
        field_max (float): Upper magnetic field boundary used when calculating
            the maximum torque.
        plots (bool): Controls whether plots are generated within the
            max_Torque function.
        title (str or None): Optional title for the maximum torque versus field
            plot. If None, a default title is used.
        torque_plots (bool): Controls whether a plot of maximum torque versus
            applied magnetic field is generated.

    Outputs:
        T_max_dict (dict): Dictionary mapping each applied magnetic field to
            its calculated maximum torque.
        T_err_dict (dict): Dictionary mapping each applied magnetic field to
            the uncertainty in the calculated maximum torque.
    '''

    T_max_dict= {} #Initialize a dictionary to store the maximum torque calculated at each applied magnetic field
    T_err_dict = {} #Initialize a dictionary to store the uncertainty associated with the maximum torque at each applied magnetic field
    for field in fields: #Loop through each specified applied magnetic field value
        T_max, T_max_err = max_Torque(x_fit, y_fit, field, smoothed_fields, x_err, y_err, field_min = -100, field_max = 100, plots = False) #Calculate the maximum torque and its uncertainty at the current applied magnetic field
        T_max_dict[field] = T_max #Store the calculated maximum torque using the applied magnetic field as the dictionary key
        T_err_dict[field] = T_max_err #Store the calculated maximum torque uncertainty using the applied magnetic field as the dictionary key

    if torque_plots: #Generate a maximum torque versus applied magnetic field plot if requested
        keys = list(T_max_dict.keys()) #Create a list containing the applied magnetic field values used as keys in the maximum torque dictionary
        norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization object to map the applied magnetic field values to the range of the colormap
        cmap = viridis #Set the colormap used to distinguish different applied magnetic field values
        plt.figure(figsize= (10,8)) #Create a new figure with a width of 10 inches and height of 8 inches
        plt.plot(keys,[T_max_dict[key] for key in keys],linestyle = '--',color = 'black', zorder = 0) #Plot maximum torque as a function of applied magnetic field using a dashed connecting line
        for i in range(len(keys)): #Loop through each applied magnetic field value to plot the individual maximum torque measurements
            color = cmap(norm(keys[i])) #Determine the plotting color corresponding to the current applied magnetic field value
            plt.scatter(keys[i],T_max_dict[keys[i]], color=color, linewidths=2, edgecolors='black', s=100) #Plot the maximum torque at the current applied magnetic field using the corresponding colormap color
        
        plt.xlabel('Field(mT)') #Label the x-axis with the applied magnetic field
        plt.ylabel('Maximum Torque (N*m)') #Label the y-axis with the maximum torque and its units
        if title is not None: #Check whether a custom plot title was provided
            plt.title(title) #Set the plot title to the user-provided title
        else: #Use the default title if no custom title was provided
            plt.title('Torque vs Field') #Set the plot title to the default torque-versus-field description
        plt.grid(alpha=0.3) #Add a partially transparent grid to improve readability of the plot
        plt.show() #Display the maximum torque versus applied magnetic field plot

    return T_max_dict, T_err_dict #Return the maximum torque values and their corresponding uncertainties

def plot_specific_moment_err(data, applied_field, error, field_value, ylabel = None, plot = True):
    """
    Extracts the magnetic moment and associated uncertainty at a specified
    applied-field value for each measurement angle and optionally plots the
    resulting moments with error bars.

    Inputs:
        data (dict): Dictionary mapping each measurement angle to an array
            of magnetic moment values corresponding to the applied-field
            values.
        applied_field (np.ndarray): Array of applied magnetic field values
            corresponding to the magnetic moment data.
        error (dict): Dictionary mapping each measurement angle to an array
            of uncertainties corresponding to the magnetic moment values.
        field_value (float): Applied magnetic field value at which the
            magnetic moment and uncertainty are evaluated.
        ylabel (str or None): Optional label for the y-axis. If None, the
            default label 'Magnetic Moment (emu)' is used.
        plot (bool): Determines whether the extracted moments and their
            uncertainties are plotted.

    Outputs:
        moment_dict (dict): Dictionary mapping each measurement angle to
            its magnetic moment at the specified applied-field value.
        moment_err_dict (dict): Dictionary mapping each measurement angle to
            the uncertainty in its magnetic moment at the specified
            applied-field value.
    """
    keys = sorted(data.keys()) #Sort the measurement-angle keys from smallest to largest
    norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization object that maps the measurement angles to the range used by the colormap
    cmap = viridis #Set the colormap used to distinguish measurements taken at different angles
    moment_dict = {} #Initialize a dictionary for storing the magnetic moment at the specified field for each angle
    moment_err_dict = {} #Initialize a dictionary for storing the magnetic moment uncertainty at the specified field for each angle

    i = 0 #Initialize an index counter for iterating through the measurement angles
    if plot: #Create the figure only if plotting is enabled
        plt.figure(figsize = (10,8)) #Create a figure with a width of 10 inches and a height of 8 inches
    for key in data.keys(): #Loop through each measurement angle in the data dictionary
        color = cmap(norm(key)) #Assign a colormap color based on the current measurement angle
        idx = np.searchsorted(applied_field, field_value, side='left') #Find the index corresponding to the specified applied-field value
        if idx >= len(applied_field):  #Check whether the requested field value lies beyond the available applied-field range
            idx = len(applied_field) - 1 #Use the final available data point if the requested field is beyond the available range
        mom = data[key][idx] #Extract the magnetic moment at the selected applied-field index for the current measurement angle
        mom_err = error[key][idx] #Extract the uncertainty in the magnetic moment at the selected applied-field index
        if plot: #Add the extracted magnetic moment and uncertainty to the plot if plotting is enabled
            #plt.scatter(key, mom, label = str(key), color = color, linewidths=2, edgecolors='black', s  = 100) #Optional scatter-plot representation of the magnetic moment
            plt.errorbar(key, mom, yerr = mom_err, xerr = 5, label = str(key), fmt = 'o', markersize = 12, linewidth = 5, color = color,
                         elinewidth = 2, capsize = 5, ecolor = 'black', markeredgecolor='black', markeredgewidth=2) #Plot the magnetic moment with vertical and horizontal error bars
            plt.legend(loc = 'best') #Display a legend using the location that best avoids overlapping the plotted data
            plt.xlabel('Measured Angle') #Label the x-axis with the measured sample angle
            plt.xticks(list(data.keys())) #Set the x-axis tick marks to the measurement angles contained in the data dictionary
            if not ylabel: #Check whether a custom y-axis label was provided
                plt.ylabel('Magnetic Moment (emu)') #Use the default magnetic moment label if no custom label was provided
            else: #Use the user-provided y-axis label when one is available
                plt.ylabel(ylabel) #Set the y-axis label to the supplied label
            plt.title(f'Moment at {field_value} mT') #Set the plot title to indicate the applied-field value being evaluated
        moment_dict[key] = mom #Store the magnetic moment for the current angle in the output dictionary
        moment_err_dict[key] = mom_err #Store the magnetic moment uncertainty for the current angle in the output dictionary
        i += 1 #Increment the iteration counter
    if plot: #Display the completed plot if plotting is enabled
        plt.show() #Render the figure

    return moment_dict, moment_err_dict #Return the magnetic moments and their corresponding uncertainties for all measurement angles

def calculate_theta_err(x_moment_dict, y_moment_dict, y_err, x_err, field_value, plot=True, ylabel=None, offset=0):
    """
    Calculates the magnetic moment angle and its propagated uncertainty from
    the X- and Y-direction magnetic moment components and optionally plots the
    resulting angles with error bars.

    Inputs:
        x_moment_dict (dict): Dictionary mapping each measurement angle to
            its X-direction magnetic moment.
        y_moment_dict (dict): Dictionary mapping each measurement angle to
            its Y-direction magnetic moment.
        y_err (dict): Dictionary mapping each measurement angle to the
            uncertainty in its Y-direction magnetic moment.
        x_err (dict): Dictionary mapping each measurement angle to the
            uncertainty in its X-direction magnetic moment.
        field_value (float): Applied magnetic field value at which the
            magnetic moment angle is being evaluated.
        plot (bool): Determines whether the calculated angles and their
            uncertainties are plotted.
        ylabel (str or None): Optional label for the y-axis. If None, the
            default label 'Magnetic Moment Angle' is used.
        offset (float): Angular offset in degrees to subtract from each
            calculated magnetic moment angle.

    Outputs:
        theta_dict (dict): Dictionary mapping each measurement angle to the
            calculated magnetic moment angle in degrees.
        theta_err_dict (dict): Dictionary mapping each measurement angle to
            the propagated uncertainty in the magnetic moment angle in
            degrees.
    """
    theta_dict = {} #Initialize a dictionary for storing the calculated magnetic moment angle at each measurement angle
    theta_err_dict = {} #Initialize a dictionary for storing the uncertainty in the calculated magnetic moment angle at each measurement angle

    for key in x_moment_dict: #Loop through each measurement angle in the X-direction moment dictionary
        x = x_moment_dict[key] #Extract the X-direction magnetic moment for the current measurement angle
        y = y_moment_dict[key] #Extract the Y-direction magnetic moment for the current measurement angle
        sx = x_err[key] #Extract the uncertainty in the X-direction magnetic moment
        sy = y_err[key] #Extract the uncertainty in the Y-direction magnetic moment

        # angle in radians
        theta_rad = np.arctan2(y, x)   #Calculate the magnetic moment angle in radians using the X- and Y-direction components
        theta_deg = np.rad2deg(theta_rad) - offset #Convert the angle to degrees and subtract the specified angular offset

        # error propagation (in radians first)
        denom = x**2 + y**2 #Calculate the squared magnitude of the magnetic moment vector, which appears in the derivatives used for uncertainty propagation
        dtheta_dx = -y / denom #Calculate the partial derivative of the magnetic moment angle with respect to the X-direction moment
        dtheta_dy =  x / denom #Calculate the partial derivative of the magnetic moment angle with respect to the Y-direction moment
        sigma_theta_rad = np.sqrt((dtheta_dx * sx)**2 + (dtheta_dy * sy)**2) #Propagate the X- and Y-direction moment uncertainties to determine the angular uncertainty in radians

        # convert error to degrees
        sigma_theta_deg = np.rad2deg(sigma_theta_rad) #Convert the propagated angular uncertainty from radians to degrees

        theta_dict[key] = theta_deg #Store the calculated magnetic moment angle for the current measurement angle
        theta_err_dict[key] = sigma_theta_deg #Store the calculated angular uncertainty for the current measurement angle

    # ---- Plotting ----
    if plot: #Generate a plot of the calculated magnetic moment angles if plotting is enabled
        keys = sorted(theta_dict.keys()) #Sort the measurement-angle keys from smallest to largest
        norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization object that maps measurement angles to the range used by the colormap
        cmap = viridis #Set the colormap used to distinguish measurements taken at different angles

        plt.figure(figsize=(10,8)) #Create a figure with a width of 10 inches and a height of 8 inches
        for key in keys: #Loop through each measurement angle to plot its calculated magnetic moment angle
            color = cmap(norm(key)) #Assign a colormap color based on the current measurement angle
            plt.errorbar(key, theta_dict[key], yerr=theta_err_dict[key], xerr = 5, label = str(key), fmt = 'o', markersize = 12, linewidth = 5, color = color,
                         elinewidth = 2, capsize = 5, ecolor = 'black', markeredgecolor='black', markeredgewidth=2) #Plot the magnetic moment angle with horizontal and vertical error bars
        plt.legend([str(k) for k in keys], loc='best') #Display a legend identifying the measurement angle associated with each plotted point
        plt.xlabel('Measured Angle') #Label the x-axis with the measured sample angle
        plt.xticks(keys) #Set the x-axis tick marks to the measurement angles contained in the data
        plt.ylabel(ylabel if ylabel else 'Magnetic Moment Angle') #Set the y-axis label to the provided label or use the default magnetic moment angle label
        plt.title(f'Angle at {field_value} mT') #Set the plot title to indicate the applied magnetic field value being evaluated

    return theta_dict, theta_err_dict #Return the calculated magnetic moment angles and their corresponding uncertainties

def calculate_torque_err(moment_dict_x, moment_dict_y, x_err, y_err, field_value=None, B_vec=None, plot=True, ylabel=None, y_min = None, y_max = None):
    """
    Calculates the magnetic torque from the cross product of the magnetic
    moment and applied magnetic field vectors and propagates the uncertainty
    from the X- and Y-direction magnetic moment measurements.

    Inputs:
        moment_dict_x (dict): Dictionary mapping each measurement angle to
            its X-direction magnetic moment in emu.
        moment_dict_y (dict): Dictionary mapping each measurement angle to
            its Y-direction magnetic moment in emu.
        x_err (dict or scalar): Dictionary mapping each measurement angle to
            the uncertainty in the X-direction magnetic moment, or a single
            scalar uncertainty applied to all measurements, in emu.
        y_err (dict or scalar): Dictionary mapping each measurement angle to
            the uncertainty in the Y-direction magnetic moment, or a single
            scalar uncertainty applied to all measurements, in emu.
        field_value (float or None): Applied magnetic field in mT. When
            B_vec is not provided, this value is interpreted as the X
            component of the magnetic field.
        B_vec (array-like or None): Three-component magnetic field vector
            [Bx, By, Bz] in mT. If provided, this takes precedence over
            field_value.
        plot (bool): Determines whether the calculated torque and its
            uncertainty are plotted.
        ylabel (str or None): Optional label for the y-axis. If None, the
            default label 'Magnetic Torque (N·m)' is used.
        y_min (float or None): Optional lower limit for the y-axis.
        y_max (float or None): Optional upper limit for the y-axis.

    Outputs:
        T_dict (dict): Dictionary mapping each measurement angle to the
            calculated Z-component of magnetic torque in N·m.
        Terr_dict (dict): Dictionary mapping each measurement angle to the
            one-standard-deviation uncertainty in the Z-component of magnetic
            torque in N·m.
    """

    # Keys and ordering
    keys = sorted(moment_dict_x.keys()) #Sort the measurement-angle keys to ensure that all moment and uncertainty arrays use the same ordering

    # Convert moments and errors from emu to A·m^2 (1 emu = 1e-3 A·m^2)
    mx = np.array([moment_dict_x[k] for k in keys], dtype=float) * 1e-3 #Extract the X-direction magnetic moments and convert them from emu to A·m^2
    my = np.array([moment_dict_y[k] for k in keys], dtype=float) * 1e-3 #Extract the Y-direction magnetic moments and convert them from emu to A·m^2

    # x_err / y_err may be dicts or scalars
    def to_err_array(err_input, keys):
        #Convert either a dictionary or scalar uncertainty into an array with the same ordering as the measurement-angle keys
        if isinstance(err_input, dict): #Check whether the supplied uncertainty is stored in a dictionary
            return np.array([err_input[k] for k in keys], dtype=float) * 1e-3 #Extract the uncertainties in key order and convert them from emu to A·m^2
        else:
            # scalar
            return np.full(len(keys), float(err_input), dtype=float) * 1e-3 #Create an array containing the same scalar uncertainty for every measurement and convert it from emu to A·m^2

    mx_err = to_err_array(x_err, keys) #Convert the X-direction moment uncertainties into an ordered array in A·m^2
    my_err = to_err_array(y_err, keys) #Convert the Y-direction moment uncertainties into an ordered array in A·m^2

    # Build magnetic field vector in Tesla
    if B_vec is not None: #Use the explicitly provided three-dimensional magnetic field vector when available
        B_vec = np.asarray(B_vec, dtype=float) #Convert the supplied magnetic field vector to a NumPy array
        if B_vec.size != 3: #Check that the supplied magnetic field vector contains exactly three components
            raise ValueError("B_vec must be length-3 (Bx, By, Bz) in mT.") #Raise an error if the magnetic field vector does not contain three components
        B = B_vec * 1e-3 #Convert the magnetic field vector from mT to Tesla
    elif field_value is not None: #Use the supplied scalar field value when no full field vector was provided
        # previous default geometry: field along +x
        B = np.array([field_value * 1e-3, 0.0, 0.0], dtype=float) #Construct a magnetic field vector pointing entirely in the positive X direction and convert it from mT to Tesla
    else: #Handle the case where neither a full field vector nor a scalar field value was supplied
        raise ValueError("Either field_value or B_vec must be provided.") #Raise an error because the magnetic field is required to calculate torque

    Bx, By, Bz = B #Extract the X-, Y-, and Z-components of the magnetic field vector
    Tz = mx * By - my * Bx  #Calculate the Z-component of the magnetic torque for every measurement angle

    Var_Tz = (By**2) * (mx_err**2) + (Bx**2) * (my_err**2) #Propagate the independent X- and Y-direction moment uncertainties to calculate the variance of the Z-component of torque
    Tz_err = np.sqrt(Var_Tz) #Take the square root of the torque variance to obtain the one-standard-deviation torque uncertainty

    # Prepare dictionary outputs in N·m units (mx,my were converted from emu -> A·m^2; B in T)
    T_dict = {k: float(Tz[i]) for i, k in enumerate(keys)} #Store the calculated Z-component of torque for each measurement angle in a dictionary
    Terr_dict = {k: float(Tz_err[i]) for i, k in enumerate(keys)} #Store the calculated torque uncertainty for each measurement angle in a dictionary

    # Optional plotting (scatter with y-error bars)
    if plot: #Generate a torque plot if plotting is enabled
        norm = Normalize(vmin=min(keys), vmax=max(keys)) #Create a normalization object that maps the measurement angles to the range used by the colormap
        cmap = viridis #Set the colormap used to distinguish measurements taken at different angles
        plt.figure(figsize=(10, 8)) #Create a figure with a width of 10 inches and a height of 8 inches
        for i, k in enumerate(keys): #Loop through each measurement angle to plot its calculated torque
            color = cmap(norm(k)) #Assign a colormap color based on the current measurement angle
            plt.errorbar(k, Tz[i],yerr=Tz_err[i], xerr = 5, fmt = 'o', markersize = 12, linewidth = 5, color = color,
                         elinewidth = 2, capsize = 5, ecolor = 'black', markeredgecolor='black', markeredgewidth=2) #Plot the torque with horizontal and vertical error bars
        plt.legend([str(k) for k in keys], loc='best') #Display a legend identifying the measurement angle associated with each plotted point
        plt.xlabel('Measured Angle') #Label the x-axis with the measured sample angle
        plt.xticks(keys) #Set the x-axis tick marks to the measurement angles contained in the data
        if y_min is not None: #Check whether a lower y-axis limit was provided
            plt.ylim([y_min,y_max]) #Set the y-axis limits using the supplied lower and upper bounds
        plt.ylabel(ylabel if ylabel else 'Magnetic Torque (N·m)') #Set the y-axis label to the provided label or use the default magnetic torque label
        title_B = f"B = [{Bx:.3e}, {By:.3e}, {Bz:.3e}] T" if B_vec is not None else f"B_x = {Bx:.3e} T" #Create a plot title describing the applied magnetic field vector or X-direction field
        plt.title(f'Torque at {title_B}') #Set the plot title to indicate the magnetic field used for the torque calculation
        plt.grid(True) #Display a grid to improve readability of the plot
        plt.show() #Render the completed torque plot

    return T_dict, Terr_dict #Return the calculated torque values and their corresponding uncertainties

def ridge_solution(X, y, alpha=1e-8):
    """
    Calculates the ridge regression solution for a linear model by adding
    L2 regularization to the ordinary least-squares solution.

    Inputs:
        X (np.ndarray): Design matrix containing the predictor variables.
        y (np.ndarray): Array containing the observed response values.
        alpha (float): Ridge regularization strength. Larger values impose
            stronger regularization and reduce the influence of correlated
            predictors.

    Outputs:
        np.ndarray: Array containing the fitted regression coefficients
            obtained using ridge regularization.
    """
    XT_X = X.T @ X #Calculate the matrix product of the transpose of the design matrix and the design matrix
    p = XT_X.shape[0] #Determine the number of predictor variables, corresponding to the number of rows and columns in X^T X
    return np.linalg.inv(XT_X + alpha*np.eye(p)) @ (X.T @ y) #Add the ridge regularization term to X^T X, invert the resulting matrix, and multiply by X^T y to calculate the regularized regression coefficients

def residuals_ab(params, X, y):
    """
    Calculates the residuals between the predicted and observed values for
    a two-parameter linear model.

    Inputs:
        params (array-like): Array containing the model coefficients A and B.
        X (np.ndarray): Design matrix containing the predictor variables.
        y (np.ndarray): Array containing the observed response values.

    Outputs:
        np.ndarray: Array of residuals calculated as the predicted values
            minus the observed values.
    """
    A, B = params #Extract the two model coefficients, A and B, from the parameter array
    pred = X @ np.array([A, B]) #Calculate the predicted values by multiplying the design matrix by the coefficient vector
    return pred - y #Return the residuals between the predicted and observed values

def fit_torque_to_model(x_fit, y_fit, field, smoothed_fields, x_err, y_err, field_min = -100, field_max = 100, plots = False, final_plots = False):
    '''
    Fits the measured torque at a specified applied magnetic field to a
    two-component torque model using the fitted X- and Y-direction magnetic
    moment data. The function calculates the magnetic moment magnitude and
    orientation, constructs two torque-model regressors, and fits their
    coefficients using both ordinary least squares and a robust Huber fit.
    Optional diagnostic and final plots can be generated.

    Inputs:
        x_fit (dict): Dictionary containing fitted X-direction magnetic moment
            data for each measurement angle.
        y_fit (dict): Dictionary containing fitted Y-direction magnetic moment
            data for each measurement angle.
        field (float): Applied magnetic field at which the torque model is
            evaluated.
        smoothed_fields (numpy.ndarray): Applied magnetic field values
            corresponding to the fitted magnetic moment data.
        x_err (dict): Dictionary containing uncertainties associated with the
            fitted X-direction magnetic moment data.
        y_err (dict): Dictionary containing uncertainties associated with the
            fitted Y-direction magnetic moment data.
        field_min (float): Lower applied-field boundary used to restrict the
            Y-direction fitted data.
        field_max (float): Upper applied-field boundary used to restrict the
            Y-direction fitted data.
        plots (bool): Determines whether intermediate diagnostic plots and
            printed fitting information are generated.
        final_plots (bool): Determines whether final torque-fit, residual, and
            three-dimensional fit plots are generated.

    Outputs:
        T_dict (dict): Dictionary containing the calculated torque for each
            measurement angle at the specified applied magnetic field.
        phi_dict (dict): Dictionary containing the calculated magnetic moment
            orientation for each measurement angle in degrees.
        yhat_s (numpy.ndarray): Model-predicted torque values sorted by
            measurement angle.
        theta_s (numpy.ndarray): Measurement angles in radians, sorted from
            smallest to largest.
        T_err (dict): Dictionary containing the calculated torque uncertainty
            for each measurement angle.
    '''

    mask = np.ones_like(smoothed_fields, dtype=bool) #Create a boolean mask initially set to True for every smoothed-field data point
    mask &= smoothed_fields >= field_min #Keep only smoothed-field data points greater than or equal to the specified lower field boundary
    mask &= smoothed_fields <= field_max #Keep only smoothed-field data points less than or equal to the specified upper field boundary
    smoothed_field = smoothed_fields[mask] #Apply the field mask to obtain the smoothed-field values within the specified fitting range

    lang_poly_y_fit_aoi = {key:y_fit[key][mask] for key in y_fit.keys()} #Apply the field mask to the fitted Y-direction data for every measurement angle
    y_aoi_err = {key:y_err[key][mask] for key in y_err.keys()} #Apply the same field mask to the Y-direction fitting uncertainties for every measurement angle
    lang_poly_x_fit_aoi = x_fit #Store the fitted X-direction data without applying the field mask
    x_aoi_err = x_err #Store the X-direction fitting uncertainties without applying the field mask

    moment_dict_x,x_mom_err = plot_specific_moment_err(lang_poly_x_fit_aoi, smoothed_field, x_aoi_err, field, ylabel = None, plot = plots) #Extract the X-direction magnetic moment and uncertainty at the specified applied field for each measurement angle
    moment_dict_y,y_mom_err = plot_specific_moment_err(lang_poly_y_fit_aoi, smoothed_field, y_aoi_err, field, ylabel = None, plot = plots) #Extract the Y-direction magnetic moment and uncertainty at the specified applied field for each measurement angle
    phi_dict, phi_err = calculate_theta_err(moment_dict_x, moment_dict_y, y_mom_err, x_mom_err, field, plot = plots, ylabel = None, offset = 0) #Calculate the magnetic moment orientation and its uncertainty from the X- and Y-direction magnetic moments

    T_dict, T_err = calculate_torque_err(moment_dict_x, moment_dict_y, x_mom_err, y_mom_err, field, plot = plots, ylabel = 'Magetic Torque (N*m)') #Calculate the magnetic torque and its uncertainty from the X- and Y-direction magnetic moments

    m = np.array(([np.sqrt(moment_dict_x[key]**2 + moment_dict_y[key]**2) for key in moment_dict_x])) #Calculate the magnitude of the magnetic moment for each measurement angle from its X- and Y-components
    T = np.array(([T_dict[key] for key in T_dict])) #Convert the calculated torque values into a NumPy array
    H = np.array(([field]*len(T))) #Create an array containing the applied magnetic field value for every measurement angle
    theta = np.array(([np.deg2rad(key) for key in x_fit.keys()])) #Convert the measurement angles from degrees to radians
    phi = np.array(([np.deg2rad(phi_dict[key]) for key in x_fit.keys()])) #Convert the calculated magnetic moment orientations from degrees to radians
    µ0 = 4*np.pi*10**(-7) #Define the vacuum permeability in SI units

    # x1 = - m * np.sin(2*phi) #Previous form of the first torque-model regressor without the vacuum permeability and squared moment terms
    # x2 = - H * np.sin(theta - phi) #Previous form of the second torque-model regressor without the vacuum permeability and moment terms
    x1 = - µ0*np.square(m)*np.sin(2*phi) #Calculate the first torque-model regressor based on the squared magnetic moment and its orientation
    x2 = - µ0*np.multiply(m,H)*np.sin(theta - phi) #Calculate the second torque-model regressor based on the magnetic moment, applied field, and angular difference between field and moment
    X = np.column_stack((x1, x2)) #Combine the two torque-model regressors into a two-column design matrix
    y = T #Use the calculated torque values as the target values for the model fit

    # --- Diagnostics ---
    XT_X = X.T @ X #Calculate the matrix used in the ordinary least-squares solution
    cond_X = np.linalg.cond(XT_X) #Calculate the condition number of the design matrix to assess potential numerical ill-conditioning
    corr = np.corrcoef(X.T) #Calculate the correlation matrix between the two torque-model regressors

    if plots: #Print diagnostic information when intermediate plotting and diagnostics are enabled
        print("n =", len(y)) #Print the number of torque measurements included in the fit
        print("cond(X^T X) =", cond_X) #Print the condition number of the regression matrix
        print("regressor correlation matrix:\n", corr) #Print the correlation matrix between the two model regressors
        print("regressor ranges:", X.min(axis=0), X.max(axis=0)) #Print the minimum and maximum values of each model regressor
        print("y range:", y.min(), y.max()) #Print the minimum and maximum measured torque values

    try: #Attempt to calculate the ordinary least-squares solution directly
        beta_ls = np.linalg.inv(X.T @ X) @ (X.T @ y) #Calculate the ordinary least-squares coefficients using the normal equation
    except np.linalg.LinAlgError: #If the regression matrix cannot be inverted, use a regularized solution instead
        beta_ls = ridge_solution(X, y, alpha=1e-8) #Calculate the regression coefficients using ridge regularization
    A_ls, B_ls = beta_ls #Separate the fitted ordinary least-squares coefficients into A and B

    if plots: #Print the ordinary least-squares coefficients when diagnostics are enabled
        print("OLS A,B =", A_ls, B_ls) #Display the fitted A and B coefficients from the ordinary least-squares model

    p0 = beta_ls.copy() #Use the ordinary least-squares coefficients as the initial guess for the robust Huber fit
    res_huber = least_squares(residuals_ab, p0, args=(X, y), loss='huber', f_scale=1.0,
                            bounds=([-np.inf, -np.inf],[np.inf, np.inf]), max_nfev=2000) #Perform a robust nonlinear least-squares fit using a Huber loss function to reduce the influence of outliers
    A_hub, B_hub = res_huber.x #Extract the fitted A and B coefficients from the Huber regression

    if plots: #Print Huber regression diagnostics when requested
        print("Huber-fit A,B =", A_hub, B_hub) #Display the fitted A and B coefficients from the Huber regression
        print("Huber success:", res_huber.success, res_huber.message) #Display whether the Huber optimization succeeded and its associated status message

    if cond_X > 1e8: #Check whether the regression matrix is sufficiently ill-conditioned to warrant testing ridge regularization
        alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2] #Define a range of ridge regularization strengths to evaluate
        for a in alphas: #Loop through each candidate ridge regularization strength
            b_ridge = ridge_solution(X, y, alpha=a) #Calculate the ridge-regression coefficients using the current regularization strength
            rss = np.sum((X@b_ridge - y)**2) #Calculate the residual sum of squares for the ridge-regression solution
            if plots: #Print the ridge-regression diagnostics when requested
                print(f"alpha={a:.0e}, A,B={b_ridge}, RSS={rss:.3e}") #Display the regularization strength, fitted coefficients, and residual sum of squares
    

    beta = (A_hub, B_hub) if res_huber.success else beta_ls #Use the Huber coefficients if the robust fit succeeded; otherwise use the ordinary least-squares coefficients
    y_hat = X @ beta #Calculate the torque predicted by the selected model for each measurement angle
    resid = y - y_hat #Calculate the residual between the measured and model-predicted torque

    # --- plot: use theta as independent variable, sort so line is smooth ---
    sort_idx = np.argsort(theta) #Determine the indices that sort the measurement angles from smallest to largest
    theta_s = theta[sort_idx] #Sort the measurement angles using the calculated sorting indices
    y_s = y[sort_idx] #Sort the measured torque values using the same angle ordering
    yhat_s = y_hat[sort_idx] #Sort the model-predicted torque values using the same angle ordering
    resid_s = resid[sort_idx] #Sort the torque residuals using the same angle ordering

    fit_thetas = np.linspace(0,2*np.pi,200) #Generate a smooth set of angles spanning 0 to 2π for evaluating a continuous model curve
    comp1 = X[:,0]*beta[0] #Calculate the contribution to the fitted torque from the first model component
    comp2 = X[:,1]*beta[1] #Calculate the contribution to the fitted torque from the second model component

    if final_plots: #Generate the final torque-fit, residual, and three-dimensional plots when requested
            # --- Existing 2D fit + residuals ---
            RSS = np.sum(resid**2) #Calculate the residual sum of squares for the selected torque model
            TSS = np.sum((y - y.mean())**2) #Calculate the total sum of squares relative to the mean measured torque
            R2 = 1 - RSS / TSS #Calculate the coefficient of determination for the fitted torque model
            print("RSS:", RSS, "R2:", R2) #Print the residual sum of squares and R-squared value

            keys = theta #Store the measurement angles in radians for colormap normalization
            norm = Normalize(vmin=min(keys), vmax=max(keys)) #Normalize the measurement angles to the range required by the colormap
            cmap = viridis #Set the colormap used for the measurement-angle points

            # 2D plot of torque vs θ
            for i in range(len(theta)): #Loop through each measurement angle to plot the measured and fitted torque values
                color = cmap(norm(keys[i])) #Determine the plotting color corresponding to the current measurement angle
                plt.scatter(np.rad2deg(theta[i]), T[i], color=color, linewidths=2, edgecolors='black', s=100) #Plot the measured torque at the current measurement angle
                plt.scatter(np.rad2deg(theta[i]), yhat_s[i], color=color, linewidths=2, edgecolors='black', s=100, alpha=0.5) #Plot the model-predicted torque at the current measurement angle with reduced transparency

            plt.plot(np.rad2deg(theta_s), yhat_s, color='k', linestyle='--', lw=2,
                    label=f'Fit (A={beta[0]:.3g}, B={beta[1]:.3g})', zorder=0) #Plot the fitted torque as a dashed line through the sorted measurement angles
            plt.xlabel(r'$\theta$ (deg)') #Label the x-axis with the measurement angle
            plt.ylabel('T (Torque)') #Label the y-axis with torque
            plt.title('Torque Fit vs θ') #Set the title of the torque-versus-angle plot
            plt.legend() #Display the legend identifying the fitted model
            plt.grid(alpha=0.3) #Add a partially transparent grid to the plot
            plt.show() #Display the torque fit plot

            # Residuals
            plt.figure(figsize=(8,3)) #Create a new figure for plotting the model residuals
            plt.scatter(np.rad2deg(theta), resid, color='C3', alpha=0.8) #Plot the torque residuals as a function of measurement angle
            plt.axhline(0, color='k', ls='--') #Add a horizontal reference line at zero residual
            plt.xlabel(r'$\theta$ (deg)') #Label the x-axis with the measurement angle
            plt.ylabel('Residuals') #Label the y-axis with the torque residual
            plt.title('Residuals vs θ') #Set the residual plot title
            plt.grid(alpha=0.3) #Add a partially transparent grid to the residual plot
            plt.show() #Display the residual plot

            # --- NEW: 3D surface plot of T vs x1, x2 ---
            from mpl_toolkits.mplot3d import Axes3D #Import the 3D plotting toolkit required to create a three-dimensional Matplotlib axis

            fig = plt.figure(figsize=(10, 8)) #Create a new figure for the three-dimensional torque model visualization
            ax = fig.add_subplot(111, projection='3d') #Create a three-dimensional subplot within the figure

            # Scatter experimental data points
            sc = ax.scatter(x1, x2, y, c=np.rad2deg(theta), cmap=viridis, s=60, edgecolors='k', alpha=0.8, label='Data') #Plot the experimental torque data in three-dimensional regressor space and color the points by measurement angle

            # Create grid for fit surface
            x_fit1 = np.linspace(np.min(x1), np.max(x1), 50) #Generate 50 evenly spaced values spanning the range of the first torque-model regressor
            x_fit2 = np.linspace(np.min(x2), np.max(x2), 50) #Generate 50 evenly spaced values spanning the range of the second torque-model regressor
            X1_grid, X2_grid = np.meshgrid(x_fit1, x_fit2) #Create a two-dimensional grid from the two torque-model regressor ranges
            T_fit_surface = beta[0] * X1_grid + beta[1] * X2_grid #Evaluate the fitted linear torque model across the regressor grid

            # Plot fitted plane
            ax.plot_surface(X1_grid, X2_grid, T_fit_surface, alpha=0.4, cmap=viridis, linewidth=0, zorder=0) #Plot the fitted torque model as a three-dimensional plane

            # Plot 3D line through the grid diagonal (optional)
            sort_idx = np.argsort(theta) #Determine the indices that sort the measurement angles from smallest to largest
            x1_sorted = x1[sort_idx] #Sort the first torque-model regressor according to measurement angle
            x2_sorted = x2[sort_idx] #Sort the second torque-model regressor according to measurement angle
            T_sorted = y[sort_idx] #Sort the measured torque values according to measurement angle

            # Smooth trajectory of measured data (3D curve)
            ax.plot(x1_sorted, x2_sorted, T_sorted, color='k', lw=2.5, label='Measured trajectory') #Connect the experimental data points in three-dimensional regressor space to show their trajectory with measurement angle

            ax.set_xlabel(r'$x_1 = -\mu_0 m^2 \sin(2\phi)$') #Label the first regressor axis with its mathematical definition
            ax.set_ylabel(r'$x_2 = -\mu_0 mH \sin(\theta-\phi)$') #Label the second regressor axis with its mathematical definition
            ax.set_zlabel('Torque (T)') #Label the vertical axis with torque
            ax.set_title('Torque Fit in 3D') #Set the title of the three-dimensional torque fit
            ax.view_init(20, 135) #Set the viewing elevation and azimuth angles for the three-dimensional plot
            plt.colorbar(sc, ax=ax, label=r'$\theta$ (deg)') #Add a colorbar showing the measurement angle associated with each experimental point
            ax.legend() #Display the legend identifying the measured trajectory
            plt.tight_layout() #Adjust the plot layout to prevent labels and other elements from overlapping
            plt.show() #Display the three-dimensional torque fit plot

    
    return T_dict, phi_dict, yhat_s, theta_s, T_err #Return the calculated torque, magnetic moment orientation, sorted model prediction, sorted measurement angles, and torque uncertainty
