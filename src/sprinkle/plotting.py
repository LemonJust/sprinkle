"""
Plotting utils.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_slice_with_insert(img: np.ndarray,
                           slice: int = 80, 
                           zoom_in_patch: tuple[int, int, int, int] | None = None, # was good for full synapse image (200,500, 400, 400),
                           vmin: int = 100, 
                           vmax: int = 2000):
    """
    plot the full size image and an insert of the center in one figure next to each other
    draw a box around the area that is shown in the insert.

    Args:
        img: 3D numpy array, zyx
        slice: z slice in the 3D image to plot
        zoom_in_patch: tuple of (number of rows, number of columns, zoom in row, zoom in column) of the insert, zero-based.
        vmin, vmax: int, min and max values for the color scale
    """
    if zoom_in_patch is None:
        # split image innto 3x3 grid
        n_row, n_col = 3, 3
        # and zoom innto the center
        row, col = 1, 1
    else:
         n_row, n_col, row, col = zoom_in_patch

    # calculate the size and position of the insert
    hight = img.shape[1] // n_row
    width = img.shape[2] // n_col
    zoom_in_region = (col*width, row*hight) # y, x

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img[slice,:,:], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.imshow(img[slice,
               zoom_in_region[1]:zoom_in_region[1] + hight,
               zoom_in_region[0]:zoom_in_region[0] + width],
               cmap='gray', vmin=vmin, vmax=vmax)
    ax1.axis('off')
    ax2.axis('off')
    ax1.add_patch(plt.Rectangle( zoom_in_region, width, hight, 
                                 linewidth=1, edgecolor='w', facecolor='none'))
    
def plot_segmentation_example(img, segmentation, vmin = 100, vmax = 2000, show_slices = None):
    # only import stardist if plotting segmentation
    from stardist import random_label_cmap
    lbl_cmap = random_label_cmap()

    n_slices = img.shape[0]
    width, height = img.shape[1:]

    if show_slices is None:
        show_slices = [3*n_slices//8, 5*n_slices//8]

    fig, ax = plt.subplots(2,2, figsize=(16,16*(int(height/width))))
    for a, i_s, plot_seg in zip(ax.flat,[show_slices[0],show_slices[0], show_slices[1],show_slices[1]],
                                [False,True,False,True]):
        a.imshow(img[i_s], cmap='gray', vmin=vmin, vmax=vmax)
        if plot_seg:
            a.imshow(segmentation[i_s], cmap=lbl_cmap, alpha=0.5)
        a.set_title(i_s)
    [a.axis('off') for a in ax.flat]
    plt.tight_layout()