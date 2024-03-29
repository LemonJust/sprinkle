"""
Prepare training data for the neuron classification into excitatory and inhibitory neurons.
Need to crop some area around the centroid of the neuron and keep it as numpy array.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import tifffile as tif

import numpy as np

def crop_neurons(labels_file: str | Path, 
                 image_files: [str] | [Path]):
    """
    Given a csv file with the centroids of the neurons, labels and an image file, crop the neurons from the image and prepare the data and the labels.

    Args:
        labels_file: path to the csv file containing the centroids of the neurons and their labels
        image_file: list of paths to the image files, each file contains one channel of the image.
    """

    # read the labels file
    labels = pd.read_csv(labels_file)

    # read the images
    image_ch0 = tif.imread(image_files[0])
    image_ch1 = tif.imread(image_files[1])
    image_ch2 = tif.imread(image_files[2])

    # crop the neurons
    box_size = 56
    half_box_size = box_size // 2
    cropped_neurons = []

    for idx, row in labels.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        neuron_label = row['label']

        # crop the neurons
        neuron_ch0 = image_ch0[z-half_box_size:z+half_box_size, 
                               y-half_box_size:y+half_box_size, 
                               x-half_box_size:x+half_box_size]
        neuron_ch1 = image_ch1[z-half_box_size:z+half_box_size,
                                 y-half_box_size:y+half_box_size,
                                 x-half_box_size:x+half_box_size]
        neuron_ch2 = image_ch2[z-half_box_size:z+half_box_size,
                                    y-half_box_size:y+half_box_size,
                                    x-half_box_size:x+half_box_size]

        # save the cropped neuron
        cropped_neurons.append((neuron_label, neuron_ch0, neuron_ch1, neuron_ch2))



