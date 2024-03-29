"""
This module contains the function and classes to segment the synapse and cells from the images.
"""
from __future__ import print_function, unicode_literals, absolute_import, division, annotations

import numpy as np
import numpy.typing as npt
import yaml
from dataclasses import dataclass, field, asdict

import tifffile as tif
from csbdeep.utils import normalize
from pathlib import Path

from sprinkle.segmentation.stardist_utils import stardist_segment
from sprinkle.image_utils import Image

@dataclass
class Parameters:
    """
    Attributes:
        n_tiles: tuple[int,int,int] = (1,4,4), 
        prob_thr: float = 0.1, 
        scale_probability: int = 10000
    """
    n_tiles: tuple[int,int,int] = (1,4,4)
    prob_thr: float = 0.1
    scale_probability: int = 10000

    def as_dict(self):
        return asdict(self)

@dataclass
class Segmentation:
    """
    A class to perform segmentation of the image with the given model.
    Attributes:
        entity: the entity to segment: must be 'cell' or 'synapse'
        image_type: image type to segment with stage, label, transformed, segmented flags
        image: image to segment with file_name, resolution, channel
        model: folder with the saved StarDist3D model to use for segmentation
        parameters: a dict of parameters to use for segmentation: n_tiles, prob_thr, scale. 
            n_tiles: number of tiles to use for prediction
            prob_thr: probability threshold for the prediction
            scale_probability: scale to use for saving the probability image
        output_folder: a folder to save the segmentation results, should point to the folder holding processed images in the project data.
    """
    image: Image
    model_folder: str
    parameters: Parameters

    labels_image: npt.NDArray[np.uint16] = field(init=False)
    probability_image: npt.NDArray[np.uint16] = field(init=False)
    centroids: npt.NDArray[np.float32] = field(init=False)
    probabilities: npt.NDArray[np.float32] = field(init=False)

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates a Segmentation from a dictionary.
        """
        # d = d.copy()
        s = {}

        if 'resolution_xyz' in d['image'] and d['image']['resolution_xyz'] is not None:
            assert len(d['image']['resolution_xyz']) == 3, "resolution_xyz must be a tuple of 3 elements"
            d['image']['resolution_xyz'] = tuple(d['image']['resolution_xyz'])
        else:
            d['image']['resolution_xyz'] = (1.0,1.0,1.0)

        s['image'] = Image(**d['image'])
        s['model_folder'] = d['model']['model_folder']
        s['parameters'] = Parameters(**d['parameters'])
        return cls(**s)
    
    @classmethod
    def from_yaml_file(cls, config_file: str | Path):
        """
        Creates a SegmentationTask from a yaml config file.
        the config file should have the following structure:
            image: 
                file_name: /inhipy/data/raw/fixed_gfp.tif
                channel: 0
            model_folder: /inhipy/models/images/stardist_xxx
            parameters:
                n_tiles: [1,4,4] # number of tiles to use for prediction (z,y,x), change this to fit your GPU memory
                prob_thr: 0.1 # probability threshold for the prediction, set low to avoid false negatives
                scale_probability: 10000 # scale to use for saving the probability image (float values are scaled to uint16)
        Args: 
            config_file: path to the config file
        """
        with open(config_file) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(d)

    def preprocess_image(self, img: npt.NDArray) -> npt.NDArray:
        """
        Preprocesses the image for segmentation. 
        Simple normalization of the image to 1-99.8 percentile range.
        """  
        img = normalize(img, 1,99.8, axis=(0,1,2))
        return img
    
    def run(self) -> tuple:
        """
        Segments the image and saves the results to the output directory.

        Returns:
            label_image: the labeled image, 3D numpy array of uint16
            probability_image: the probability image, 3D numpy array of uint16
            centroids: the centroids of the labels, list of tuples (z,y,x)
            probabilities: the probabilities of the labels, list of floats
        """
        img = self.preprocess_image(self.image.load())

        # segment the image
        labels, details = stardist_segment(self.model_folder,
                                           img, 
                                           n_tiles = self.parameters.n_tiles, 
                                           prob_thresh = self.parameters.prob_thr, 
                                           return_predict=True)
        # prepare the results 
        label_image = labels[0]
        probability_image = details[0]
        centroids = labels[1]['points']
        probabilities = labels[1]['prob']

        return label_image, probability_image, centroids, probabilities

