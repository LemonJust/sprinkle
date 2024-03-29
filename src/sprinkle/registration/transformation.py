from __future__ import annotations
import ants 

import numpy as np
import numpy.typing as npt
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt

from sprinkle.registration.ants_utils import transform_image, transform_points
from dataclasses import dataclass
from sprinkle.segmentation.segmentation import Image


@dataclass
class ImageTransformation:
    """
    A class to represent a transformation of an image.
    Attributes:
        interpolation: str the interpolation to use when applying the transformation. 'linear', 'nearestNeighbor', 'multiLabel'
        transformation_list: list[str] lists the paths to the transformations to apply in the order they should be applied.
    """
    image: Image
    reference_image: Image
    interpolation: str
    transformation_list: list[str]
    direction: str

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates a Transformation from a dictionary.
        """
        d = d.copy()
        d['image'] = Image(**d['image'])
        d['reference_image'] = Image(**d['reference_image'])
        return cls(**d)

    def run(self) -> npt.NDArray:
        """
        Runs the transformation.
        """
        resolution_xyz = [self.image.resolution_xyz["x"], 
                          self.image.resolution_xyz["y"], 
                          self.image.resolution_xyz["z"]]
        reference_resolution_xyz = [self.reference_image.resolution_xyz["x"], 
                                          self.reference_image.resolution_xyz["y"], 
                                          self.reference_image.resolution_xyz["z"]]
        # run the transformation
        transformed_image = transform_image(image = self.image.load(),
                                            resolution_xyz = resolution_xyz,
                                            reference_image = self.reference_image.load(),
                                            reference_image_resolution_xyz = reference_resolution_xyz,
                                            transformlist = self.transformation_list,
                                            interpolation = self.interpolation,
                                            verbose = True)
        return transformed_image

# !!! Point transformation is working as a script for now: 
# see "transform_coordinates" in scripts/save_transform_class_centroids.py
# !!!    
# @dataclass
# class PointsTransformation:
#     """
#     A class to represent a transformation.
#     Attributes:
#         interpolation: str the interpolation to use when applying the transformation. 'linear', 'nearestNeighbor', 'multiLabel'
#         transformation_list: list[str] lists the paths to the transformations to apply in the order they should be applied.
#     """
#     points: npt.NDArray
#     reference_image: Image
#     transformation_list: list[str]
#     direction: str

#     @classmethod
#     def from_dict(cls, d: dict):
#         """
#         Creates a Transformation from a dictionary.

#         Args:
#             d (dict): the dictionary containing the transformation information
#                 must contain the following keys:
#                 - points: the points to transform
#         """
#         d = d.copy()
#         d['points'] = np.array(d['points'])
#         d['reference_image'] = Image(**d['reference_image'])
#         return cls(**d)

#     def run(self) -> npt.NDArray:
#         """
#         Runs the transformation.
#         """
#         reference_resolution_xyz = [self.reference_image.resolution_xyz["x"], 
#                                           self.reference_image.resolution_xyz["y"], 
#                                           self.reference_image.resolution_xyz["z"]]
#         # run the transformation
#         transformed_points = transform_points(points = self.points,
#                                               reference_image = self.reference_image.load(),
#                                               reference_image_resolution_xyz = reference_resolution_xyz,
#                                               transformlist = self.transformation_list,
#                                               verbose = True)
#         return transformed_points
                                                     
    





