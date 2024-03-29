"""
Contains classes for dealing with images.
"""
from __future__ import annotations
from pathlib import Path
import tifffile as tif
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from sprinkle.db.manager import DatabaseManager


def save_image_as_uint16(img: npt.NDArray , out_file: str | Path, 
                         resolution: list[float, float, float] = [1.0, 1.0, 1.0], 
                         scale: int | None = None):
    """
    Save the image as uint16 tiff file after scaling it by scale. 
    Scale is used to scale float values, such as probability, to the range of uint16.

    Args:
        img (np.ndarray): image to save
        out_file (str): path to save the image to
        resolution (tuple[float, float, float], optional): resolution of the image. In XYZ order.
        scale (float, optional): scale to apply to the image.
    """
    if scale is not None:
        img = img * scale


    #make sure  all values can fint into the uint range
    img[img > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
    img[img < np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
    # assert np.max(img) < 65535, "some labels image values are too large for uint16"
    # assert np.min(img) >= 0, "some labels image values are too small for uint16"

    img = img.astype(np.uint16)

    resolution_x, resolution_y, resolution_z = resolution
    
    tif.imwrite(out_file, img, 
                resolution=(1. / resolution_x, 1. / resolution_y),
                metadata={'spacing': resolution_z, 'unit': 'um', 'axes': 'ZYX'},
                imagej=True)
    
def create_class_image(labels_image: npt.NDArray, class_labels: list[int]) -> npt.NDArray:
    """
    Creates an image containing elements of a certain class from the labels image and the classification.

    Args:
        labels_image (np.ndarray): the labels image
        class_labels (list[int]): the list of label ids to keep from the labels image

    Returns:
        np.ndarray: the class image, containing only the class ids provided
    """

    class_img = np.zeros_like(labels_image)
    class_mask = np.isin(labels_image, class_labels)
    class_img[class_mask] = labels_image[class_mask]

    return class_img

@dataclass
class Image:
    """
    Attributes:
        file_name: path to the image file
        channel: channel to segment
        resolution_xyz: (x, y, z) resolution in um
    """
    file_name: str 
    channel: int
    label: str
    image_space: str
    resolution_xyz: tuple[float, float, float] # will be converted to a dict in __post_init__

    def __post_init__(self):
        # convert spacing to a dictionary
        self.resolution_xyz = {k: v for k, v in zip(['x', 'y', 'z'], self.resolution_xyz)}
    
    def load(self):
        """
        Loads the image to segment based on the image file name and channel.
        """
        img = tif.imread(self.file_name)
        if len(img.shape) == 4:
            img = img[:, self.channel, : , : ] # tiffile loads in ZCYX format
        elif len(img.shape) != 3:
            raise ValueError(f"Image {self.file_name} has {len(img.shape)} dimensions, expected 3 or 4.")
        return img
