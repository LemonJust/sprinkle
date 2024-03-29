"""
Contain the valid values for the fields of the database.
"""
from enum import Enum

class Stages(Enum):
    """
    Valid values for image types.stage .
    """
    raw = "raw"
    segmented = "segmented"
    classified = "classified"

class Labels(Enum):
    """
    Valid values for image types.labels .
    """
    gfp = "gfp"
    dapi = "dapi"
    huc = "huc"
    hucd = "hucd"
    hoechst = "hoechst"
    labels = "labels"
    probabilities = "probabilities"


class ImageSpaces(Enum):
    """
    Valid values for image spaces.
    """
    live1 = "live1"
    live2 = "live2"
    fixed = "fixed"

class SegmentationTypes(Enum):
    """
    Valid values for segmentation types.
    """
    synapse = "synapse"
    nuclei = "nuclei"

class SegmentationModelTypes(Enum):
    """
    Valid values for segmentation model types.
    """
    stardist = "stardist3D"