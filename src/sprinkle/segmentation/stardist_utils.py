from __future__ import annotations

from stardist.models import StarDist3D
import numpy as np
import numpy.typing as npt
from pathlib import Path
np.random.seed(6)

def load_stardist_model(model_folder: str | Path, verbose: bool = True):
    """
    Loads the model from the model_folder.
    """
    model_folder = Path(model_folder)
    if verbose:
        print(f"Loading model from {model_folder}")
    return StarDist3D(None, name=model_folder.name, basedir=str(model_folder.parent))

def stardist_segment(model_folder: str | Path, img: npt.NDArray , 
                     n_tiles: tuple[int, int, int] =(1,4,4), 
                     prob_thresh: float = 0.1, 
                     return_predict: bool = True):
    """
    Segments the image with the given stardist model.
    Probability threshold is set low by default to avoid false negatives. False positives will be filtered out later.

    Args:
        model_folder: path to the model folder
        img: 3D image to segment
        n_tiles: number of tiles to use for prediction (z,y,x), change this to fit your GPU memory
        prob_thresh: probability threshold for the prediction, set low to avoid false negatives
        return_predict: if True, returns the prediction details, set to True to get the probability image. 

    Returns:
        labels: a tuple 
            (3D numpy array of labeled image, 
            dict of details with the probability and center of each label among other things)
        details: a tuple (3D numpy array of probability image, 3Dxn_rays numpy array for distribution image). 
            None if return_predict is False.
    """
    model = load_stardist_model(model_folder)

    if return_predict:
        labels, details = model.predict_instances(img, n_tiles=n_tiles, prob_thresh=prob_thresh, return_predict=True)
    else:
        labels = model.predict_instances(img, n_tiles=n_tiles, prob_thresh=prob_thresh)
        details = None

    return labels, details
