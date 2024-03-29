"""
This module contains the function and classes to run the diffrent tasks.
"""
from __future__ import annotations

from pathlib import Path
from sprinkle.segmentation.segmentation import Segmentation
from sprinkle.db.manager import DatabaseManager
from sprinkle.segmentation.logger import SegmentationLogger, SegmentationTaskAlreadyInDatabase


def run_segmentation_from_dict(d: dict, 
                               db: DatabaseManager, 
                               skip_exist: bool = True, 
                               overwrite: bool = False) -> None:
    """
    Runs the segmentation task defined in the config dictionary, 
    use this to run the segmentation task as a part of the pipeline.

    Args:
        d: the dictionary defining the segmentation task
        db: the database manager
        skip_exist: whether to skip the segmentation task if it is already in the database
        overwrite: whether to overwrite the segmentation task if it is already in the database
    """
    segmentation = Segmentation.from_dict(d)
    seg_logger = SegmentationLogger(db)

    try:
        # log the segmentation task in the database, 
        # will raise an exception if the model or image are not found
        seg_logger.log_segmentation(segmentation)
    except SegmentationTaskAlreadyInDatabase:
        if skip_exist:
            print(f"Segmentation task for image {segmentation.image.file_name} with channel {segmentation.image.channel} and model {segmentation.model_folder} is already in the database. The task will be skipped.")
            return
        elif overwrite:
            raise NotImplementedError
            print(f"Overwriting segmentation task for image {segmentation.image.file_name} with channel {segmentation.image.channel} and model {segmentation.model_folder}. Existing label and probability images will be overwritten.")
        else:
            raise 

    # run the segmentation
    label_image, probability_image, centroids, probabilities = segmentation.run()

    # save the results in the database and on disk
    seg_logger.save_and_log_labels_image(label_image)
    seg_logger.save_and_log_probabilities_image(probability_image)
    seg_logger.log_centroids(centroids, probabilities)

def run_segmentation_from_yaml(config_file: str | Path):
    """
    Runs the segmentation task solo defined in the yaml config file. 
    The yaml structure should be as follows, defining the database file in addition to the segmentation task:

    db_file: path to the database file

    segmentation:
        image:
            file_name: path to the image file
            channel: channel to segment
        model_folder: folder with the saved StarDist3D model to use for segmentation
        parameters: a dict of parameters to use for segmentation: n_tiles, prob_thr, scale.
            n_tiles: number of tiles to use for prediction
            prob_thr: probability threshold for the prediction
            scale_probability: scale to use for saving the probability image
    """
    pass

if __name__ == "__main__":
    run_segmentation_from_dict()


                                
