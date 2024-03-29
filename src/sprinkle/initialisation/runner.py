"""
Runs the initialisation task defined in the config dictionary,
use this to run the initialisation task as a part of the pipeline.
"""
from __future__ import annotations

from sprinkle.db.manager import DatabaseManager
from sprinkle.initialisation.logger import InitialisationLogger, ImageAlreadyInDatabase, SegmentationModelAlreadyInDatabase

def run_initialisation_from_dict(d: dict, db: DatabaseManager) -> None:
    """
    Runs the initialisation task defined in the config dictionary, 
    use this to run the segmentation task as a part of the pipeline.

    Will skip images and segmentation models that are already in the database.
    """
    initialisation_logger = InitialisationLogger(db)

    # add images to the database
    stage = "raw"
    transformed = False
    if "raw_images" in d:
        for image in d["raw_images"]:
            image = image["image"] # each image is a dict with one key
            if "resolution_xyz" not in image:
                image["resolution_xyz"] = None
            try:
                initialisation_logger.log_raw_image(image["channel"], image["file_name"],
                                                    stage, image["label"], transformed,
                                                    image["image_space"], 
                                                    image["resolution_xyz"])
            except ImageAlreadyInDatabase:
                print(f"Image {image['file_name']} with channel {image['channel']} is already in the database.")

    # add segmentation models to the database
    if "segmentation_models" in d:
        for model in d["segmentation_models"]:
            model = model["model"] # each model is a dict with one key
            try:
                initialisation_logger.log_segmentation_model(model["model_folder"], 
                                                            model["model_type"], 
                                                            model["segmentation_type"])
            except SegmentationModelAlreadyInDatabase:
                print(f"Segmentation model {model['model_folder']} is already in the database.")
