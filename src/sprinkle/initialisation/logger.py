"""
Checks and logs the raw images + models in the database.
"""
from __future__ import annotations

from sprinkle.db.manager import DatabaseManager
from pathlib import Path


class ImageAlreadyInDatabase(Exception):
    """Image is already in the database."""
    def __init__(self, image_file: str, channel: int):
        self.msg = f"The image {image_file} with channel {channel} is already in the database." 
    def __str__(self):
        return self.msg
    
class SegmentationModelAlreadyInDatabase(Exception):
    """Segmentation model is already in the database."""
    def __init__(self, model_folder: str):
        self.msg = f"The segmentation model {model_folder} is already in the database." 
    def __str__(self):
        return self.msg

class InitialisationLogger:
     
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def log_raw_image(self,
                      channel: int,
                      file_name: str | Path,
                      stage: str,
                      label: str,
                      transformed: bool,
                      image_space_name: str,
                      resolution_xyz: list[float, float, float] | None = None):
        """
        Log a raw image in the database.

        Args:
            channel: the channel of the image
            file_name: the filename of the image
            stage: the stage of the image
            label: the label of the image
            transformed: whether the image is transformed
            image_space_name: the name of the image space
            resolution_xyz: the resolution of the image space
        """
        if isinstance(file_name, Path):
            file_name = file_name.as_posix()

        # check that file exists on disk
        if not Path(file_name).exists():
            raise Exception(f"Image {file_name} does not exist. Check the file name and path.")
        
        # check that image already exists in the database
        if self.db.get_image(channel, file_name) is not None:
            raise ImageAlreadyInDatabase(file_name,channel)

        # if image space is not in the database, resolution must be provided
        if resolution_xyz is None and self.db._get_or_create_image_space(image_space_name) is None:
            raise Exception(f"Image space {image_space_name} does not exist in the database. Resolution must be provided.")

        self.db.add_image(channel, file_name, stage, label, transformed, image_space_name, resolution_xyz)

    def log_segmentation_model(self,
                                 model_folder: str | Path,
                                 model_type: str,
                                 segmentation_type: str):
        """
        Log a segmentation model in the database.

        Args:
            model_folder: the folder of the segmentation model
            model_type: the type of the segmentation model : 'stardist3D'
            segmentation_type: the type of the segmentation : 'synapse' or 'nuclei'
        """
        if isinstance(model_folder, Path):
            model_folder = model_folder.as_posix()

        # check that model folder exists
        if not Path(model_folder).exists():
            raise Exception(f"Model folder {model_folder} does not exist. Check the folder name and path.")
        
        # check that model already exists in the database
        if self.db.get_segmentation_model(model_folder) is not None:
            raise SegmentationModelAlreadyInDatabase(model_folder)
        
        # get the segmentation training id
        # TODO: add segmentation training id processing here
        segmentation_training_id = None

        print(f"Adding segmentation model {model_folder} to the database.")
        
        self.db.add_segmentation_model(model_folder, model_type, segmentation_type, segmentation_training_id)

    
          
        

        

                        
