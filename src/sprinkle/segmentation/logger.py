from __future__ import annotations

from sprinkle.db.manager import DatabaseManager

from pathlib import Path
import numpy.typing as npt
from sprinkle.image_utils import save_image_as_uint16
from sprinkle.segmentation.segmentation import Segmentation

class SegmentationTaskAlreadyInDatabase(Exception):
    """Segmentation task is already in the database."""
    def __init__(self, image_id: int, segmentation_model_id: int, parameters: dict):
        self.msg = f"The segmentation task with image_id {image_id}, segmentation_model_id {segmentation_model_id} and parameters {parameters} is already in the database." 
    def __str__(self):
        return self.msg

class SegmentationLogger:
     
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager
        # generated during log_segmentation_task:
        self.segmentation_task = None
        self.image = None
        self.scale_probability = None

    def log_segmentation_task(self, 
                              model_folder: str | Path, 
                              image_filename: str,
                              image_channel: int,
                              parameters: dict = {} ) -> None:
        """
        Log a segmentation run in the database.

        Args:
            model_folder: the folder of the segmentation model
            image_filename: the filename of the image to segment
            image_channel: the channel of the image to segment
            parameters: the parameters of the segmentation model, 
                must contain n_tiles, prob_thr and scale_probability
        """
        # check that parameters have the right format
        assert "n_tiles" in parameters, "Parameters must contain n_tiles."
        assert "prob_thr" in parameters, "Parameters must contain prob_thr."
        assert "scale_probability" in parameters, "Parameters must contain scale_probability."
        self.scale_probability = parameters["scale_probability"]

        # get the segmentation model
        segmentation_model = self.db.get_segmentation_model(model_folder)
        if segmentation_model is None:
            raise Exception(f"Segmentation model {model_folder} not found in database.")
        
        # get the image that was segmented
        self.image = self.db.get_image(image_channel, image_filename)
        if self.image is None:
            raise Exception(f"Image {image_filename} with channel {image_channel} not found in database.")
        
        # check that the segmentation task doesn't already exist in the database
        if self.db.get_segmentation_task(image_id = self.image.id, 
                                        segmentation_model_id = segmentation_model.id, 
                                        **parameters) is not None:
            raise SegmentationTaskAlreadyInDatabase(self.image.id, segmentation_model.id, parameters)
        
        # add the segmentation task (ommitting the labels and probabilities images)
        self.segmentation_task = self.db.add_segmentation_task(self.image.id, 
                                                               None, None, 
                                                               segmentation_model.id, 
                                                                parameters["n_tiles"],
                                                                parameters["prob_thr"],
                                                                parameters["scale_probability"])
        
    def log_segmentation(self, segmentation: Segmentation):
        """
        A wrapper around log_segmentation_task that takes a Segmentation object as input.
        """
        self.log_segmentation_task(segmentation.model_folder,
                                      segmentation.image.file_name,
                                      segmentation.image.channel,
                                      segmentation.parameters.as_dict())

        
    def _construct_labels_image_filename(self) -> str:
        """
        Construct the filename of the labels image: will add the '_labels' tag and segmentation task id to the original image filename.

        Returns:
            the filename of the labels image
        """
    
        suffix = Path(self.image.file_name).suffix
        st_id = self.segmentation_task.id
        labels_image_filename = self.image.file_name.replace(suffix, f"_labels_st{st_id}{suffix}")

        return labels_image_filename
    
    def _construct_probabilities_image_filename(self) -> str:
        """
        Construct the filename of the probabilities image: will add the '_probs' tag and segmentation task id to the original image filename.

        Returns:
            the filename of the probabilities image
        """

        suffix = Path(self.image.file_name).suffix
        st_id = self.segmentation_task.id
        probabilities_image_filename = self.image.file_name.replace(suffix, f"_probs_st{st_id}{suffix}")

        return probabilities_image_filename
        
    def save_and_log_labels_image(self, labels_image: npt.NDArray) -> str:
        """
        Save the labels image to the database.

        Args:
            labels_image: the labels image
        """
        # TODO: don't use _get_or_create_image_space, 
        # use the image space of the original image directly, 
        # maybe add a special add_modified_image method for this purpose ?

        if self.segmentation_task is None:
            raise Exception("Segmentation task not logged. Call log_segmentation_task first.")

        # save the labels image
        labels_image_filename = self._construct_labels_image_filename()
        save_image_as_uint16(labels_image, labels_image_filename)
        # TODO : danger! labels_image is first an array, but then an Image object! 
        
        # add the labels image to the database
        channel = 0
        stage = "segmented"
        label = "labels"
        transformed = False
        image_space_name = self.db._get_or_create_image_space(id = self.image.image_space_id).name

        labels_image = self.db.add_image(channel, labels_image_filename, 
                                         stage, label, transformed, 
                                         image_space_name)
        
        # update the segmentation task
        self.db.update_segmentation_task(self.segmentation_task.id, 
                                         labels_image_id = labels_image.id)
        return labels_image_filename
        
    def save_and_log_probabilities_image(self, probabilities_image: npt.NDArray)-> str:
        """
        Save the probabilities image to the database.

        Args:
            probabilities_image: the probabilities image
        """
        # TODO: don't use _get_or_create_image_space, 
        # use the image space of the original image directly, 
        # maybe add a special add_modified_image method for this purpose ?

        if self.segmentation_task is None:
            raise Exception("Segmentation task not logged. Call log_segmentation_task first.")

        # save the probabilities image
        probabilities_image_filename = self._construct_probabilities_image_filename()
        save_image_as_uint16(probabilities_image, probabilities_image_filename, 
                             scale = self.scale_probability)
        
        # add the probabilities image to the database
        channel = 0
        stage = "segmented"
        label = "probabilities"
        transformed = False
        image_space_name = self.db._get_or_create_image_space(id = self.image.image_space_id).name

        probabilities_image = self.db.add_image(channel, probabilities_image_filename, stage, label, transformed, image_space_name)
        
        # update the segmentation task
        self.db.update_segmentation_task(self.segmentation_task.id, 
                                         probabilities_image_id = probabilities_image.id)
        return probabilities_image_filename

    def log_centroids(self, 
                      centroids_xyz: list[tuple[float, float, float]], 
                      probabilities: list[float] ) -> None:
        """
        Log the centroids of the segmentation task.

        Args:
            centroids_xyz: the centroids coordinates to log in XYZ order
            probabilities: the probabilities of the centroids
        """

        if self.segmentation_task is None:
            raise Exception("Segmentation task not logged. Call log_segmentation_task first.")
        
        # construct the centroids labels (add 1 to the index to get the label since 0 is background)
        labels = [i_centroid + 1 for i_centroid in range(len(centroids_xyz))]

        # add the centroids to the database
        self.db.add_centroids(labels, centroids_xyz, probabilities, self.segmentation_task.id)