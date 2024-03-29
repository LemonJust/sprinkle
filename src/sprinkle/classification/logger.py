from __future__ import annotations

from sprinkle.classification.classification import Classification
from sprinkle.db.manager import DatabaseManager
from pathlib import Path
from dataclasses import dataclass, field
from sprinkle.segmentation.segmentation import Segmentation
from sprinkle.image_utils import create_class_image, save_image_as_uint16
import numpy.typing as npt
import tifffile as tif

class ClassificationTaskAlreadyInDatabase(Exception):
    """Segmentation task is already in the database."""
    def __init__(self, classification_method: str, segmentation_task_id: int, parameters: dict):
        self.msg = f"The {classification_method} classification task for centroids from segmentation task {segmentation_task_id}, and parameters {parameters} is already in the database." 
    def __str__(self):
        return self.msg
    

class ClassificationLogger:

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager
        # generated during log_classification_task:
        self.classification_task = None
        self.labels_image = None # initialized in _get_segmentation_task

    def _get_segmentation_task(self,
                               model_folder, 
                                image_filename,
                                image_channel,
                                parameters):
        # TODO: this repeats the code in src/segmentation/logger.py 
        #       should be refactored into a common function

        # get the segmentation model
        segmentation_model = self.db.get_segmentation_model(model_folder)
        if segmentation_model is None:
            raise Exception(f"Segmentation model {model_folder} not found in database.")
        
        # get the image that was segmented
        image = self.db.get_image(image_channel, image_filename)
        if image  is None:
            raise Exception(f"Image {image_filename} with channel {image_channel} not found in database.")
        
        # check that the segmentation task doesn't already exist in the database
        segmentation_task = self.db.get_segmentation_task(image_id = image.id, 
                                        segmentation_model_id = segmentation_model.id, 
                                        **parameters)
        if segmentation_task is None:
            raise Exception(f"Segmentation task for image {image_filename} with channel {image_channel} and segmentation model {model_folder} not found in database.")
        
        self.labels_image = self.db.get_image(id = segmentation_task.labels_image_id)
        if self.labels_image is None:
            raise Exception(f"Labels image for segmentation task {segmentation_task.id} not found in database.")
        
        return segmentation_task
    
    def _process_classification_method_params(self, classification_method: str, parameters: dict):
        if classification_method == "prob_thr":
            assert "prob_thr" in parameters, "Parameters for classification method 'prob_thr' must contain prob_thr."
            assert "class_name" in parameters, "Parameters for classification method 'prob_thr' must contain class_name."
            return
        elif classification_method == "intensity_thr":
            assert "images" in parameters, "Parameters for classification method 'intensity_thr' must contain images."
            assert "thresholds" in parameters, "Parameters for classification method 'intensity_thr' must contain thresholds."
        else:
            raise Exception(f"Classification method {classification_method} not supported.")

    def log_classification_task(self,
                                classification_method: str,
                                segmentation_model_folder: str | Path,
                                segmentation_image_filename: str,
                                segmentation_image_channel: int,
                                segmentation_parameters: dict = {},
                                classifier_parameters: dict = {} ) -> None:
        """
        Log a classification run in the database.

        Args:
            classification_method: the classification method
            segmentation_model_folder: the folder of the segmentation model
            segmentation_image_filename: the filename of the image to segment
            segmentation_image_channel: the channel of the image to segment
            segmentation_parameters: the parameters of the segmentation model,
                must contain n_tiles, prob_thr and scale_probability
            classifier_parameters: the parameters of the classifier,
        """

        # check that parameters have the right format
        self._process_classification_method_params(classification_method, classifier_parameters)

        # get segmentation task
        segmentation_task = self._get_segmentation_task(segmentation_model_folder, 
                                                        segmentation_image_filename,
                                                        segmentation_image_channel,
                                                        segmentation_parameters)

        # check that the classification task doesn't already exist in the database
        if self.db.get_classification_task(classification_method = classification_method,
                                            segmentation_task_id = segmentation_task.id, 
                                            parameters = classifier_parameters) is not None:
            raise ClassificationTaskAlreadyInDatabase(classification_method, 
                                                      segmentation_task.id, 
                                                      classifier_parameters)
        
        # add the classification task
        self.classification_task = self.db.add_classification_task(classification_method, 
                                                                   segmentation_task.id, 
                                                                   classifier_parameters)

    def log_classification(self, classification: Classification) -> None:
        """
        Log a classification run in the database. Takes a Classification object as input.

        Args:
            classification: the classification to log
        """
        # log classification task
        self.log_classification_task(classification.method, 
                                     classification.segmentation.model_folder,
                                     classification.segmentation.image.file_name,
                                     classification.segmentation.image.channel,
                                     classification.segmentation.parameters.as_dict(),
                                     classification.parameters)


    def _construct_class_image_filename(self, class_name: str) -> str:
        """
        Construct the filename of the class image: will add the class_name tag and segmentation task id to the original image filename.

        Returns:
            the filename of the labels image
        """
        suffix = Path(self.labels_image.file_name).suffix
        ct_id = self.classification_task.id
        class_image_filename = self.labels_image.file_name.replace(suffix, f"_{class_name}_ct{ct_id}{suffix}")

        return class_image_filename
    
    def _generate_class_images(self, 
                               classified_centroid_labels: list[int], 
                               assigned_class_names: list[str]) -> tuple[str, str, npt.NDArray]:
        """
        Create the class images for each class_name.
        """
        # get the unique class names
        class_names = list(set(assigned_class_names))
        img = tif.imread(self.labels_image.file_name)

        for class_name in class_names:
            # TODO : this is a bit inefficient, should be refactored (class_label_id stuff)
            class_image_filename = self._construct_class_image_filename(class_name)
            centroid_label_ids = [centroid_label_id for centroid_label_id, class_name_ in 
                                  zip(classified_centroid_labels, assigned_class_names) 
                                  if class_name_ == class_name]
            # grab centroid label for each centroid id
            class_img = create_class_image(img, centroid_label_ids)
            yield (class_name, class_image_filename, class_img)

    def save_and_log_class_image(self, 
                                  class_name: str,
                                  class_image_filename: str, 
                                  class_img: npt.NDArray):
        
        save_image_as_uint16(class_img, class_image_filename)

        channel = 0
        stage = "classified"
        label = "labels"
        transformed = False
        image_space_name = self.db._get_or_create_image_space(id = self.labels_image.image_space_id).name

        class_img = self.db.add_image(channel, class_image_filename, 
                                         stage, 
                                         label, 
                                         transformed, 
                                         image_space_name)
        
        # add to the ClassifiedImage table:
        self.db.add_classified_image(class_img.id, class_name, self.classification_task.id)

    def create_save_and_log_class_images(self, classified_centroid_labels, class_names):
        filenames = []
        for class_name, class_image_filename, class_img in \
            self._generate_class_images(classified_centroid_labels, class_names):

            self.save_and_log_class_image(class_name, class_image_filename, class_img)
            filenames.append(class_image_filename)
        return filenames

    def log_classified_centroids(self,
                                 classified_centroids_ids: list[int], 
                                 assigned_class_names: list[str]) -> None:
        """
        Log the classified centroids in the database.

        Args:
            classified_centroids_ids: the list of centroids ids
            assigned_class_names: the list of class names for each centroid id
        """
        # check that the classification task is logged
        if self.classification_task is None:
            raise Exception("Classification task not logged. Call log_classification_task first.")
        
        # log the class centroids
        self.db.add_classified_centroids(classified_centroids_ids,
                                            assigned_class_names,
                                            self.classification_task.id)