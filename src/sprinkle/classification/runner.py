from __future__ import annotations

from pathlib import Path
from sprinkle.classification.classification import Classification
from sprinkle.db.manager import DatabaseManager
from sprinkle.classification.logger import ClassificationLogger, ClassificationTaskAlreadyInDatabase

def _prepare_classifier_parameters(d: dict, db: DatabaseManager):
    """
    Prepares the parameters for the classifier.
    """
    dd = d.copy()
    if dd["method"] == "prob_thr":
        return dd
    elif dd["method"] == "intensity_thr":
        for image_label, image_dict in dd["parameters"]["images"].items():
            # get image parameters from the database:
            # 1. get image, make sure it exists
            assert "channel" in image_dict, "The channel of the image must be specified."
            assert "file_name" in image_dict, "The file name of the image must be specified."
            image = db.get_image(channel = image_dict["channel"],
                                    file_name = image_dict["file_name"])
            assert image is not None, f"The image with channel {image_dict['channel']} and file name {image_dict['file_name']} is not in the database."

            # 2. get image type and space to get label, image space and resolution
            image_type = db._get_or_create_image_type(id = image.image_type_id)
            image_space = db._get_or_create_image_space(id = image.image_space_id)
            resolution_xyz = [image_space.resolution_x, 
                              image_space.resolution_y, 
                              image_space.resolution_z]

            # 3. create image dict
            dd["parameters"]["images"][image_label] = {
                                        "file_name": image.file_name,
                                        "channel": image.channel,
                                        "label": image_type.label,
                                        "image_space": image_space.name,
                                        "resolution_xyz": resolution_xyz}
        return dd

def _prepare_classifier_input(classification: Classification, db: DatabaseManager):
    """
    Prepares the arguments for the classifier.

    Args:
        classification: the classification task
    """
    def get_centroids_for_classififcation(classification):
        segmentation_model = db.get_segmentation_model(classification.segmentation.model_folder)
        image = db.get_image(classification.segmentation.image.channel,
                                classification.segmentation.image.file_name)
        segmentation_task = db.get_segmentation_task(image_id = image.id,
                                segmentation_model_id = segmentation_model.id,
                                n_tiles = classification.segmentation.parameters.n_tiles,
                                prob_thr = classification.segmentation.parameters.prob_thr,
                                scale_probability = classification.segmentation.parameters.scale_probability)
        
        centroids = db.get_centroids(segmentation_task.id)
        return centroids
    
    if classification.method == "prob_thr":
        centroids = get_centroids_for_classififcation(classification)
        centroid_ids = [centroid.id for centroid in centroids]
        probabilities = [centroid.probability for centroid in centroids]
        centroid_labels = [centroid.label for centroid in centroids]

        return centroid_ids, centroid_labels, probabilities

    if classification.method == "intensity_thr":
        centroids = get_centroids_for_classififcation(classification)
        centroid_ids = [centroid.id for centroid in centroids]
        centroid_zyx = [[centroid.z, centroid.y, centroid.x] for centroid in centroids]
        centroid_labels = [centroid.label for centroid in centroids]

        return centroid_ids, centroid_labels, centroid_zyx

def run_classification_from_dict(d: dict, db: DatabaseManager,
                                 skip_exist: bool = True,
                                    overwrite: bool = False) -> None:
    """
    Runs the classification task defined in the config dictionary,
    use this to run the classification task as a part of the pipeline.

    Args:
        d: the dictionary defining the classification task
        db: the database manager
        skip_exist: whether to skip the classification task if it is already in the database
        overwrite: whether to overwrite the classification task if it is already in the database
    """
    d = _prepare_classifier_parameters(d, db)
    classification = Classification.from_dict(d)
    cls_logger = ClassificationLogger(db)

    try:
        # log the classification task in the database,
        # will raise an exception if the model or image are not found
        cls_logger.log_classification(classification)
    except ClassificationTaskAlreadyInDatabase:
        if skip_exist:
            print(f"Classification task for segmentation task for image {classification.segmentation.image.file_name}, channel{classification.segmentation.image.channel} with classification method {classification.method} is already in the database. The task will be skipped.")
            return
        elif overwrite:
            # TODO: implement overwriting
            print(f"Overwriting classification task for segmentation task for image {classification.segmentation.image.file_name}, channel{classification.segmentation.image.channel} with classification method {classification.method}. Existing class image will be overwritten.")
            raise NotImplementedError
        else:
            raise

    # prepare the arguments for the classification method
    cls_input = _prepare_classifier_input(classification, db)
    # run the classification
    class_centroid_ids, classified_centroid_labels, class_names = classification.run(*cls_input)

    # save the results in the database and on disk
    cls_logger.log_classified_centroids(class_centroid_ids, class_names)
    cls_logger.create_save_and_log_class_images(classified_centroid_labels, class_names)