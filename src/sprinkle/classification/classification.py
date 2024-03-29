"""
This module contains the function and classes to classify the synapse and cells.
"""
from __future__ import print_function, unicode_literals, absolute_import, division, annotations

import numpy as np
import numpy.typing as npt
import yaml
from dataclasses import dataclass, field
from pathlib import Path

from sprinkle.segmentation.segmentation import Segmentation
from sprinkle.image_utils import Image

@dataclass
class Classification:
    method: str
    segmentation: Segmentation
    parameters: dict
    # at the moment only one method (ProbabilityThreshold) is supported, 
    # so classifier type is set to ProbabilityThreshold:
    classifier: ProbabilityThreshold | IntensityThreshold =  field(init=False)

    def __post_init__(self):
        """
        Initializes the classification methods.
        """
        if self.method == "prob_thr":
            self.classifier = ProbabilityThreshold.from_dict(self.parameters)
        elif self.method == "intensity_thr":
            self.classifier = IntensityThreshold.from_dict(self.parameters)
        else:
            raise ValueError(f"Classification method {self.method} not supported.")
        
    @classmethod
    def from_dict(cls, d):
        """
        Creates a Classification from a dictionary.
        """
        d = d.copy()
        d['segmentation'] = Segmentation.from_dict(d['segmentation'])
        return cls(**d)

    def run(self, *args) -> tuple[list[int], list[str]]:
        """
        Runs the classification.

        Args:
            *args: the arguments to pass to the classification method, 
                see the classification methods for details.
        """
        # run the classification
        class_centroid_ids, class_centroid_labels, class_names = self.classifier.run(*args)
        return class_centroid_ids, class_centroid_labels, class_names

@dataclass
class ProbabilityThreshold:
    """
    A class to perform probability thresholding.
    Attributes:
        prob_thr: float
        class_name: str
    """
    prob_thr: float
    class_name: str

    @classmethod
    def from_dict(self, d: dict):
        """
        Loads the parameters from a dictionary.

        Args:
            d (dict): the dictionary containing the parameters, can contain other parameters, 
            only "prob_thr" and "class_name" will be used.
        """
        assert "prob_thr" in d, "Parameters for classification method 'prob_thr' must contain prob_thr."
        assert "class_name" in d, "Parameters for classification method 'prob_thr' must contain class_name."
        prob_thr_params = {"prob_thr": d["prob_thr"], "class_name": d["class_name"]}
        return self(**prob_thr_params)

    def run(self,
            centroid_ids:list[int] ,
            centroid_labels: list[int],
            probability: list[float]) -> tuple[list[int], list[str]]:
        """
        Runs the probability thresholding.

        Args:
            centroid_ids (list[int]): the list of centroids id
            centroid_labels (list[int]): the list of centroids labels
            probability (list[float]): the list of probabilities

        Returns:
            list[int]: the list of centroids id with probability equal or above the threshold
            list[str]: the list of class names for the centroids id with probability equal or above the threshold
        """
        class_centroid_ids = [centroid_id for centroid_id, prob 
                              in zip(centroid_ids, probability) 
                              if prob >= self.prob_thr]
        class_centroid_labels = [centroid_label for centroid_label, prob
                                    in zip(centroid_labels, probability)
                                    if prob >= self.prob_thr]
        class_names = [self.class_name]*len(class_centroid_ids)
        return class_centroid_ids, class_centroid_labels, class_names
    
@dataclass
class IntensityThreshold:
    """
    A class to perform intensity thresholding based on multiple images.
    Attributes:
        images: dict of images to use for thresholding
        class_names: list(str) of class names to use for thresholding
        thresholds: dct(str, dict) a dictionary for each class name with the images and corresponding thresholds
    """
    images: dict[str, Image]
    class_names: list[str]
    thresholds: dict[str, dict[str, float]]

    @classmethod
    def from_dict(cls, d: dict):
        """
        Loads the parameters from a dictionary.

        Args:
            d (dict): the dictionary containing the parameters, can contain other parameters, 
                only "images", "class_names" and "thresholds" will be used.
                "images" need to be prepared to contain only the necessary information for the Image (file_name, channel, resolution_xyz).
        """
        assert "images" in d, "Parameters for classification method 'intensity_thr' must contain images."
        assert "thresholds" in d, "Parameters for classification method 'intensity_thr' must contain thresholds."
        for class_name in d["thresholds"]:
            assert len(d["thresholds"][class_name]) > 0, f"Parameters for classification method 'intensity_thr' must contain at least one threshold for class {class_name}."
            for thr_spec in d["thresholds"][class_name]:
                assert "image_label" in thr_spec
                assert ("prc_threshold" in thr_spec or "intensity_threshold" in thr_spec)
                assert "direction" in thr_spec

        # prepare images 
        images = {}
        for image_label, image_dict in d["images"].items():
            images[image_label] = Image(**image_dict)

        # prepare class_names and thresholds
        thresholds = {}
        class_names = []
        for class_name, thr_spec in d["thresholds"].items():
            class_names.append(class_name)

            thresholds[class_name] = []
            for thr in thr_spec:
                thresholds[class_name].append({"image_label": thr["image_label"],
                                               "direction": thr["direction"]})
                if "prc_threshold" in thr:
                    thresholds[class_name][-1]["prc_threshold"] = thr["prc_threshold"]
                elif "intensity_threshold" in thr:
                    thresholds[class_name][-1]["intensity_threshold"] = thr["intensity_threshold"]
                else:
                    raise ValueError(f"Threshold specification {thr} must contain either prc_threshold or intensity_threshold.")

        return cls(images = images, class_names = class_names, thresholds = thresholds)

    def run(self,
            centroid_ids:list[int] ,
            centroid_labels: list[int],
            centroids_zyx: list[list[float, float, float]]) -> tuple[list[int], list[str]]:
        """
        Runs the intensity thresholding.

        Args:
            centroid_ids: the list of centroids id
            centroid_labels: the list of centroids labels
            centroids_xyz: the list of centroid coordinates in pixels

        Returns:
            list[int]: the list of centroids id with intensity equal or above the threshold
            list[str]: the list of class names for the centroids id with intensity equal or above the threshold
        """
        # turn centroids into zyx array
        centroids_zyx = np.array(centroids_zyx, dtype=int)
        assert centroids_zyx.shape[1] == 3, "Centroids must be in zyx format."

        # grab the intensity for each centroid from each image
        intensity = {}
        for image_label, image in self.images.items():
            img = image.load()
            # get the intensity for each centroid
            intensity[image_label] = []
            for centroid_zyx in centroids_zyx:
                square_size = 1
                z, y, x = centroid_zyx
                intensity[image_label].append(img[z - square_size: z + square_size + 1,
                                                  y - square_size: y + square_size + 1,
                                                  x - square_size: x + square_size + 1].mean())
        
        # apply the thresholds
        class_centroids = {}
        for class_name in self.class_names:
            passed_threshold = []
            for thr in self.thresholds[class_name]:
                # get the centroid intensity values for the current image
                centroids_intensity = intensity[thr["image_label"]]
                # get the thr value for the current image
                if "prc_threshold" in thr:
                        thr_value = np.percentile(centroids_intensity, thr["prc_threshold"]*100)
                elif "intensity_threshold" in thr:
                    thr_value = thr["intensity_threshold"]
                else:
                    raise ValueError(f"Threshold specification {thr} must contain either prc_threshold or intensity_threshold.")
                
                # apply the threshold
                if thr["direction"] == "above":
                    passed_threshold.append(np.array(centroids_intensity) >= thr_value)
                elif thr["direction"] == "below":
                    passed_threshold.append(np.array(centroids_intensity) < thr_value)
                else:
                    raise ValueError(f"Threshold specification {thr} must contain direction 'above' or 'below'.")
            class_centroids[class_name] = np.logical_and.reduce(passed_threshold) # apply all thresholds to each centroid
            assert len(class_centroids[class_name]) == len(centroid_ids), "The length of the passed_threshold must be the same as the length of the centroid_ids."
        
        all_class_centroids = [class_centroids[class_name] for class_name in self.class_names]
        # check that each centroid is assigned to only one class
        assert np.all(np.sum (np.array(all_class_centroids), axis = 0) == 1 ), "Each centroid must be assigned to one and only one class."

        # get the class names for each centroid
        class_names = []
        for i_centroid, _ in enumerate(centroid_ids):
            # get the class name for the current centroid
            for class_name in self.class_names:
                if class_centroids[class_name][i_centroid]:
                    class_names.append(class_name)
                    break
        # make sure that the number of class names is the same as the number of centroids
        assert len(class_names) == len(centroid_ids), "Each centroid must be assigned to a class."

        return centroid_ids, centroid_labels, class_names

