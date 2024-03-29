"""
First script in the analysis pipeline:
1. [this one] save_transform_class_centroids.py
2. assign_synapses_to_neurons.py
3. analyze_assignments.py

This script saves the class centroids to a csv file with the following columns:
centroid_id, x, y, z
x, y, z are the coordinates of the centroid either in pixel or in the image space resolution.

Also applies the transformation to the centroids and saves the transformed centroids to a csv file.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

# from sprinkle.registration.ants_utils import transform_points
import ants
from sprinkle.db.manager import DatabaseManager
from sprinkle.db.models import Centroids

def get_centroids(db: DatabaseManager, 
                  segmentation_id: int | None = None,
                  classification_id: int | None = None,
                  class_name: str | None = None) -> pd.DataFrame:
    """
    Get the centroids of the class labels.

    Args:
        db (DatabaseManager): the database manager to use
        segmentation_id (int, optional): the id of the segmentation to get the centroids from
        classification_id (int, optional): the id of the classification to get the centroids from
        class_name (str, optional): the name of the class to get the centroids for
    """
    # get the centroids
    if class_name is None:
        assert segmentation_id is not None, "segmentation_id must be provided if class_name is not provided."
        centroids = db.get_centroids(segmentation_id)
    else:
        assert classification_id is not None, "classification_id must be provided if class_name is provided."
        centroid_classes = db.get_classified_centroids(classification_id)
        # filter the centroids
        centroid_ids = [c.centroid_id for c in centroid_classes if c.class_name == class_name]
        centroids = db.get_centroids(centroid_ids = centroid_ids)
    return centroids

def get_coordinates_pixel(centroids: [Centroids]) -> pd.DataFrame:
    """
    Get the coordinates of the centroids in pixel space.

    Args:
        centroids ([Centroids]): the centroids to get the coordinates for

    Returns:
        pd.DataFrame: a dataframe containing the coordinates of the centroids
    """
    coordinates_df = pd.DataFrame([{"centroid_id": c.id, "x": c.x, "y": c.y, "z": c.z} for c in centroids])
    return coordinates_df

def get_resolution_xyz(db: DatabaseManager, 
                    image_space_name: str) -> list[float, float, float]:
    """
    Get the resolution of the image space in XYZ order.
    """
    # get the image space
    image_space = db._get_or_create_image_space(name = image_space_name)
    # get the resolution
    resolution_xyz = [image_space.resolution_x, image_space.resolution_y, image_space.resolution_z]
    return resolution_xyz

def get_resolution_zyx(db: DatabaseManager, 
                    image_space_name: str) -> list[float, float, float]:
    """
    Get the resolution of the image space in ZYX order.
    """
    # get the image space
    image_space = db._get_or_create_image_space(name = image_space_name)
    # get the resolution
    resolution_zyx = [image_space.resolution_z, image_space.resolution_y, image_space.resolution_x]
    return resolution_zyx


def get_coordinates_image_space(db: DatabaseManager,
                                centroids: [Centroids], 
                                image_space_name: str) -> pd.DataFrame:
    """
    Get the coordinates of the centroids in image space.

    Args:
        db (DatabaseManager): the database manager to use
        centroids ([Centroids]): the centroids to get the coordinates for
        image_space_name (str): the name of the image space to get the coordinates in 

    Returns:
        pd.DataFrame: a dataframe containing the coordinates of the centroids
    """
    resolution_xyz = get_resolution_xyz(db, image_space_name)
    # get the coordinates
    coordinates = get_coordinates_pixel(centroids)
    # convert to image space
    coordinates[["x", "y", "z"]] = coordinates[["x", "y", "z"]].multiply(resolution_xyz)
    return coordinates

def create_centroids_csv(db: DatabaseManager, 
                        out_file: str | Path,
                        image_space_name: str,
                        segmentation_id: int | None = None,
                        classification_id: int | None = None,
                        class_name: str | None = None) -> None:
    """
    Create a csv file containing the class centroids.

    Args:
        db (DatabaseManager): the database manager to use
        classification_id (int, optional): the id of the classification to get the centroids from
        class_name (str, optional): the name of the class to get the centroids for
        image_space_name (str): the name of the image space to get the coordinates in
        out_file (str | Path): the path to save the csv file to. Will be saved with the suffix _pix and _um for pixel and image space coordinates respectively.
    """
    assert segmentation_id is not None or classification_id is not None, "segmentation_id or classification_id must be provided."

    if segmentation_id is not None:
        assert classification_id is None, "classification_id cannot be provided if segmentation_id is provided."
    if classification_id is not None:
        assert segmentation_id is None, "segmentation_id cannot be provided if classification_id is provided."
    
    # get the centroids
    centroids = get_centroids(db, segmentation_id = segmentation_id, classification_id = classification_id, class_name = class_name)
    # get the coordinates
    coordinates_pix = get_coordinates_pixel(centroids)
    coordinates_um = get_coordinates_image_space(db, centroids, image_space_name)
    # save to csv
    out_file = Path(out_file)
    # add tag pix to the outfile name
    out_file_pix = out_file.with_name(out_file.stem + "_pix" + out_file.suffix)
    coordinates_pix.to_csv(out_file_pix, index=False)
    # add tag um to the outfile name
    out_file_um = out_file.with_name(out_file.stem + "_um" + out_file.suffix)
    coordinates_um.to_csv(out_file_um, index=False)

# def transform_coordinates_from_csv(db: DatabaseManager, 
#                           coordinates: pd.DataFrame,
#                           registration_id: int,
#                           fixed_image_space_name: str,
#                           out_file: str | Path) -> None:
#     """
#     Transform the coordinates using the given registration and save to a csv file.

#     Args:
#         db: the database manager to use
#         coordinates: the coordinates to transform
#         registration_id: the id of the registration to use for the transformation
#         resolution_zyx: the resolution of the reference image
#         fixed_image_space_name: the name of the fixed image space (into which the coordinates will be transformed)
#         out_file: the path to save the csv file to
#     """
#     # get the registration
#     registration = db.get_registration_task(id = registration_id)
#     # get the transformation list
#     transformation_list = registration.forward_transformations
#     # sort the transformation list by order in descending order 
#     # (the point transformation should be applied in reverse order to the image transformation)
#     transformation_list.sort(key = lambda t: t.order, reverse = True)
#     # get the file names
#     tranform_files = [t.file_name for t in transformation_list]
    
#     # apply the transform
#     n_dim = 3
#     whichtoinvert = None
#     verbose = True
#     transformed_points = ants.apply_transforms_to_points(n_dim, 
#                                                          coordinates, 
#                                                          tranform_files,
#                                                          whichtoinvert = whichtoinvert, 
#                                                          verbose=verbose)
#     # undo resolution
#     resolution_zyx = get_resolution_zyx(db, fixed_image_space_name)
#     transformed_points[["z","y","x"]] = transformed_points[["z","y","x"]].divide(resolution_zyx)
#     # save to csv
#     transformed_points.to_csv(out_file, index=False)

def transform_coordinates(db: DatabaseManager,
                          out_file: str | Path,
                          registration_id: int,
                          moving_image_space_name: str,
                          fixed_image_space_name: str,
                          segmentation_id: int | None = None, 
                          classification_id: int | None = None, 
                          class_name: str | None = None) -> None:
    """
    Transform the coordinates using the given registration and save to a csv file.
    It is a bit complecated. 
    Read this: https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians

    Args:
        db: the database manager to use
        coordinates: the coordinates to transform
        registration_id: the id of the registration to use for the transformation
        resolution_zyx: the resolution of the reference image
        moving_image_space_name: the name of the moving image space (from which the coordinates will be transformed)
        fixed_image_space_name: the name of the fixed image space (into which the coordinates will be transformed)
        out_file: the path to save the csv file to
    """
    # get the centroids
    assert segmentation_id is not None or classification_id is not None, "segmentation_id or classification_id must be provided."
    assert segmentation_id is None or classification_id is None, "segmentation_id and classification_id cannot be provided at the same time."
    centroids = get_centroids(db, 
                            segmentation_id = segmentation_id, 
                            classification_id = classification_id, 
                            class_name = class_name)

    # get the coordinates
    coordinates = get_coordinates_image_space(db, centroids, moving_image_space_name)

    # get the transformation list
    registration = db.get_registration_task(id = registration_id)
    transformation_list = registration.inverse_transformations
    # sort the transformation list by order in ascending order
    transformation_list.sort(key = lambda t: t.order, reverse = False)
    # get the transformation file names
    tranform_files = [t.file_name for t in transformation_list]
    
    # apply the transform
    n_dim = 3
    # If the transform list is a matrix followed by a warp field,
    # whichtoinvert defaults to (True,False). Which is what we want, so we don't need to provide it.
    whichtoinvert = None
    verbose = True
    transformed_points = ants.apply_transforms_to_points(n_dim, 
                                                         coordinates, 
                                                         tranform_files,
                                                         whichtoinvert = whichtoinvert, 
                                                         verbose=verbose)
    # save to csv
    out_file = Path(out_file)
    # add um to the outfile name
    out_file_um = out_file.with_name(out_file.stem + "_um" + out_file.suffix)
    transformed_points.to_csv(out_file_um, index=False)

    # undo resolution
    resolution_xyz = get_resolution_xyz(db, fixed_image_space_name)
    transformed_points[["x","y","z"]] = transformed_points[["x","y","z"]].divide(resolution_xyz)

    # save to csv
    # add pix to the outfile name
    out_file_pix = out_file.with_name(out_file.stem + "_pix" + out_file.suffix)
    transformed_points.to_csv(out_file_pix, index=False)

def create_inhibitory_neurons_csv():
    db = DatabaseManager("/mnt/sprinkle/datasets/2024/Subject_1-26MC_test/1-26MC_pipeline_full_size.sqlite",
                            echo = False)
    classification_id = 1
    class_name = "inhibitory_neuron"
    image_space_name = "fixed"
    out_file = "/mnt/sprinkle/datasets/2024/Subject_1-26MC/Centroids_hoechst_labels_inhibitory_neuron.csv"

    create_centroids_csv(db, out_file, image_space_name, classification_id = classification_id, class_name = class_name)

def transform_inhibitory_neurons(db_path: str | Path,
                                 out_file: str | Path) -> None:
    
    db = DatabaseManager(db_path, echo = False)

    classification_id = 1
    class_name = "inhibitory_neuron"
    if "1-26MC" in db_path:
        registration_id = 3
    else:
        registration_id = 2
    out_file = out_file
    fixed_image_space_name = "live2"
    moving_image_space_name = "fixed"
    transform_coordinates(db, out_file, registration_id, moving_image_space_name, fixed_image_space_name, classification_id = classification_id, class_name = class_name)

def create_tp2_synapses_csv(db_path: str | Path,
                            out_file: str | Path) -> None:
    db = DatabaseManager(db_path, echo = False)
    if "1-26XY" in db_path:
        segmentation_id = 4
    if "1-26WM" in db_path:
        segmentation_id = 3
    else:
        segmentation_id = 3
    image_space_name = "live1"

    create_centroids_csv(db, out_file, image_space_name, segmentation_id = segmentation_id)

def transform_tp1_synapses(db_path: str | Path,
                           out_file: str | Path) -> None:
    db = DatabaseManager(db_path, echo = False)
    
    if "1-26MC" in db_path:
        segmentation_id = 2
        registration_id = 4
    elif "1-26XY" in db_path:
        registration_id = 1
        segmentation_id = 3
    elif "1-26WM" in db_path:
        registration_id = 2 # SyN
        segmentation_id = 2
    else:
        segmentation_id = 2
        registration_id = 1
    out_file = out_file
    fixed_image_space_name = "live2"
    moving_image_space_name = "live1"
    transform_coordinates(db, out_file, registration_id, moving_image_space_name, fixed_image_space_name, segmentation_id = segmentation_id)

def transform_all_nuclei(db_path: str | Path,
                         out_file: str | Path) -> None:
    db = DatabaseManager(db_path, echo = False)
    # specify nuclei segmentation id
    # and the registration from fixed to live2
    if "1-26MC" in db_path:
        segmentation_id = 1
        registration_id = 3
    elif "1-26XY" in db_path:
        registration_id = 3
        segmentation_id = 2
    elif "1-26WM" in db_path:
        registration_id = 4 # SyN
        segmentation_id = 1
    else:
        segmentation_id = 1
        registration_id = 2 
    out_file = out_file
    fixed_image_space_name = "live2"
    moving_image_space_name = "fixed"
    transform_coordinates(db, out_file, registration_id, moving_image_space_name, fixed_image_space_name, segmentation_id = segmentation_id)

def process_test():
    db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26MC_test/1-26MC_pipeline_full_size.sqlite"

    inhibitory_neuron_out_file = "/mnt/sprinkle/datasets/2024/Subject_1-26MC/Centroids_1-26NA_1-26NA_fixed_channel3_hoechst_labels_st1_inhibitory_neuron_ct1_SyN_multiLabel_tt5.csv"
    tp1_synapse_out_file = "/mnt/sprinkle/datasets/2024/Subject_1-26MC/Centroids_1-26N4_1-26N4_live_tp1_labels_st2_SyN_multiLabel_tt6.csv"
    tp2_synapse_out_file = "/mnt/sprinkle/datasets/2024/Subject_1-26MC/Centroids_1-26N6_1-26N6_live_tp2.csv"
    transform_inhibitory_neurons(db_path = db_path, out_file = inhibitory_neuron_out_file)
    transform_tp1_synapses(db_path = db_path, out_file = tp1_synapse_out_file)
    create_tp2_synapses_csv(db_path = db_path, out_file = tp2_synapse_out_file)

def process_subject(db_path: str | Path) -> None:
    
    # create centroids folder inside the subject folder
    subject_folder = Path(db_path).parent
    centroids_folder = subject_folder / "centroids"
    centroids_folder.mkdir(exist_ok = True)

    nuclei_out_file = centroids_folder / "nuclei_in_tp2.csv"
    tp1_synapse_out_file = centroids_folder / "tp1_synapses_in_tp2.csv"
    tp2_synapse_out_file = centroids_folder / "tp2_synapses.csv"
    transform_all_nuclei(db_path, nuclei_out_file)
    transform_tp1_synapses(db_path, tp1_synapse_out_file)
    create_tp2_synapses_csv(db_path, tp2_synapse_out_file)
                            


if __name__ == "__main__":
    db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26WM/1-26WM_pipeline_full_size.sqlite"
    process_subject(db_path)

    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26MC/1-26MC_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26ME/1-26ME_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26MG/1-26MG_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26PG/1-26PG_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26PJ/1-26PJ_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26PM/1-26PM_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26PP/1-26PP_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26PT/1-26PT_pipeline_full_size.sqlite"
    # process_subject(db_path)
    # db_path = "/mnt/sprinkle/datasets/2024/Subject_1-26PW/1-26PW_pipeline_full_size.sqlite"
    # process_subject(db_path)

