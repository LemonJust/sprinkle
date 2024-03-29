from __future__ import annotations

from pathlib import Path
import yaml

from sprinkle.db.manager import DatabaseManager
from sprinkle.initialisation.runner import run_initialisation_from_dict
from sprinkle.segmentation.runner import run_segmentation_from_dict
from sprinkle.classification.runner import run_classification_from_dict
from sprinkle.registration.runner import run_registration_from_dict, run_image_transformation_from_dict

def run_pipeline_from_yaml(config_file: str | Path):
    """
    Runs the pipeline defined in the yaml config file. 
    The yaml structure should be as follows, defining the database file in addition to the segmentation task:

    db_file: path to the database file

    initialisation:
        raw_images:
            - file_name: path to the image file
              channel: channel to segment
              label: label of the image
              image_space: name of the image space
              resolution_xyz: resolution of the image space
        segmentation_models:
            - model_folder: folder with the saved StarDist3D model to use for segmentation
              model_type: the type of the model (e.g. StarDist3D)
              segmentation_type: the type of segmentation (e.g. nuclei)
              segmentation_training_id: the id of the segmentation training to use for the model (optional)
              parameters: a dict of parameters to use for segmentation: n_tiles, prob_thr, scale.
                n_tiles: number of tiles to use for prediction
                prob_thr: probability threshold for the prediction
                scale_probability: scale to use for saving the probability image

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
    # load the yaml file
    with open(config_file) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)

    # create or connect to the database
    if "db_file" not in d:
        raise ValueError("The database file must be specified in the config file.")
    db = DatabaseManager(d["db_file"], echo=False)

    # run the initialisation
    if "initialisation" in d:
        run_initialisation_from_dict(d["initialisation"], db)

    # run segmentations
    if "segmentations" in d:
        for sd in d["segmentations"]:
            run_segmentation_from_dict(sd["segmentation"], db)

    # run classifications
    if "classifications" in d:
        for cd in d["classifications"]:
            run_classification_from_dict(cd["classification"], db)
            print("___\nClassification done\n___")

    # run registrations
    if "registrations" in d:
        for rd in d["registrations"]:
            run_registration_from_dict(rd["registration"], db)

    # run transformations
    if "transformations" in d:
        for td in d["transformations"]:
            if "image" in td["transformation"]:
                run_image_transformation_from_dict(td["transformation"], db)
            elif "points" in td["transformation"]:
                raise NotImplementedError
            else:
                raise ValueError("Transformation must be either for an image or for points.")


def run_all():
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26MC/1-26MC_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)

    config = "/mnt/sprinkle/datasets/2024/Subject_1-26ME/1-26ME_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26MG/1-26MG_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26PG/1-26PG_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26PJ/1-26PJ_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26PM/1-26PM_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)

    config = "/mnt/sprinkle/datasets/2024/Subject_1-26PP/1-26PP_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)

    config = "/mnt/sprinkle/datasets/2024/Subject_1-26PT/1-26PT_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26PW/1-26PW_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)

def run_test():
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26MC_test/1-26MC_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)

if __name__ == "__main__":
    config = "/mnt/sprinkle/datasets/2024/Subject_1-26WM/1-26WM_pipeline_full_size.yaml"
    run_pipeline_from_yaml(config)
    # config = "/mnt/sprinkle/datasets/2024/Subject_1-26WJ/1-26WJ_pipeline_full_size.yaml"
    # run_pipeline_from_yaml(config)
    # config = "/mnt/sprinkle/datasets/2024/Subject_1-26WG/1-26WG_pipeline_full_size.yaml"
    # run_pipeline_from_yaml(config)
    # config = "/mnt/sprinkle/datasets/2024/Subject_1-26WE/1-26WE_pipeline_full_size.yaml"
    # run_pipeline_from_yaml(config)
    


