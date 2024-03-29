from __future__ import annotations

from pathlib import Path
from sprinkle.registration.registration import Registration
from sprinkle.registration.transformation import ImageTransformation
from sprinkle.db.manager import DatabaseManager
from sprinkle.registration.logger import RegistrationLogger, RegistrationTaskAlreadyInDatabase
from sprinkle.registration.logger import ImageTransformationLogger, ImageTransformationTaskAlreadyInDatabase

def _prepare_registration_parameters(d: dict, db: DatabaseManager):
    """
    Prepares the parameters for the registration: makes sure image dicts contain the resolution_xyz, file_name and channel keys and nothing else.
    """
    dd = d.copy()
    for image_role in ["fixed_image", "moving_image"]:
        image_dict = dd[image_role]
        image = db.get_image(channel = image_dict["channel"],
                                    file_name = image_dict["file_name"])
        if "resolution_xyz" not in image_dict or image_dict["resolution_xyz"] is None:
            image_space = db._get_or_create_image_space(id = image.image_space_id)
            resolution_xyz = [image_space.resolution_x, 
                                    image_space.resolution_y, 
                                    image_space.resolution_z]
            image_dict["resolution_xyz"] = resolution_xyz
            # make sure the image dict only contains the necessary information
        dd[image_role] = {  
                            "image_space": image_dict["image_space"],
                            "label": image_dict["label"],
                            "file_name": image.file_name,
                            "channel": image.channel,
                            "resolution_xyz": image_dict["resolution_xyz"]}
    return dd

def run_registration_from_dict(d: dict, db: DatabaseManager,
                               skip_exist: bool = True,
                                  overwrite: bool = False) -> None:
    """
    Runs the registration task defined in the config dictionary,
    use this to run the registration task as a part of the pipeline.

    Args:
        d: the dictionary defining the registration task
        db: the database manager
        skip_exist: whether to skip the registration task if it is already in the database
        overwrite: whether to overwrite the registration task if it is already in the database
    """
    d = _prepare_registration_parameters(d, db)
    registration = Registration.from_dict(d)
    reg_logger = RegistrationLogger(db)

    try:
        # log the registration task in the database,
        # will raise an exception if the model or image are not found
        reg_logger.log_registration(registration)

    except RegistrationTaskAlreadyInDatabase:
        if skip_exist:
            print(f"Registration task for fixed image {repr(registration.fixed_image)} and moving image {repr(registration.moving_image)} with registration type {registration.registration_type} is already in the database. The task will be skipped.")
            return
        elif overwrite:
            # TODO: implement overwriting
            print(f"Overwriting registration task for fixed image {repr(registration.fixed_image)} and moving image {repr(registration.moving_image)} with registration type {registration.registration_type}. Existing transformation will be overwritten.")
            raise NotImplementedError
        else:
            raise

    # run the registration
    registration_tag = f"{registration.registration_type}_{reg_logger.registration_task.id}_"
    if "transformation_folder" in d:
        transformation_folder = (Path(d["transformation_folder"]) / registration_tag ).as_posix()
    else:
        transformation_folder = registration_tag
    fwdtransforms, invtransforms = registration.run(transformation_folder = transformation_folder)

    # save the forward and inverse transformations in the database
    reg_logger.log_transformations(fwdtransforms, invtransforms)

def _prepare_transformation_parameters(d: dict, db: DatabaseManager):
    dd = d.copy()
    image = db.get_image(channel = dd["image"]["channel"],
                                file_name = dd["image"]["file_name"])
    if "resolution_xyz" not in dd["image"] or dd["image"]["resolution_xyz"] is None:
        image_space = db._get_or_create_image_space(id = image.image_space_id)
        resolution_xyz = [image_space.resolution_x, 
                                image_space.resolution_y, 
                                image_space.resolution_z]
        dd["image"]["resolution_xyz"] = resolution_xyz
        # make sure the image dict only contains the necessary information
    dd["image"] = {
                    "image_space": dd["image"]["image_space"],
                    "label": dd["image"]["label"],
                    "file_name": image.file_name,
                    "channel": image.channel,
                    "resolution_xyz": dd["image"]["resolution_xyz"]}
    # get registration task
    reg_dd = dd["transforms"]["registration"]
    fixed_image = db.get_image(channel = reg_dd["fixed_image"]["channel"],
                                file_name = reg_dd["fixed_image"]["file_name"])
    
    fixed_image_space = db._get_or_create_image_space(id = image.image_space_id)
    fixed_resolution_xyz = [fixed_image_space.resolution_x, 
                            fixed_image_space.resolution_y, 
                            fixed_image_space.resolution_z]
    
    moving_image = db.get_image(channel = reg_dd["moving_image"]["channel"],
                                file_name = reg_dd["moving_image"]["file_name"])
    
    registration_task = db.get_registration_task(fixed_image_id = fixed_image.id,
                                                moving_image_id = moving_image.id,
                                                registration_type = reg_dd["registration_type"])
    # get transforms
    if dd["transforms"]["direction"] == "forward":
        transforms = registration_task.forward_transformations
    elif dd["transforms"]["direction"] == "inverse":
        transforms = registration_task.inverse_transformations
    else:
        raise Exception(f"Unknown transformation type {dd['transforms']['direction']}. Must be one of 'forward' or 'inverse'.")
    
    # assemble the dictionary
    transform_files = [transform.file_name for transform in transforms]
    order = [transform.order for transform in transforms]
    sorted_transforms = [transform for _, transform in sorted(zip(order, transform_files))]

    ddd = {"image": dd["image"],
           "reference_image": { "image_space": reg_dd["fixed_image"]["image_space"],
                                "label": reg_dd["fixed_image"]["label"],
                                "file_name": fixed_image.file_name,
                                "channel": fixed_image.channel,
                                "resolution_xyz": fixed_resolution_xyz},
            "interpolation": dd["interpolation"],
            "transformation_list": sorted_transforms,
            "direction": dd["transforms"]["direction"]}

    return ddd

def run_image_transformation_from_dict(d: dict, db: DatabaseManager,
                                       skip_exist: bool = True,
                                       overwrite: bool = False) -> None:
    """
    Runs the transformation task defined in the config dictionary,
    use this to run the transformation task as a part of the pipeline.

    Args:
        d: the dictionary defining the transformation task
        db: the database manager
        skip_exist: whether to skip the transformation task if it is already in the database
        overwrite: whether to overwrite the transformation task if it is already in the database
    """
    d_trm = _prepare_transformation_parameters(d, db)
    d_reg = _prepare_registration_parameters(d["transforms"]["registration"], db)

    trm_logger = ImageTransformationLogger(db)
    registration = Registration.from_dict(d_reg)
    transformation = ImageTransformation.from_dict(d_trm)

    try:
        # log the registration task in the database,
        # will raise an exception if the model or image are not found
        trm_logger.log_transformation(registration, transformation)

    except ImageTransformationTaskAlreadyInDatabase:
        if skip_exist:
            #TODO: write a better message
            print(f"Transformation task for image {transformation.image.file_name}, channel {transformation.image.channel} with reference image {transformation.reference_image.file_name}, channel {transformation.reference_image.channel} and {registration.registration_type} registration is already in the database. The task will be skipped.")
            return
        elif overwrite:
            # TDOD: implement overwriting
            print(f"Overwriting transformation task for image {repr(d['image'])} with reference image {repr(d['reference_image'])}. Existing transformation will be overwritten.")
            raise NotImplementedError
        else:
            raise

    # run the transformation
    transformed_image = transformation.run()

    # save the transformed image 
    trm_logger.log_transformed_image(transformed_image)