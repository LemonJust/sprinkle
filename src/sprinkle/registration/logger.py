from __future__ import annotations

from sprinkle.db.manager import DatabaseManager

from pathlib import Path
import numpy.typing as npt
from sprinkle.image_utils import save_image_as_uint16
from sprinkle.registration.registration import Registration
from sprinkle.registration.transformation import ImageTransformation

# supported registration types
# more info: https://antspy.readthedocs.io/en/latest/registration.html
ANTS_TRANSFORMS = ["Translation", 
                    "Rigid", 
                    "Similarity", 
                    "QuickRigid",
                    "DenseRigid",
                    "BOLDRigid",
                    "Affine",
                    "AffineFast",
                    "BOLDAffine",
                    "TRSAA",
                    "Elastic",
                    "ElasticSyN",
                    "SyN",
                    "SyNRA",
                    "SyNOnly",
                    "SyNCC",
                    "SyNBold",
                    "SyNBoldAff",
                    "SyNAggro",
                    "SyNAggroAff"]

class RegistrationTaskAlreadyInDatabase(Exception):
    """Registration task is already in the database."""
    def __init__(self, fixed_image_id: int, 
                 moving_image_id: int, 
                 registration_type: str):
        self.msg = f"The registration task with fixed_image_id {fixed_image_id}, moving_image_id {moving_image_id} and registration_type {registration_type} is already in the database." 
    def __str__(self):
        return self.msg

    
class RegistrationLogger:

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager
        # generated during log_registration_task:
        self.registration_task = None
        self.fixed_image = None
        self.moving_image = None
        self.forward_transformation = None
        self.inverse_transformation = None

    def log_registration_task(self,
                                fixed_image_filename: str | Path,
                                fixed_image_channel: int | None,
                                moving_image_filename: str | Path,
                                moving_image_channel: int | None,
                                registration_type: str) -> None:
            """
            Log a registration run in the database.
    
            Args:
                fixed_image_filename: the filename of the fixed image
                fixed_image_channel: the channel of the fixed image
                moving_image_filename: the filename of the moving image
                moving_image_channel: the channel of the moving image
                registration_type: the type of registration to perform, 
                    must be one of the types supported by ANTs
            """
            # check that registration type is valid
            assert registration_type in ANTS_TRANSFORMS, "Registration type must be one of supported types. For more info see https://antspy.readthedocs.io/en/latest/registration.html"

            # get fixed image 
            fixed_image_channel = 0 if fixed_image_channel is None else fixed_image_channel
            self.fixed_image = self.db.get_image(fixed_image_channel, fixed_image_filename)
            if self.fixed_image is None:
                raise Exception(f"Image {fixed_image_filename} with channel {fixed_image_channel} not found in database.")
            
            # get moving image
            moving_image_channel = 0 if moving_image_channel is None else moving_image_channel
            self.moving_image = self.db.get_image(moving_image_channel, moving_image_filename)
            if self.moving_image is None:
                raise Exception(f"Image {moving_image_filename} with channel {moving_image_channel} not found in database.")
            
            # check that the registration task doesn't already exist in the database
            if self.db.get_registration_task(fixed_image_id = self.fixed_image.id,
                                            moving_image_id = self.moving_image.id,
                                            registration_type = registration_type) is not None:
                raise RegistrationTaskAlreadyInDatabase(self.fixed_image.id,
                                                        self.moving_image.id,
                                                        registration_type)
    
            # log the registration task
            self.registration_task = self.db.add_registration_task(self.fixed_image.id,
                                                                    self.moving_image.id,
                                                                    registration_type)
            
    def log_registration(self,
                         registration: Registration) -> None:
        """
        Log a registration in the database. Takes the registration object as input.
        """
        self.log_registration_task(registration.fixed_image.file_name,
                                    registration.fixed_image.channel,
                                    registration.moving_image.file_name,
                                    registration.moving_image.channel,
                                    registration.registration_type)
        
    def log_transformations(self,
                            forward_transformations: list[str],
                            inverse_transformations: list[str]) -> None:
        """
        Log the transformations generated by a registration in the database 
        and add them to the registration task.
        """
        # check that the registration task has been logged
        if self.registration_task is None:
            raise Exception("Registration task must be logged before transformations.")
        
        # create and add forward transformations to RegistrationTask
        for i_transform, forward_transformation_file in enumerate(forward_transformations):
            forward_transformation = self.db.add_forward_transformation(
                file_name = forward_transformation_file,
                order = i_transform,
                registration_task_id = self.registration_task.id)

        # create and add inverse transformations to RegistrationTask
        for i_tranform, inverse_transformation_file in enumerate(inverse_transformations):
            inverse_transformation = self.db.add_inverse_transformation(
                file_name = inverse_transformation_file,
                order = i_tranform,
                registration_task_id = self.registration_task.id)
            
class ImageTransformationTaskAlreadyInDatabase(Exception):
    """Transformation task is already in the database."""
    def __init__(self, 
                 image_id: int, 
                 registration_task_id: int, 
                 direction: str):
        self.msg = f"The transformation task for image_id {image_id} with registration_task_id {registration_task_id} and direction {direction} is already in the database." 
    def __str__(self):
        return self.msg
    
class ImageTransformationLogger:

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager
        # generated during log_registration_task:
        self.transformation_task = None
        self.image = None
        self.reference_image = None
        self.registration_task = None

    def log_transformation_task(self,
                                image_filename: str | Path,
                                image_channel: int,
                                interpolation: str,
                                registration_fixed_filename: str | Path,
                                registration_fixed_image_channel: int,
                                registration_moving_filename: str | Path,
                                registration_moving_image_channel: int,
                                registration_type: str,
                                direction: str) -> None:
        """
        Log a image transformation task in the database.

        Args:
            image_filename: the filename of the image to transform
            image_channel: the channel of the image to transform
            interpolation: the interpolation method to use for the transformation
            registration_fixed_filename: the filename of the fixed image used for registration
            registration_fixed_image_channel: the channel of the fixed image used for registration
            registration_moving_filename: the filename of the moving image used for registration
            registration_moving_image_channel: the channel of the moving image used for registration
            registration_type: the type of registration used to generate the transformation
            direction: the direction of the transformation, must be one of "forward" or "inverse"
        """

        # get image
        self.image = self.db.get_image(image_channel, image_filename)
        if self.image is None:
            raise Exception(f"Image {image_filename} with channel {image_channel} not found in database.")
        
        # get reference image
        self.reference_image = self.db.get_image(registration_fixed_image_channel, registration_fixed_filename)
        if self.reference_image is None:
            raise Exception(f"Image {registration_fixed_filename} with channel {registration_fixed_image_channel} not found in database.")
        
        # get moving image
        registration_moving_image = self.db.get_image(registration_moving_image_channel, registration_moving_filename)
        
        # get registration task
        self.registration_task = self.db.get_registration_task(fixed_image_id = self.reference_image.id,
                                                        moving_image_id = registration_moving_image.id,
                                                        registration_type = registration_type)
        if self.registration_task is None:
            raise Exception(f"Registration task with fixed image {registration_fixed_filename} and moving image {registration_moving_filename} not found in database.")
        
        # check that the transformation task doesn't already exist in the database
        if self.db.get_image_transformation_task(image_id = self.image.id,
                                                 interpolation = interpolation,
                                                 registration_id = self.registration_task.id,
                                                 direction = direction) is not None:
            raise ImageTransformationTaskAlreadyInDatabase(self.image.id, 
                                                           self.registration_task.id,
                                                            direction)

        # log the transformation task
        self.transformation_task = self.db.add_image_transformation_task(image_id = self.image.id,
                                                                         interpolation= interpolation,
                                                                         registration_id = self.registration_task.id,
                                                                         direction = direction)

    def log_transformation(self, registration: Registration, transformation: ImageTransformation) -> None:
        """
        Log an image transformation in the database. Takes the registration and transformation objects as input.
        """
        self.log_transformation_task(image_filename = transformation.image.file_name,
                                     image_channel = transformation.image.channel,
                                     interpolation = transformation.interpolation,
                                     registration_fixed_filename = registration.fixed_image.file_name,
                                     registration_fixed_image_channel = registration.fixed_image.channel,
                                     registration_moving_filename = registration.moving_image.file_name,
                                     registration_moving_image_channel = registration.moving_image.channel,
                                     registration_type = registration.registration_type,
                                     direction = transformation.direction)
        
    def _construct_transformed_image_filename(self) -> str:
        """
        Construct the filename of the transformed image.
        """

        # construct the filename
        transformed_image_filename = f"{Path(self.image.file_name).stem}_{self.registration_task.registration_type}_{self.transformation_task.interpolation}_tt{self.transformation_task.id}.tif"
        transformed_image_filename = Path(self.image.file_name).parent / transformed_image_filename
        return transformed_image_filename
        
    def log_transformed_image(self, transformed_image: npt.NDArray) -> str:
        """
        Save the transformed image to disk and log it in the database.
        
        Args: 
            transformed_image: the transformed image as a numpy array
        """
        if self.transformation_task is None:
            raise Exception("Transformation task must be logged before transformed image.")
        
        # save the transformed image
        transformed_image_filename = self._construct_transformed_image_filename()
        channel = 0
        save_image_as_uint16(transformed_image, transformed_image_filename)

        # add the transformed image to the database
        transformed_image = self.db.add_transformed_image(original_image_id = self.image.id,
                                                          reference_image_id = self.reference_image.id,
                                                          transformed_image_file_name = transformed_image_filename,
                                                          transformed_image_channel = channel)
        
        # update the transformation task
        self.db.update_image_transformation_task(self.transformation_task.id,
                                                    transformed_image.id)
        
        return transformed_image_filename
                                


