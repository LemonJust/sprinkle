"""
Contains the SQLAlchemy models for the database: 
    - ImageSpace
    - ImageType
    - Image
    - SegmentationTraining
    - SegmentationModel
    - SegmentationTask
    - Centroids
    - ClassifiedCentroids
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Column, UniqueConstraint
from sqlalchemy import Integer, String, Boolean, Float
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship



class Base(DeclarativeBase):
    pass

# IMAGE-RELATED TABLES ________________________________________________________________

class ImageSpace(Base):
    """
    ImageSpace table. 
    Describes the space in which the images are: 
    images can be transformed to other spaces by running registration and/or applying transformations.

    Attributes:
        id: the id of the image space
        name: the name of the image space: "live1", "live2", "fixed".
    """
    __tablename__ = "image_space"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    resolution_x = Column(Float, nullable=False)
    resolution_y = Column(Float, nullable=False)
    resolution_z = Column(Float, nullable=False)

class ImageType(Base):
    """
    ImageType table. 
    Describes the type of the image, wether it was processed or not, and the content of the image.

    Attributes:
        id: the id of the image type
        stage: the name of the image type: "raw", "segmented", "class".
        label: the label of the image type: "gfp", "dapi", "huc", "labels", "probabilities".
        transformed: boolean indicating whether the image type is transformed or not.
    """
    __tablename__ = "image_type"

    id = Column(Integer, primary_key=True)
    stage = Column(String, nullable=False)
    label = Column(String, nullable=False)
    transformed = Column(Boolean, nullable=False)

class Image(Base):
    """
    Image table. All the images are stored in this table.

    Attributes:
        id: the id of the image
        channel: the channel of the image
        file_name: the file name of the image
        image_type_id: the id of the image type
        image_space_id: the id of the image space
    """
    __tablename__ = "image"

    id = Column(Integer, primary_key=True)
    channel = Column(Integer, nullable=False)
    file_name = Column(String, nullable=False)
    image_type_id = Column(Integer, ForeignKey("image_type.id"), nullable=False)
    image_space_id = Column(Integer, ForeignKey("image_space.id"), nullable=False)
    __table_args__ = (UniqueConstraint('channel', 'file_name'),)

class ClassifiedImage(Base):
    """
    ClassifiedImage table. 
    The classified images are stored in this table.

    Attributes:
        id: the id of the classified image
        image_id: the id of the image that has the classification result
        class_name: the name of the class to assign to the image
        classification_task_id: the id of the classification task
    """
    __tablename__ = "classified_image"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    class_name = Column(String, nullable=False)
    classification_task_id = Column(Integer, ForeignKey("classification_task.id"), nullable=False)
    __table_args__ = (UniqueConstraint('image_id', 'class_name', 'classification_task_id'),)

# SEGMENTATION-RELATED TABLES _________________________________________________________

class SegmentationTraining(Base):
    """
    SegmentationTraining table. 
    Information about the segmentation training runs are stored in this table.

    Attributes:
        id: the id of the segmentation training
        parameters: json string with the parameters of the segmentation training
    """
    __tablename__ = "segmentation_training"

    id = Column(Integer, primary_key=True)
    parameters = Column(String, nullable=False)

class SegmentationModel(Base):
    """
    SegmentationModel table. 
    The trained segmentation models are stored in this table.

    Attributes:
        id: the id of the segmentation model
        folder: the folder of the segmentation model
        model_type: the type of the segmentation model: "stardist3D" only for now.
        segmentation_type: the type of the segmentation task: "synapse", "nuclei".
        segmentation_training_id: the id of the segmentation training
    """
    __tablename__ = "segmentation_model"

    id = Column(Integer, primary_key=True)
    folder = Column(String, nullable=False, unique=True)
    model_type = Column(String, nullable=False)
    segmentation_type = Column(String, nullable=False)
    segmentation_training_id = Column(Integer, ForeignKey("segmentation_training.id"), nullable=True)

class SegmentationParameters(Base):
    """
    SegmentationParameters table.
    Information about the parameters of the segmentation run.

    Attributes:
        id: the id of the segmentation parameters
        n_tiles_z: the number of tiles to use for prediction
        n_tiles_y: the number of tiles to use for prediction
        n_tiles_x: the number of tiles to use for prediction
        prob_thr: the probability threshold for the prediction
        scale_probability: the scale to use for saving the probability image
    """
    __tablename__ = "segmentation_parameters"

    id = Column(Integer, primary_key=True)
    n_tiles_x = Column(Integer, nullable=False)
    n_tiles_y = Column(Integer, nullable=False)
    n_tiles_z = Column(Integer, nullable=False)
    prob_thr = Column(Float, nullable=False)
    scale_probability = Column(Integer, nullable=False)
    __table_args__ = (UniqueConstraint('n_tiles_z', 'n_tiles_y', 'n_tiles_x', 'prob_thr', 'scale_probability'),)

class SegmentationTask(Base):
    """
    SegmentationTask table.
    The segmentation tasks are stored in this table, 
    they have the information about the inputs and outputs of the segmentation run.

    Attributes:
        id: the id of the segmentation task
        image_id: the id of the image to segment
        labels_image_id: the id of the labels image (produced by the segmentation model)
        probabilities_image_id: the id of the probabilities image (produced by the segmentation model)
        segmentation_model_id: the id of the segmentation model
        parameters_id: the id of the segmentation parameters
    """
    __tablename__ = "segmentation_task"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    labels_image_id = Column(Integer, ForeignKey("image.id"), nullable=True)
    probabilities_image_id = Column(Integer, ForeignKey("image.id"), nullable=True)
    segmentation_model_id = Column(Integer, ForeignKey("segmentation_model.id"), nullable=False)
    parameters_id = Column(Integer, ForeignKey("segmentation_parameters.id"), nullable=False)
    __table_args__ = (UniqueConstraint('image_id', 'segmentation_model_id', 'parameters_id'),)

# CENTROIDS-RELATED TABLES ____________________________________________________________

class Centroids(Base):
    """
    Centroids table.

    Attributes:
        id: the id of the centroid
        label: the label of the centroid (0 for background), as shown in the labels image
        x: the x coordinate of the centroid in pixels
        y: the y coordinate of the centroid in pixels
        z: the z coordinate of the centroid in pixels
        probability: the probability of the centroid
        segmentation_task_id: the id of the segmentation task
    """
    __tablename__ = "centroids"

    id = Column(Integer, primary_key=True)
    label = Column(Integer, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    probability = Column(Float, nullable=False)
    segmentation_task_id = Column(Integer, ForeignKey("segmentation_task.id"), nullable=False)
    __table_args__ = (UniqueConstraint('x', 'y', 'z', 'segmentation_task_id'),)
    __table_args__ = (UniqueConstraint('label', 'segmentation_task_id'),)

class ClassifiedCentroids(Base):
    """
    ClassifiedCentroids table.

    Attributes:
        id: the id of the classified centroid
        centroid_id: the id of the candidate centroid
    """
    __tablename__ = "classified_centroids"

    id = Column(Integer, primary_key=True)
    centroid_id = Column(Integer, ForeignKey("centroids.id"), nullable=False)
    class_name = Column(String, nullable=False)
    classification_task_id = Column(Integer, ForeignKey("classification_task.id"), nullable=False)
    __table_args__ = (UniqueConstraint('centroid_id', 'classification_task_id'),)

# class TransformedCentroids(Base):
#     """
#     TransformedCentroids table.

#     Attributes:
#         id: the id of the transformed centroid
#         centroid_id: the id of the centroid
#         x: the x coordinate of the centroid in pixels
#         y: the y coordinate of the centroid in pixels
#         z: the z coordinate of the centroid in pixels
#         transformation_task_id: the id of the transformation task
#     """
#     __tablename__ = "transformed_centroids"

#     id = Column(Integer, primary_key=True)
#     centroid_id = Column(Integer, ForeignKey("centroids.id"), nullable=False)
#     transformation_task_id = Column(Integer

# CLASSIFICATION-RELATED TABLES ________________________________________________________

class ClassificationTask(Base):
    """
    ClassificationTask table.

    Attributes:
        id: the id of the classification task
        classification_method: the method used for classification: "prob_thr" only for now.
    """
    __tablename__ = "classification_task"

    id = Column(Integer, primary_key=True)
    classification_method = Column(String, nullable=False)

class ProbThrClassificationTask(Base):
    """
    Probability Thresholding ClassificationTask table. A subclass of ClassificationTask.

    Attributes:
        id: the id of the probability thresholding classification task
        classification_task_id: the id of the classification task
        segmentation_task_id: the id of the segmentation task that produced the centroids to classify
        class_name: the name of the class to assign to the centroids
        prob_thr: the probability threshold for the classification
    """
    __tablename__ = "prob_thr_classification_task"

    id = Column(Integer, primary_key=True)
    classification_task_id = Column(Integer, ForeignKey("classification_task.id"), nullable=False)
    segmentation_task_id = Column(Integer, ForeignKey("segmentation_task.id"), nullable=False)
    class_name = Column(String, nullable=False)
    prob_thr = Column(Float, nullable=False)
    __table_args__ = (UniqueConstraint('prob_thr', 'class_name','segmentation_task_id'),)

class IntensityThrClassificationTask(Base):
    """
    Intensity Thresholding ClassificationTask table. A subclass of ClassificationTask.

    Attributes:
        id: the id of the intensity thresholding classification task
        classification_task_id: the id of the classification task
        segmentation_task_id: the id of the segmentation task that produced the centroids to classify
    """
    __tablename__ = "intensity_thr_classification_task"

    id = Column(Integer, primary_key=True)
    classification_task_id = Column(Integer, ForeignKey("classification_task.id"), nullable=False)
    segmentation_task_id = Column(Integer, ForeignKey("segmentation_task.id"), nullable=False)
    # TODO: add intensity threshold table and link to it ? 
    # TODO: add a table that stores extracted intensity features per centroid? from FeatureExtractionTask?

# REGISTRATION AND TRANSFORMATION ________________________________________________________________

class RegistrationTask(Base): 
    """
    RegistrationTask table.
    The registration tasks are stored in this table, 
    they have the information about the inputs and outputs of the registration run.

    Attributes:
        id: the id of the registration task
        fixed_image_id: the id of the fixed image
        moving_image_id: the id of the moving image
        registration_type: the type of registration to perform,
            must be one of the types supported by ANTs
    """
    __tablename__ = "registration_task"

    id = Column(Integer, primary_key=True)
    fixed_image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    moving_image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    registration_type = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint('fixed_image_id', 'moving_image_id', 'registration_type'),)

    fixed_image = relationship("Image", foreign_keys=[fixed_image_id])
    moving_image = relationship("Image", foreign_keys=[moving_image_id])

    forward_transformations = relationship("ForwardTransformations", back_populates="registration_task")
    inverse_transformations = relationship("InverseTransformations", back_populates="registration_task")


class ForwardTransformations(Base):
    """
    ForwardTransformations table.

    Attributes:
        id: the id of the forward transformation
        file_name: the file name of the forward transformation
        order: the order of the transformation
        registration_task_id: the id of the registration task
    """
    __tablename__ = "forward_transformations"

    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    order = Column(Integer, nullable=False)
    registration_task_id = Column(Integer, ForeignKey("registration_task.id"), nullable=False)
    __table_args__ = (UniqueConstraint('order', 'registration_task_id'),)

    registration_task = relationship("RegistrationTask", back_populates="forward_transformations")

class InverseTransformations(Base):
    """
    InverseTransformations table.

    Attributes:
        id: the id of the inverse transformation
        file_name: the file name of the inverse transformation
        order: the order of the transformation
        registration_task_id: the id of the registration task
    """
    __tablename__ = "inverse_transformations"

    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    order = Column(Integer, nullable=False)
    registration_task_id = Column(Integer, ForeignKey("registration_task.id"), nullable=False)
    __table_args__ = (UniqueConstraint('order', 'registration_task_id'),)

    registration_task = relationship("RegistrationTask", back_populates="inverse_transformations")

class ImageTransformationTask(Base):
    """
    ImageTransformationTask table.
    The transformation tasks are stored in this table, 
    they have the information about the inputs and outputs of the transformation run.

    Attributes:
        id: the id of the transformation task
        image_id: the id of the image to transform
        transformed_image_id: the id of the transformed image
        interpolation: the interpolation to use when applying the transformation: "linear", "nearestNeighbor", "multiLabel"
        registration_id: the id of the registration task that produced the transformation
        direction: the direction of the transformation: "forward", "inverse"
    """
    __tablename__ = "image_transformation_task"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    transformed_image_id = Column(Integer, ForeignKey("image.id"), nullable=True)
    interpolation = Column(String, nullable=False)
    registration_id = Column(Integer, ForeignKey("registration_task.id"), nullable=False)
    direction = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint('image_id', 'interpolation', 'registration_id', 'direction'),)   

    registration = relationship("RegistrationTask", foreign_keys=[registration_id])
    image = relationship("Image", foreign_keys=[image_id])
    transformed_image = relationship("Image", foreign_keys=[transformed_image_id])

# class CentroidTransformationTask:
#     """
#     CentroidTransformationTask table.
#     The transformation tasks are stored in this table,
#     they have the information about the inputs and outputs of the transformation run.

#     Attributes:
#         id: the id of the transformation task
#         segmentation_task_id: the id of the segmentation task that produced the centroids to transform
#         registration_id: the id of the registration task that produced the transformation
#     """
#     __tablename__ = "centroid_transformation_task"

#     id = Column(Integer, primary_key=True)
#     segmentation_task_id = Column(Integer, ForeignKey("segmentation_task.id"), nullable=False)
#     registration_id = Column(Integer, ForeignKey("registration_task.id"), nullable=False)
#     __table_args__ = (UniqueConstraint('segmentation_task_id', 'transformation_id'),)

#     registration = relationship("RegistrationTask", foreign_keys=[registration_id])
#     segmentation_task = relationship("SegmentationTask", foreign_keys=[segmentation_task_id])
