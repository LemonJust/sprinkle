"""
DB manager for the database.
"""
from __future__ import annotations

from pathlib import Path
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sprinkle.db.fields import Stages, Labels, ImageSpaces, SegmentationModelTypes, SegmentationTypes
from sprinkle.db.models import Base

from sprinkle.db.models import ImageSpace, ImageType, Image, ClassifiedImage
from sprinkle.db.models import SegmentationParameters, SegmentationModel, SegmentationTask
from sprinkle.db.models import Centroids, ClassifiedCentroids
from sprinkle.db.models import ClassificationTask, ProbThrClassificationTask, IntensityThrClassificationTask
from sprinkle.db.models import RegistrationTask, ForwardTransformations, InverseTransformations
from sprinkle.db.models import ImageTransformationTask

from sqlalchemy.engine import Engine
from sqlalchemy import event


# enable foreign key support for sqlite
# more info:
# https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#foreign-key-support
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

class DatabaseManager:
    """
    Database manager for sqlalchemy database.
    """

    def __init__(self, db_path: str | Path | None = None, echo = True):
        """
        Args:
            db_path (str | Path): path to the database file.
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)

        if db_path is None:
            db_path = None
            # create in-memory database
            self.engine = create_engine("sqlite://", echo = echo)
        else:
            self.db_path = db_path
            # connect to existing or create database file if it doesn't exist
            self.engine = create_engine(f"sqlite:///{self.db_path}", echo = echo)

        # create tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.session = self.open_session()

    def open_session(self):
        """
        Open a session to the database.
        """
        return sessionmaker(bind=self.engine)()
    
    def close_session(self):
        """
        Close the session to the database.
        """
        self.session.close()

    # IMAGE-RELATED METHODS ____________________________________________________________

    def _get_or_create_image_type(self, 
                                  stage: str | None = None, 
                                  label: str | None = None, 
                                  transformed: bool | None = None,
                                  id: int | None = None) -> ImageType:
        """
        Get an image type from the database, or create it if it doesn't exist.

        Args:
            stage: the name of the image type: "raw", "segmented", "classified".
            label: the label of the image type: "gfp", "dapi", "huc", "labels", "probabilities".
            transformed: boolean indicating whether the image type is transformed or not.
            id: id of the image type. If provided, stage, label and transformed are ignored. When id is None, stage, label and transformed must be provided.

        Returns:
            ImageType: the image type object from the database.
        """
        if id is not None:
            image_type = self.session.query(ImageType).filter_by(id=id).one_or_none()
            if image_type is None:
                raise Exception(f"Image space with id {id} not found in database.")
        else:
            assert stage is not None and label is not None and transformed is not None, "When id is None, stage, label and transformed must be provided."

            stage = Stages(stage).value
            label = Labels(label).value

            image_type = self.session.query(ImageType).filter_by(stage=stage, label=label, 
                                                                    transformed=transformed).one_or_none()
            if image_type is None:
                image_type = ImageType(stage=stage, 
                                        label=label, 
                                        transformed=transformed)
                try:
                    self.session.add(image_type)
                    self.session.commit()
                except:
                    self.session.rollback()
                    raise

        return image_type

    def _get_or_create_image_space(self, 
                                   name: str | None = None, 
                                   resolution_xyz: list[float, float, float] | None = None, 
                                   id: int | None = None) -> ImageSpace:
        """
        Get an image space from the database, or create it if it doesn't exist.

        Args:
            name: image space: "live1", "live2", "fixed".
            resolution_xyz: resolution of the image space in XYZ order. 
                If None, the image space must already exist in the database and will be identified by its name.
            id: id of the image space. If provided, name and resolution_xyz are ignored. When id is None, name must be provided.

        Returns:
            ImageSpace: the image space object from the database.
        """

        if id is not None:
            image_space = self.session.query(ImageSpace).filter_by(id=id).one_or_none()
            if image_space is None:
                raise Exception(f"Image space with id {id} not found in database.")
            return image_space
        
        assert name is not None, "When id is None, name must be provided."
        
        name = ImageSpaces(name).value
        
        image_space = self.session.query(ImageSpace).filter_by(name=name).one_or_none()

        if image_space is None:
            assert resolution_xyz is not None, f"Image space {name} not found in database. Please provide resolution to create it: resolution_xyz = [x,y,z] can not be None."
            image_space = ImageSpace(name=name, 
                                        resolution_x=resolution_xyz[0],
                                        resolution_y=resolution_xyz[1],
                                        resolution_z=resolution_xyz[2])
            try:
                self.session.add(image_space)
                self.session.commit()
            except:
                self.session.rollback()
                raise
        return image_space

    def add_image(self, 
                  channel: int, file_name: str | Path,
                  stage: str, label: str, transformed: bool, 
                  image_space_name: str,
                  resolution_xyz: list[float, float, float] | None = None) -> Image:
        """
        Add an image to the database.

        Args:
            channel: channel of the image in the image file.
            file_name: path to the image file.
            stage: the name of the image type: "raw", "segmented", "class".
            label: the label of the image type: "gfp", "dapi", "huc", "labels", "probabilities".
            transformed: boolean indicating whether the image type is transformed or not.
            image_space_name: image space name: "live1", "live2", "fixed"
            resolution_xyz: resolution of the image space in XYZ order. 
                If None, the image space must already exist in the database and will be identified by its name.

        Returns:
            the image object added to the database.
        """
        if isinstance(file_name, Path):
            file_name = file_name.as_posix()

        image_type = self._get_or_create_image_type(stage,label,transformed)
        image_space = self._get_or_create_image_space(image_space_name, resolution_xyz)

        image = Image(channel=channel, 
            file_name=file_name, 
            image_type_id=image_type.id, 
            image_space_id=image_space.id)
        try:
            self.session.add(image)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return image
    
    def add_transformed_image(self,
                              original_image_id: int,
                              reference_image_id: int,
                              transformed_image_file_name: str | Path,
                              transformed_image_channel: int) -> Image:
        """
        Add a transformed image to the Image table. 
        Will get the information about the image type and image space from the original and reference images.
        """
        if isinstance(transformed_image_file_name, Path):
            transformed_image_file_name = transformed_image_file_name.as_posix()

        original_image = self.get_image(id = original_image_id)
        reference_image = self.get_image(id = reference_image_id)

        original_image_type = self._get_or_create_image_type(id = original_image.image_type_id)
        transformed_image_type = self._get_or_create_image_type(stage = original_image_type.stage,
                                                                label = original_image_type.label,
                                                                transformed = True)
        reference_image_space = self._get_or_create_image_space(id = reference_image.image_space_id)

        image = Image(channel = transformed_image_channel,
                      file_name = transformed_image_file_name,
                      image_type_id = transformed_image_type.id,
                      image_space_id = reference_image_space.id)
        try:
            self.session.add(image)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return image
    
    def get_image(self,
                    channel: int | None = None,
                    file_name: str | None = None,
                    stage: str | None = None,
                    label: str | None = None,
                    transformed: bool | None = None,
                    image_space: str | None = None,
                    id: int | None = None) -> Image:
        """
        Get an image from the database.

        Args:
            channel: channel of the image in the image file.
            file_name: path to the image file.
            stage: the name of the image type: "raw", "segmented", "class".
            label: the label of the image type: "gfp", "dapi", "huc", "labels", "probabilities".
            transformed: boolean indicating whether the image type is transformed or not.

        Returns:
            the image object from the database.
        """
        stage = Stages(stage).value if stage is not None else None
        label = Labels(label).value if label is not None else None
        image_space = ImageSpaces(image_space).value if image_space is not None else None

        image_type = None
        if stage is not None or label is not None or transformed is not None:
            image_type = self._get_or_create_image_type(stage,label,transformed)
            
        if image_space is not None:
            image_space = self._get_or_create_image_space(image_space)

        query = self.session.query(Image)
        if id is not None:
            query = query.filter_by(id=id)
        if channel is not None:
            query = query.filter_by(channel=channel)
        if file_name is not None:
            query = query.filter_by(file_name=file_name)
        if image_type is not None:
            query = query.filter_by(image_type_id=image_type.id)
        if image_space is not None:
            query = query.filter_by(image_space_id=image_space.id)
        
        # get image
        image = query.one_or_none()
        return image
    
    def get_images(self):
        """
        Get all images from the database.
        """
        return self.session.query(Image).all()

    def add_classified_image(self, 
                             image_id: int, 
                             class_name: str, 
                             classification_task_id: int) -> ClassifiedImage:
        """
        Add a classified image to the database. 
        Links an image to a class and a classification task.
        One classification task can have multiple classified images (one per class).

        Args:
            image_id: id of the image.
            class_name: name of the class.
            classification_task_id: id of the classification task.

        Returns:
            the classified image object added to the database.
        """
        classified_image = ClassifiedImage(image_id=image_id, 
                                            class_name=class_name, 
                                            classification_task_id=classification_task_id)
        try:
            self.session.add(classified_image)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return classified_image
    
    # SEGMENTATION-RELATED METHODS ____________________________________________________________

    def _get_or_create_segmentation_parameters(self,
                                               n_tiles: tuple[int, int, int],
                                               prob_thr: float,
                                               scale_probability: float,
                                               id: int | None = None) -> SegmentationParameters:
        """
        Get a segmentation parameters from the database, or create it if it doesn't exist.

        Args:
            n_tiles: number of tiles in XYZ order.
            prob_thr: probability threshold for the prediction.
            scale_probability: scale to use for saving the probability image.
            id: id of the segmentation parameters. If provided, parameters are ignored. When id is None, parameters must be provided.

        Returns:
            SegmentationParameters: the segmentation parameters object from the database.
        """
        if id is not None:
            segmentation_parameters = self.session.query(SegmentationParameters).filter_by(id=id).one_or_none()
            if segmentation_parameters is None:
                raise Exception(f"Segmentation parameters with id {id} not found in database.")
            return segmentation_parameters
        
        assert n_tiles is not None and prob_thr is not None and scale_probability is not None, "When id is None, n_tiles, prob_thr and scale_probability must be provided."

        segmentation_parameters = self.session.query(
            SegmentationParameters).filter_by(n_tiles_x=n_tiles[2],
                                              n_tiles_y=n_tiles[1],
                                              n_tiles_z=n_tiles[0],
                                              prob_thr=prob_thr,
                                              scale_probability=scale_probability).one_or_none()
        if segmentation_parameters is None:
            segmentation_parameters = SegmentationParameters(n_tiles_x=n_tiles[2],
                                                             n_tiles_y=n_tiles[1],
                                                             n_tiles_z=n_tiles[0],
                                                             prob_thr=prob_thr,
                                                             scale_probability=scale_probability)
            try:
                self.session.add(segmentation_parameters)
                self.session.commit()
            except:
                self.session.rollback()
                raise
            
        return segmentation_parameters
    
    def add_segmentation_model(self, 
                               folder: str | Path, 
                               model_type: str, 
                               segmentation_type: str,
                               segmentation_training_id: int | None) -> SegmentationModel:
        """
        Add a segmentation model to the database.

        Args:
            folder: path to the folder of the segmentation model.
            model_type: type of the segmentation model: "unet", "unet2d", "unet3d", "unet2d3d".
            segmentation_type: type of the segmentation: "synapse", "nuclei".
            segmentation_training_id: id of the segmentation training used to train the model.
        Returns:
            the segmentation model object added to the database.
        """

        #TODO : add segmentation_training info to the database

        if isinstance(folder, Path):
            folder = folder.as_posix()

        model_type = SegmentationModelTypes(model_type).value
        segmentation_type = SegmentationTypes(segmentation_type).value

        segmentation_model = SegmentationModel(folder=folder,
                                               model_type=model_type,
                                               segmentation_type=segmentation_type,
                                               segmentation_training_id=segmentation_training_id)
        try:
            self.session.add(segmentation_model)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return segmentation_model
    
    def get_segmentation_model(self,
                               folder: str | None = None,
                               model_type: str | None = None,
                               segmentation_type: str | None = None,
                               segmentation_training_id: int | None = None) -> SegmentationModel:
        """
        Get a segmentation model from the database.

        Args:
            folder: path to the folder of the segmentation model.
            model_type: type of the segmentation model: "unet", "unet2d", "unet3d", "unet2d3d".
            segmentation_type: type of the segmentation: "synapse1", "synapse2", "nuclei".
            segmentation_training_id: id of the segmentation training used to train the model.
        Returns:
            the segmentation model object from the database.
        """

        model_type = SegmentationModelTypes(model_type).value if model_type is not None else None
        segmentation_type = SegmentationTypes(segmentation_type).value if segmentation_type is not None else None

        query = self.session.query(SegmentationModel)
        if folder is not None:
            query = query.filter_by(folder=folder)
        if model_type is not None:
            query = query.filter_by(model_type=model_type)
        if segmentation_type is not None:
            query = query.filter_by(segmentation_type=segmentation_type)
        if segmentation_training_id is not None:
            query = query.filter_by(segmentation_training_id=segmentation_training_id)

        # get segmentation model
        segmentation_model = query.one_or_none()
        return segmentation_model
    
    def get_segmentation_models(self):
        """
        Get all segmentation models from the database.
        """
        return self.session.query(SegmentationModel).all()

    def add_segmentation_task(self,
                                image_id: int,
                                labels_image_id: int | None,
                                probabilities_image_id: int | None,
                                segmentation_model_id: int,
                                n_tiles: tuple[int,int,int],
                                prob_thr: float,
                                scale_probability: float) -> SegmentationTask:
        """
        Adds a segmentation task to the database.

        Args:
            image_id: id of the image to segment.
            labels_image_id: id of the labels image (produced by the segmentation model).
            probabilities_image_id: id of the probabilities image (produced by the segmentation model).
            segmentation_model_id: id of the segmentation model.
            n_tiles: number of tiles to use for prediction in ZYX order.
            prob_thr: probability threshold for the prediction.
            scale_probability: scale to use for saving the probability image.

        Returns:
            the segmentation task object added to the database.
        """

        parameters = self._get_or_create_segmentation_parameters(n_tiles,prob_thr,scale_probability)
        
        segmentation_task = SegmentationTask(image_id=image_id,
                                             labels_image_id=labels_image_id,
                                             probabilities_image_id=probabilities_image_id,
                                             segmentation_model_id=segmentation_model_id,
                                             parameters_id=parameters.id)
        try:
            self.session.add(segmentation_task)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return segmentation_task

    def update_segmentation_task(self,
                                segmentation_task_id: int,
                                labels_image_id: int | None = None,
                                probabilities_image_id: int | None = None) -> SegmentationTask:
        """
        Add a labels image and/or a probabilities image to a segmentation task.
        """
        
        segmentation_task = self.session.query(SegmentationTask).filter_by(id=segmentation_task_id).one_or_none()
        if segmentation_task is None:
            raise Exception(f"Segmentation task with id {segmentation_task_id} not found in database.")
        
        if labels_image_id is not None:
            # check that labels image exists
            labels_image = self.session.query(Image).filter_by(id=labels_image_id).one_or_none()
            if labels_image is None:
                raise Exception(f"Image with id {labels_image_id} not found in database.")
            
        if probabilities_image_id is not None:
            # check that probabilities image exists
            probab_image = self.session.query(Image).filter_by(id=probabilities_image_id).one_or_none()
            if probab_image is None:
                raise Exception(f"Image with id {probabilities_image_id} not found in database.")
            
        try:
            if labels_image_id is not None:
                segmentation_task.labels_image_id = labels_image_id
            if probabilities_image_id is not None:
                segmentation_task.probabilities_image_id = probabilities_image_id
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return segmentation_task

    def get_segmentation_task(self,
                                image_id: int | None = None,
                                labels_image_id: int | None = None,
                                probabilities_image_id: int | None = None,
                                segmentation_model_id: int | None = None,
                                n_tiles: tuple[int,int,int] | None = None,
                                prob_thr: float | None = None,
                                scale_probability: float | None = None,
                                id: int | None = None) -> list[SegmentationTask]:
        """
        Get a segmentation task from the database.

        Args:
            image_id: id of the image to segment.
            labels_image_id: id of the labels image (produced by the segmentation model).
            probabilities_image_id: id of the probabilities image (produced by the segmentation model).
            segmentation_model_id: id of the segmentation model.
            n_tiles: number of tiles to use for prediction in ZYX order.
            prob_thr: probability threshold for the prediction.
            scale_probability: scale to use for saving the probability image.
            id: id of the segmentation task.
        Returns:
            the segmentation task object from the database.
        """
        parameters = None
        if n_tiles is not None and prob_thr is not None and scale_probability is not None:
            parameters = self._get_or_create_segmentation_parameters(n_tiles,prob_thr,scale_probability)

        query = self.session.query(SegmentationTask)
        if id is not None:
            query = query.filter_by(id=id)
        if image_id is not None:
            query = query.filter_by(image_id=image_id)
        if labels_image_id is not None:
            query = query.filter_by(labels_image_id=labels_image_id)
        if probabilities_image_id is not None:
            query = query.filter_by(probabilities_image_id=probabilities_image_id)
        if segmentation_model_id is not None:
            query = query.filter_by(segmentation_model_id=segmentation_model_id)
        if parameters is not None:
            query = query.filter_by(parameters_id=parameters.id)

        segmentation_tasks = query.one_or_none()
        return segmentation_tasks
    
    # CENTROIDS-RELATED METHODS ____________________________________________________________
    
    def add_centroids(self,
                      labels: list[int],
                      centroids_zyx: list[tuple[float,float,float]],
                      probabilities: list[float],
                      segmentation_task_id: int) -> Centroids:
        
        centroids = [Centroids(label = l, 
                               x=x,y=y,z=z,
                               probability=p, 
                               segmentation_task_id=segmentation_task_id) 
                               for l,(z,y,x),p in zip(labels, centroids_zyx,probabilities)]
        try:
            self.session.add_all(centroids)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return centroids
    
    def get_centroids(self, 
        segmentation_task_id: int | None = None,
        centroid_ids: [int] | None = None) -> list[Centroids]:
            
        if centroid_ids is not None:
            assert isinstance(centroid_ids, list), "centroid_ids must be a list of centroid ids."

            centroids = self.session.query(Centroids).filter(Centroids.id.in_(centroid_ids)).all()
        else:
            assert segmentation_task_id is not None, "segmentation_task_id must be provided if centroid_ids is not provided."
            assert isinstance(segmentation_task_id, int), "segmentation_task_id must be an integer."

            centroids = self.session.query(Centroids).filter_by(segmentation_task_id=segmentation_task_id).all()

        return centroids
    
    def add_classified_centroids(self,
                     centroid_ids: list[int],
                     class_names: list[str],
                     classification_task_id: int) -> list[ClassifiedCentroids]:
        
        classified_centroids = [ClassifiedCentroids(centroid_id=c_id, class_name = cn, classification_task_id=classification_task_id) for c_id, cn in zip(centroid_ids, class_names)]
        try:
            self.session.add_all(classified_centroids)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return classified_centroids

    def get_classified_centroids(self, classification_task_id: int) -> list[ClassifiedCentroids]:
                
            classified_centroids = self.session.query(ClassifiedCentroids).filter_by(classification_task_id=classification_task_id).all()
            return classified_centroids

    # CLASSIFICATION-RELATED METHODS ____________________________________________________________

    def _add_prob_thr_classification_task(self,
                                            classification_task_id: int,
                                            segmentation_task_id: int,
                                            class_name: str,
                                            prob_thr: float) -> ProbThrClassificationTask:
        prob_thr_classification_task = ProbThrClassificationTask(
                                                        classification_task_id=classification_task_id,
                                                        segmentation_task_id=segmentation_task_id,
                                                        class_name=class_name,
                                                        prob_thr=prob_thr)
        try :
            self.session.add(prob_thr_classification_task)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return prob_thr_classification_task

    def _add_intensity_thr_classification_task(self,
                                            classification_task_id: int,
                                            segmentation_task_id: int) -> IntensityThrClassificationTask:
        intensity_thr_classification_task = IntensityThrClassificationTask(
                                                        classification_task_id=classification_task_id,
                                                        segmentation_task_id=segmentation_task_id)
        try :
            self.session.add(intensity_thr_classification_task)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return intensity_thr_classification_task
    
    def add_classification_method_task(self,
                                        classification_method: str,
                                        classification_task_id: int,
                                        segmentation_task_id: int,
                                        parameters: dict) -> ProbThrClassificationTask | IntensityThrClassificationTask:
        if classification_method == "prob_thr":
            assert "prob_thr" in parameters, "Parameters for classification method 'prob_thr' must contain prob_thr."
            assert "class_name" in parameters, "Parameters for classification method 'prob_thr' must contain class_name."
            classification_method_task = self._add_prob_thr_classification_task(classification_task_id,
                                                                                segmentation_task_id,
                                                                                parameters["class_name"],
                                                                                parameters["prob_thr"])
        elif classification_method == "intensity_thr":
            classification_method_task = self._add_intensity_thr_classification_task(classification_task_id,
                                                                                segmentation_task_id)
        else: 
            raise Exception(f"Classification method {classification_method} not supported.")
        
        return classification_method_task
    
    def add_classification_task(self, 
                                classification_method: str,
                                segmentation_task_id: int,
                                parameters: dict) -> ClassificationTask:
        """
        Add a classification task to the database.

        Args:
            classification_method: the classification method: "prob_thr".
            segmentation_task_id: id of the segmentation task.
            parameters: parameters of the classification method.

        Returns:
            the classification task object added to the database.
        """

        classification_task = ClassificationTask(classification_method=classification_method)

        try :
            self.session.add(classification_task)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        # add method-specific classification task 
        self.add_classification_method_task(classification_method,
                                            classification_task.id,
                                            segmentation_task_id,
                                            parameters)
        
        return classification_task
    
    def _get_prob_thr_classification_task(self,
                                            classification_task_id: int | None = None,
                                            segmentation_task_id: int | None = None,
                                            class_name: str | None = None,
                                            prob_thr: float | None = None,
                                            id: int | None = None) -> ProbThrClassificationTask:
        query = self.session.query(ProbThrClassificationTask)
        if classification_task_id is not None:
            query = query.filter_by(classification_task_id=classification_task_id)
        if segmentation_task_id is not None:
            query = query.filter_by(segmentation_task_id=segmentation_task_id)
        if class_name is not None:
            query = query.filter_by(class_name=class_name)
        if prob_thr is not None:
            query = query.filter_by(prob_thr=prob_thr)
        if id is not None:
            query = query.filter_by(id=id)

        classification_method_task = query.one_or_none()

        return classification_method_task
    
    def _get_intensity_thr_classification_task(self,
                                            classification_task_id: int | None = None,
                                            segmentation_task_id: int | None = None,
                                            id: int | None = None) -> IntensityThrClassificationTask:
        query = self.session.query(IntensityThrClassificationTask)
        if classification_task_id is not None:
            query = query.filter_by(classification_task_id=classification_task_id)
        if segmentation_task_id is not None:
            query = query.filter_by(segmentation_task_id=segmentation_task_id)
        if id is not None:
            query = query.filter_by(id=id)

        classification_method_task = query.one_or_none()

        return classification_method_task
    
    def get_classification_method_task(self,
                                        classification_method: str | None = None,
                                        classification_task_id: int | None = None,
                                        segmentation_task_id: int | None = None,
                                        parameters: dict | None = None,
                                        id: int | None = None) -> ProbThrClassificationTask | IntensityThrClassificationTask:
        """
        For now we only have two classification method: "prob_thr" and "intensity_thr".
        """
        
        if classification_method == "prob_thr":
            if parameters is not None:
                assert parameters is not None and "prob_thr" in parameters, "Parameters for classification method 'prob_thr' must contain prob_thr."
                assert parameters is not None and "class_name" in parameters, "Parameters for classification method 'prob_thr' must contain class_name."
            else:
                parameters = {"prob_thr": None, "class_name": None}

            classification_method_task = self._get_prob_thr_classification_task(classification_task_id,
                                                                                segmentation_task_id, 
                                                                                parameters["class_name"],
                                                                                parameters["prob_thr"], 
                                                                                id)
        elif classification_method == "intensity_thr":
            # TODO: add parameters to the IntensityThrClassificationTask
            classification_method_task = self._get_intensity_thr_classification_task(classification_task_id,
                                                                                segmentation_task_id, 
                                                                                id)
        else:
            raise Exception(f"Classification method {classification_method} not supported.")
        
        return classification_method_task

    def get_classification_task(self,
                                classification_method: str | None = None,
                                segmentation_task_id: int | None = None,
                                parameters: dict | None = None,
                                id: int | None = None,
                                method_task_id: int | None = None) -> ClassificationTask:
        """
        Get a classification task from the database.

        Args:
            classification_method: the classification method: "prob_thr".
            segmentation_task_id: id of the segmentation task.
            parameters: parameters of the classification method.
            id: id of the classification task.
            method_task_id: id of the method-specific classification task.

        Returns:
            the classification task object from the database.
        """
        classification_task = None
        # when classification method specifics are provided, 
        # query the method-specific classification task
        if classification_method is not None and (
            segmentation_task_id is not None or parameters is not None or method_task_id is not None):
                classification_method_task = self.get_classification_method_task(classification_method,
                                                                                 id,
                                                                                 segmentation_task_id,
                                                                                 parameters,
                                                                                 method_task_id)
                # if such a method-specific classification task exists,
                # get the corresponding classification task
                # also check that the classification task id is the same as the one requrested
                if classification_method_task is not None and (
                    id is None or classification_method_task.classification_task_id == id):
                    classification_task = self.session.query(ClassificationTask).filter_by(id=classification_method_task.classification_task_id).one_or_none()

        elif classification_method is None and (
            segmentation_task_id is not None or parameters is not None or method_task_id is not None):
            raise Exception("When classification_method is None, segmentation_task_id, parameters and method_task_id must be None.")
        
        else:
            # get the classification task directly
            query = self.session.query(ClassificationTask)
            if id is not None:
                query = query.filter_by(id=id)
            if classification_method is not None:
                query = query.filter_by(classification_method=classification_method)

            classification_task = query.one_or_none()

        return classification_task

    def get_registration_task(self, 
                              fixed_image_id: int | None = None, 
                              moving_image_id: int | None = None, 
                              registration_type: str | None = None,
                              id: int | None = None):
        """
        Get a registration task from the database.
        Will rise an exception if more than one registration task is found.
        """
        registration_task = None
        query = self.session.query(RegistrationTask)
        if id is not None:
            query = query.filter_by(id=id)
        if fixed_image_id is not None:
            query = query.filter_by(fixed_image_id=fixed_image_id)
        if moving_image_id is not None:
            query = query.filter_by(moving_image_id=moving_image_id)
        if registration_type is not None:
            query = query.filter_by(registration_type=registration_type)
        registration_task = query.one_or_none()

        return registration_task
    
    def add_registration_task(self,
                              fixed_image_id: int, 
                              moving_image_id: int, 
                              registration_type: str):
            """
            Add a registration task to the database.
            """
            registration_task = RegistrationTask(fixed_image_id=fixed_image_id,
                                                 moving_image_id=moving_image_id,
                                                 registration_type=registration_type)
            try:
                self.session.add(registration_task)
                self.session.commit()
            except:
                self.session.rollback()
                raise
    
            return registration_task
    
    def add_forward_transformation(self, 
                                   file_name: str | Path,
                                   order : int,
                                   registration_task_id: int) -> ForwardTransformations:
        """
        Add a forward transformation to the database.
        """
        if isinstance(file_name, Path):
            file_name = file_name.as_posix()

        forward_transformation = ForwardTransformations(file_name=file_name,
                                                        order=order,
                                                        registration_task_id=registration_task_id)
        # get corresponding registration task
        registration_task = self.get_registration_task(id=registration_task_id)
        try:
            self.session.add(forward_transformation)
            registration_task.forward_transformations.append(forward_transformation)
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return forward_transformation

    def add_inverse_transformation(self,
                                      file_name: str | Path,
                                      order : int,
                                      registration_task_id: int) -> InverseTransformations:
        """
        Add an inverse transformation to the database.
        """
        if isinstance(file_name, Path):
            file_name = file_name.as_posix()
    
        inverse_transformation = InverseTransformations(file_name=file_name,
                                                          order=order,
                                                          registration_task_id=registration_task_id)
        # get corresponding registration task
        registration_task = self.get_registration_task(id=registration_task_id)
        try:
            self.session.add(inverse_transformation)
            registration_task.inverse_transformations.append(inverse_transformation)
            self.session.commit()
        except:
            self.session.rollback()
            raise
    
        return inverse_transformation
    
    def add_image_transformation_task(self, 
                                      image_id: int,
                                      interpolation: str,
                                      registration_id: int,
                                      direction: str) -> ImageTransformationTask:
        """
        Add an image transformation task to the database.
        """
        image_transformation_task = ImageTransformationTask(image_id=image_id,
                                                            interpolation=interpolation,
                                                            registration_id=registration_id,
                                                            direction=direction)
        # get corresponding registration task
        registration_task = self.get_registration_task(id=registration_id)
        # get corresponding image
        image = self.get_image(id=image_id)

        try:
            self.session.add(image_transformation_task)
            image_transformation_task.registration = registration_task
            image_transformation_task.image = image
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return image_transformation_task
    
    def get_image_transformation_task(self,
                                        image_id: int | None = None,
                                        interpolation: str | None = None,
                                        registration_id: int | None = None,
                                        direction: str | None = None,
                                        id: int | None = None) -> ImageTransformationTask:
            """
            Get an image transformation task from the database.
            """
            query = self.session.query(ImageTransformationTask)
            if id is not None:
                query = query.filter_by(id=id)
            if image_id is not None:
                query = query.filter_by(image_id=image_id)
            if interpolation is not None:
                query = query.filter_by(interpolation=interpolation)
            if registration_id is not None:
                query = query.filter_by(registration_id=registration_id)
            if direction is not None:
                query = query.filter_by(direction=direction)
    
            image_transformation_task = query.one_or_none()
    
            return image_transformation_task
    
    def update_image_transformation_task(self, 
                                         transformation_task_id: int,
                                         transformed_image_id: int) -> ImageTransformationTask:
        """
        Update an image transformation task by adding the information about the transfomed image.
        """
        transformation_task = self.get_image_transformation_task(id=transformation_task_id)
        if transformation_task is None:
            raise Exception(f"Image transformation task with id {transformation_task_id} not found in database.")
        
        # check that transformed image exists
        transformed_image = self.get_image(id=transformed_image_id)
        if transformed_image is None:
            raise Exception(f"Image with id {transformed_image_id} not found in database.")
        
        try:
            transformation_task.transformed_image_id = transformed_image_id
            transformation_task.transformed_image = transformed_image
            self.session.commit()
        except:
            self.session.rollback()
            raise

        return transformation_task













