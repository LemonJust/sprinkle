from __future__ import annotations
import ants 
import pandas as pd
import numpy as np
import numpy.typing as npt


# # from mpl_toolkits.mplot3d import Axes3D
# from image_utils import save_image_as_uint16

def read_image(image_path):
    return ants.image_read(image_path)

def save_image(image, file_path):
    ants.image_write(image, file_path)

def ants_registration(fixed_image: npt.NDArray , 
                      fixed_image_spacing: dict,
                      moving_image: npt.NDArray, 
                      moving_image_spacing: dict,
                      registration_type: str ='Rigid',
                      outprefix: str | None = None,
                      save_transformed_image: bool | str = True,
                      verbose: bool = True) -> tuple[str, str]:
    """
    Wrapper function for ANTs registration.

    Args:
        fixed_image: The fixed image.
        fixed_image_spacing: The spacing of the fixed image in microns. Dictionary with keys 'x', 'y', 'z'.
            It will be converted to ants image spacing (x, y, z).
        moving_image: The moving image.
        moving_image_spacing: The spacing of the moving image in microns. Dictionary with keys 'x', 'y', 'z'.
            It will be converted to ants image spacing (x, y, z).
        registration_type: The type of registration to perform, must be one of the types supported by ANTs.
        outprefix: The prefix to use for the output files.
        save_transformed_image: Whether to save the transformed image.
        verbose: Whether to print image information to the console.

    Returns:
        The paths to the forward and inverse transforms saved as temporary files.
    """
    fixed_image_spacing = [fixed_image_spacing[key] for key in 'xyz']
    moving_image_spacing = [moving_image_spacing[key] for key in 'xyz']

    # turn numpy array into ants image
    # need to flip the order of the spacing and the image axes order to get the correct orientation
    img_axes_order = [2, 1, 0]

    fixed_image = ants.from_numpy(
         np.transpose(fixed_image.astype(np.float32), # need to convert to float32 for ants
                      axes=img_axes_order), 
                      spacing=fixed_image_spacing) # need tuple in order xyz or list for ants
    moving_image = ants.from_numpy(
         np.transpose(moving_image.astype(np.float32), # need to convert to float32 for ants
                      axes=img_axes_order), 
                      spacing=moving_image_spacing) # need tuple in order xyz or list for ants

    if verbose:
        print("Fixed image:")
        print(fixed_image)
        print("Moving image:")
        print(moving_image)
    
    # run the registration and save transform to file
    rr = ants.registration(fixed=fixed_image,
                      moving=moving_image,
                      type_of_transform=registration_type,
                      verbose  = verbose,
                      outprefix = outprefix
                      )
    
    if save_transformed_image:
        warped_file_name = f"transformed_image_{np.random.randint(0, 1000)}.nii.gz"
        save_image(rr['warpedmovout'], warped_file_name)
        print(f"Saved transformed image to {warped_file_name}")

    return rr['fwdtransforms'], rr['invtransforms']

def transform_image(image: npt.NDArray,
                    resolution_xyz: npt.NDArray | list[float] | tuple[float, float, float], 
                    reference_image: npt.NDArray,
                    reference_image_resolution_xyz: npt.NDArray | list[float] | tuple[float, float, float],
                    transformlist: list[str], 
                    interpolation: str = 'linear', # 'linear', 'nearestNeighbor', 'multiLabel'
                    verbose: bool = True) -> npt.NDArray:
                    
        image = np.transpose(image, axes=[2,1,0])
        image = ants.from_numpy(image.astype(np.float32),
                                spacing=resolution_xyz)

        reference_image = np.transpose(reference_image, axes=[2,1,0])
        reference_image = ants.from_numpy(reference_image.astype(np.float32),
                                          spacing=reference_image_resolution_xyz)

        transformed_image = ants.apply_transforms(fixed=reference_image,
                                                  moving=image,
                                                  transformlist=transformlist,
                                                  interpolator=interpolation,
                                                  verbose=verbose)   
        transformed_image = transformed_image.numpy().transpose((2, 1, 0))

        return transformed_image

def transform_points(points: dict,
                     resolution_zyx: npt.NDArray | list[float] | tuple[float, float, float],
                     transformlist: list[str], 
                     whichtoinvert: list[bool] | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Transform a list of points using the given transform list.
    It is a bit complecated. 
    Read this: https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians

    Args:
        points: The points to transform, a dict wiht keys 'x', 'y', 'z'.
            Description from ants documentation: 
                data frame
                moving point set with n-points in rows of at least dim
                columns - we maintain extra information in additional
                columns. this should be a data frame with columns names x, y, z, t.

        resolution_zyx: The resolution of the points in microns (z, y, x).
        transformlist: The transform list.
            Description from ants documentation: 
                list of strings
                list of transforms generated by ants.registration where each transform is a filename.
        whichtoinvert: Which transforms to invert.
            Description from ants documentation:
                list of booleans (optional)
                Must be same length as transformlist.
                whichtoinvert[i] is True if transformlist[i] is a matrix,
                and the matrix should be inverted. If transformlist[i] is a
                warp field, whichtoinvert[i] must be False.
                If the transform list is a matrix followed by a warp field,
                whichtoinvert defaults to (True,False). Otherwise it defaults
                to [False]*len(transformlist)).

        verbose: Whether to print image information to the console.

    Returns:
        The transformed points.
    """
    # convert to pandas dataframe
    points = pd.DataFrame.from_dict(points)
    # convert to physical space
    resolution = {key:value for key, value in zip('zyx',resolution_zyx)}
    points[['z','y','x']] = points[['z','y','x']].multiply(resolution)

    # apply the transform
    n_dim = 3
    transformed_points = ants.apply_transforms_to_points(n_dim, 
                                                         points, 
                                                         transformlist,
                                                         whichtoinvert = whichtoinvert, 
                                                         verbose=verbose)
    return transformed_points


