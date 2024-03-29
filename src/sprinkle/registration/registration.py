from __future__ import annotations

from sprinkle.registration.ants_utils import ants_registration
from dataclasses import dataclass
from sprinkle.image_utils import Image

@dataclass
class Registration:
    fixed_image: Image
    moving_image: Image
    registration_type: str

    @classmethod
    def from_dict(cls, d: dict) -> Registration:
        """
        Create a registration from a dictionary.
        """
        d = d.copy()
        d['fixed_image'] = Image(**d['fixed_image'])
        d['moving_image'] = Image(**d['moving_image'])
        if 'transformation_folder' in d:
            del d['transformation_folder']

        return cls(**d)

    def run(self, transformation_folder: str | None = None)-> tuple[str, str] :
        """
        A wrapper for ants_registration.

        Args:
            transformation_folder: The folder to save the transformation files in. If None, the files are saved as temporary files.

        Returns:
            The paths to the forward and inverse transforms saved in the transformation folder or as temporary files if no transformation folder is given.
        """
        fwdtransforms, invtransforms = ants_registration(fixed_image = self.fixed_image.load(),
                                                        fixed_image_spacing = self.fixed_image.resolution_xyz,
                                                        moving_image = self.moving_image.load(),
                                                        moving_image_spacing = self.moving_image.resolution_xyz,
                                                        registration_type = self.registration_type,
                                                        outprefix = transformation_folder,
                                                        verbose = True)
        return fwdtransforms, invtransforms
