import torch

from ..base import UtilsNodeBase


class RepeatImageToCount(UtilsNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "Batch of images to repeat"
                }),
                "count": ("INT", {
                    "default": 1, "min": 1,
                    "tooltip": "Number of output images (cycles if count > batch size)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "repeat_image_to_count"

    def repeat_image_to_count(self, image, count):
        num_images = image.size(0)  # Number of images in the input batch

        # Create indices to select images from input batch
        indices = [i % num_images for i in range(count)]  # Cycle through images

        # Select images using the computed indices
        images = image[indices]
        return (images,)