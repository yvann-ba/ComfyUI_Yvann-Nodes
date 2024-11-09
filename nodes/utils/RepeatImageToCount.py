from ... import Yvann
import torch 
class UtilsNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"

class RepeatImageToCount(UtilsNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "count": ("INT", {"default": 1, "min": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "repeat_image_to_count"
    
    def repeat_image_to_count(self, image, count):
        num_images = image.size(0)  # Number of images in the input batch

        # Create indices to select images from input batch
        indices = [i % num_images for i in range(count)]  # Cycle through images to reach the desired count

        # Select images using the computed indices
        images = image[indices]
        return (images,)