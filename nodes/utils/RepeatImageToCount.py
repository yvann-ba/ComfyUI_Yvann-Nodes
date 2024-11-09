from ... import Yvann

class UtilsNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"

class RepeatImageToCount(UtilsNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "count": ("INT",),
            }
        }
    RETURN_TYPES = ("IMAGE")
    RETURN_NAMES = ("IMAGE")
    FUNCTION = "repeat_image_to_count"
    
    def repeat_image_to_count(self, image, count) :
        