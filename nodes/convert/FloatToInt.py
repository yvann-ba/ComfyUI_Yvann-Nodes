from ... import Yvann
import numpy as np

class ConvertNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔄 Convert"

class FloatToInt(ConvertNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("INTS",)
    RETURN_NAMES = ("ints",)
    FUNCTION = "convert_floats_to_ints"

    def convert_floats_to_ints(self, floats):

        floats_array = np.array(floats)

        ints_array = np.round(floats_array)

        ints_array = ints_array.astype(int)
        integers = ints_array.tolist()

        return (integers,)
