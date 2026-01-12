import numpy as np

from ..base import ConvertNodeBase


class FloatToInt(ConvertNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS", {
                    "forceInput": True,
                    "tooltip": "List of floats to round to nearest integers"
                }),
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
