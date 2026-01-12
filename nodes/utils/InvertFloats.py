import numpy as np

from ..base import UtilsNodeBase


class InvertFloats(UtilsNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS", {
                    "forceInput": True,
                    "tooltip": "List of float values to invert relative to range midpoint"
                }),
            }
        }

    RETURN_TYPES = ("FLOATS",)
    RETURN_NAMES = ("inverted_floats",)
    FUNCTION = "invert_floats"

    def invert_floats(self, floats):
        floats_array = np.array(floats)
        min_value = floats_array.min()
        max_value = floats_array.max()

        # Invert the values relative to the range midpoint
        range_midpoint = (max_value + min_value) / 2.0
        floats_invert_array = (2 * range_midpoint) - floats_array
        floats_invert_array = np.round(floats_invert_array, decimals=6)

        # Convert back to list
        floats_invert = floats_invert_array.tolist()

        return (floats_invert,)
