from ... import Yvann
import numpy as np


class UtilsNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"

class InvertFloats(UtilsNodeBase):
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"floats": ("FLOAT", {"forceInput": True}),
			}
		}
	RETURN_TYPES = ("FLOAT",)
	RETURN_NAMES = ("floats_invert",)
	FUNCTION = "invert_floats"

	def invert_floats(self, floats):
		floats_array = np.array(floats)
		min_value = floats_array.min()
		max_value = floats_array.max()

		# Invert the values relative to the range midpoint
		range_midpoint = (max_value + min_value) / 2.0
		floats_invert_array = (2 * range_midpoint) - floats_array
		floats_invert_array = np.round(floats_invert_array, decimals=3)

		# Convert back to list
		floats_invert = floats_invert_array.tolist()

		return (floats_invert,)
