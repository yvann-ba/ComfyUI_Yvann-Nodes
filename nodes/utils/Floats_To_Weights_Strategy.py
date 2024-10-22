from ... import Yvann

class UtilsNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"

class Floats_To_Weights_Strategy(UtilsNodeBase):
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"floats": ("FLOAT", {"forceInput": True}),
			}
		}
	RETURN_TYPES = ("WEIGHTS_STRATEGY",)
	RETURN_NAMES = ("WEIGHTS_STRATEGY",)
	FUNCTION = "convert"

	def convert(self, floats):
		frames = len(floats)
		weights_str = ", ".join(map(lambda x: f"{x:.3f}", floats))

		weights_strategy = {
		"weights": weights_str,
		"timing": "custom",
		"frames": frames,
		"start_frame": 0,
		"end_frame": frames,
		"add_starting_frames": 0,
		"add_ending_frames": 0,
		"method": "full batch",
		"frame_count": frames,
		}
		return (weights_strategy,)
