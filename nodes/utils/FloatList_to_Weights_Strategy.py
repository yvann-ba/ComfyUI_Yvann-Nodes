from ... import Yvann

class UtilsNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"

class FloatList_to_Weights_Strategy(UtilsNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOAT", {"forceInput": True}),
                "batch_size": ("INT", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("WEIGHTS_STRATEGY",)
    RETURN_NAMES = ("WEIGHTS_STRATEGY",)
    FUNCTION = "convert"

    def convert(self, floats, batch_size):
        frames = batch_size
        
        weights_str = ", ".join(map(lambda x: f"{x:.8f}", floats))

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
        
