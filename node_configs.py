#NOTE: this abstraction allows for both the documentation to be centrally managed and inherited
from abc import ABCMeta
class NodeConfigMeta(type):
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        if name in NODE_CONFIGS:
            for key, value in NODE_CONFIGS[name].items():
                setattr(new_class, key, value)
        return new_class

class CombinedMeta(NodeConfigMeta, ABCMeta):
    pass

def add_node_config(node_name, config):
    NODE_CONFIGS[node_name] = config

NODE_CONFIGS = {}

add_node_config("Audio_Analysis_Yvann", {
    "BASE_DESCRIPTION": """

### Analyzes video audio to generate audio-reactive weights and masks

#### Parameters:
- `smoothing_factor`: Smooths audio weights (higher values = smoother transitions)
- `global_intensity`: Adjusts overall intensity of audio weights

#### Inputs:
- `video_frames`: Input video
- `audio`: Video audio

#### Outputs:
- `Audio`: Same as input
- `Audio Weights`: Audio-reactive float values for each frame, usable with IPAdapter, CN, etc.
- `Audio Masks`: Audio weights converted to masks usable withLatent NoiseMask, CN, IPadapter...
- `Weights Graph`: Visual Graph of "Audio Weights"
"""
})

add_node_config("Math_Float_List", {
    "BASE_DESCRIPTION": """
    Not yet
    """
})