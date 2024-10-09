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

add_node_config("Audio_Reactive_Yvann", {
    "BASE_DESCRIPTION": """
### Analyze audio to generate audio-reactive weights and masks\n
### Compatible with Ipadapter, AnimateDiff, ControlNets for video generation
"""
})

# #### Inputs
# - `audio`: Audio to analyze
# - `batch_size`: Number of frames to process
# - `fps`: Frame rate for the output vid (high fps works better like 30, 50 or 60)

# #### Parameters
# - `analysis_mode`: Select the audio component to analyze
# - `threshold`, `add`, `smooth`, `multiply`: Adjust and fine-tune weights.

# #### Outputs:
# - `Audio`: Processed audio
# - `Audio Weights`: Audio-reactive float values for each frame
# - `Audio Masks`: Converted weights for masking
# - `Weights Graph`: A visual plot of the audio weights