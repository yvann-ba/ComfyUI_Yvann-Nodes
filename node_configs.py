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

#### Analyzes video audio to generate audio-reactive weights and masks

#### Parameters:
- `smoothing_factor`: Smooths audio weights (higher values = smoother transitions)
- `global_intensity`: Adjusts overall intensity of audio weights

#### Inputs:
- `video_frames`: Input video
- `audio`: Video audio

#### Outputs:
- `Audio`: Same as input
- `Audio Weights`: Audio-reactive float values for each frame, usable with IPAdapter, CN, etc.
- `Audio Masks`: Audio weights converted to masks usable with any mask input: Latent Noise Mask, CN, IPadapter etc..
- `Weights Graph`: Visual Graph of "Audio Weights", help to adjust smoothing_factor/global_intensity
"""
})

add_node_config("Audio_Drums_Analysis_Yvann", {
    "BASE_DESCRIPTION": """
#### Isolates and analyzes drums sounds in video audio to generate drum-reactive weights and masks

#### Parameters:
- `smoothing_factor`: Smooths drums audio weights (higher values = smoother transitions)
- `global_intensity`: Adjusts overall intensity of drums audio weights

#### Inputs:
- `video_frames`: Input video
- `audio`: Video audio

#### Outputs:
- `Drums Audio`: Isolated drums audio from input
- `Drums Weights`: Drums-reactive float values for each frame, usable with IPAdapter, CN, etc.
- `Drums Masks`: Drums weights converted to masks usable with any mask input: Latent Noise Mask, CN, IPadapter etc..
- `Weights Graph`: Visual Graph of "Drums Weights",help to adjust smoothing_factor/global_intensity
"""
})

add_node_config("Audio_Vocals_Analysis_Yvann", {
    "BASE_DESCRIPTION": """
    
#### Isolates and analyzes vocal sounds in video audio to generate vocal-reactive weights and masks

#### Parameters:
- `smoothing_factor`: Smooths vocals audio weights (higher values = smoother transitions)
- `global_intensity`: Adjusts overall intensity of vocals audio weights

#### Inputs:
- `video_frames`: Input video
- `audio`: Video audio

#### Outputs:
- `Vocals Audio`: Isolated vocals audio from input
- `Vocals Weights`: Vocals-reactive float values for each frame, usable with IPAdapter, CN, etc.
- `Vocals Masks`: Vocals weights converted to masks usable with any mask input: Latent Noise Mask, CN, IPadapter etc..
- `Weights Graph`: Visual Graph of "Vocals Weights", help to adjust smoothing_factor/global_intensity
"""
})
