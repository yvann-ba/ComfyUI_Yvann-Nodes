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
##Parameters
- `video_frames`: Input video frames to be processed.
- `audio`: Input audio to be processed.
- `frame_rate`: Frame rate of the video.
- `smoothing_factor`: Smoothing factor for the audio analysis
- `global_intensity`: Global intensity for the audio analysis
"""
})

add_node_config("Audio_Drums_Analysis_Yvann", {
    "BASE_DESCRIPTION": """
##Parameters
- `video_frames`: Input video frames to be processed.
- `audio`: Input audio to be processed.
- `frame_rate`: Frame rate of the video.
- `smoothing_factor`: Smoothing factor for the audio analysis
- `global_intensity`: Global intensity for the audio analysis
"""
})

add_node_config("Audio_Vocals_Analysis_Yvann", {
    "BASE_DESCRIPTION": """
##Parameters
- `video_frames`: Input video frames to be processed.
- `audio`: Input audio to be processed.
- `frame_rate`: Frame rate of the video.
- `smoothing_factor`: Smoothing factor for the audio analysis
- `global_intensity`: Global intensity for the audio analysis
"""
})
