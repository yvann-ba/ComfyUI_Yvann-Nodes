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
Analyzes audio input to generate audio-reactive weights and visualizations\n
It can extract specific elements from the audio, such as drums, vocals, bass, or analyze the full audio\n
Using AI-based audio separator [open-unmix](https://github.com/sigsep/open-unmix-pytorch), it separates these components from the input audio
"""
})

add_node_config("IPAdapter_Audio_Reactive_Yvann", {
    "BASE_DESCRIPTION": """
Receives "audio-reactive weights" from the "Audio Reactive Node" to control the blending and switch between images based on audio peaks\n
return images and associed weights to use with 2 IPadapter Batch, inspired by the "IPAdapter Weights" from [IPAdapter_Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
"""
})

add_node_config("Audio_PromptSchedule_Yvann", {
    "BASE_DESCRIPTION": """
Associates Inputs prompts with inputs floats into a scheduled prompt format, the output of this node need to be connected to a batch prompt schedule from [Fizz Nodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)
"""
})

add_node_config("Floats_To_Weights_Strategy_Yvann", {
    "BASE_DESCRIPTION": """
Convert a list of floats into an IPAdapter weights strategy, enabling use with "IPAdapter Weights From Strategy"\n
or "Prompt Schedule From Weights Strategy". This allows to pass audio_weights or any float list to the IPAdapter pipeline
"""
})

add_node_config("Invert_Floats_Yvann", {
    "BASE_DESCRIPTION": """
### Inverts all the individuals values of a list of floats
Inputs:

- floats: The list of float values to invert
\n
Outputs:

- floats_invert: The inverted list of float values, where all the individual values have been inversed
"""
})

add_node_config("Floats_Visualizer_Yvann", {
    "BASE_DESCRIPTION": """
### Generates a graph from one or more lists of floats to visually compare data, Useful for comparing audio weights from different Audio Reactive nodes

Inputs:

- floats: The primary floats values to visualize
- floats_optional2: (Optional) second floats values to visualize
- floats_optional3: (Optional) third floats values to visualize
\n
Parameters:
- title: Title of the graph
- x_label: Label for the x-axis
- y_label: Label for the y-axis
Outputs:

visual_graph: An image displaying the graph of the provided float sequences
"""
})

add_node_config("Mask_To_Float_Yvann", {
    "BASE_DESCRIPTION": """
Converts mask(s) input into float(s) value(s) by computing the mean pixel value of each mask
"""
})