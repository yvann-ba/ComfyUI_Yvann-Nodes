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

add_node_config("IPAdapterAudioReactive", {
    "BASE_DESCRIPTION": """
Receives audio-reactive weights to control blending and switching between images based on audio peaks.\n
Returns images and associated weights to use with two IPAdapter batches, inspired by "IPAdapter Weights" from [IPAdapter_Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus).

**Inputs:**

- **images**: Batch of images for transitions; each switches on an audio peak
- **audio_weights**: List of audio-reactive weights from the "Audio Reactive" node

**Parameters:**

- **blend_mode**: Blending timing function; different modes smooth weights differently
- **transition_length**: Frames over which to blend between images
- **threshold**: Minimum height for a peak to be considered
- **image_2**: The ending image for a transition, connect it to the second IPadapter batch in the image input
- **max_IPA_weight**: the range end value for IPAdapter weights, 1 by default, 1.3 for strong style transfer

**Outputs:**

- **image_1**: Starting image for a transition; connect to first IPAdapter batch image input
- **weights**: Blending weights for image transitions; connect to first IPAdapter batch weight input
- **image_2**: Ending image for a transition; connect to second IPAdapter batch image input
- **weights_invert**: Inverse blending weights; connect to second IPAdapter batch weight input
- **graph_audio_index**: Visualization of audio weights, detected peaks, and image transitions
"""
})


add_node_config("AudioAnalysis", {
    "BASE_DESCRIPTION": """
Analyzes audio input to generate audio-reactive weights and visualizations.\n
It can extract specific elements from the audio, such as drums, vocals, bass, or analyze the full audio.\n
Using AI-based audio separator [open-unmix](https://github.com/sigsep/open-unmix-pytorch), the parameters allow manual control over the audio weights.

**Inputs:**

- **audio_separation_model**: Load audio seperation model by the node in the node pack 
- **audio**: Input audio file
- **batch_size**: Number of audio frames to process
- **fps**: Frames per second for processing audio weights; should match your animation's fps

**Parameters:**

- **analysis_mode**: Selects the audio component to analyze
- **threshold**: Only weights above this value pass through
- **multiply**: Amplifies the weights by this factor, applied before normalization

**Outputs:**

- **graph_audio**: Image displaying a graph of the audio weights over time
- **processed_audio**: The separated or processed audio (e.g., drums, vocals) used in the analysis
- **original_audio**: The original audio input without modifications
- **audio_weights**: A list of audio-reactive float weights based on the processed audio
"""
})


add_node_config("AudioPromptSchedule", {
    "BASE_DESCRIPTION": """
Associates input prompts with floats into a scheduled prompt format.\n
Connect the output to a batch prompt schedule from [Fizz Nodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes).\n
Make sure to include an empty lines between each differents prompts

**Inputs:**

- **peaks_index**: Indices where prompts will change (FLOAT)
- **prompts**: Multiline string of prompts to use at each index

**Outputs:**

- **prompt_schedule**: String representation of the prompt schedule; associates each index with a prompt
"""
})

add_node_config("FloatsToWeightsStrategy", {
    "BASE_DESCRIPTION": """
Converts a list of floats into an IPAdapter weights strategy.\n
Use with "IPAdapter Weights From Strategy" or "Prompt Schedule From Weights Strategy" to pass audio weights or any float list to the IPAdapter pipeline.

**Inputs:**

- **floats**: List of float values to convert into a weights strategy

**Parameters:**

**Outputs:**

- **WEIGHTS_STRATEGY**: Dictionary containing the weights strategy for IPAdapter
"""
})

add_node_config("InvertFloats", {
    "BASE_DESCRIPTION": """
Inverts all individual values of a list of floats.

**Inputs:**

- **floats**: List of float values to invert

**Outputs:**

- **floats_invert**: Inverted list of float values
"""
})

add_node_config("FloatsVisualizer", {
    "BASE_DESCRIPTION": """
Generates a graph from one or more lists of floats to visually compare data.\n
Useful for comparing audio weights from different Audio Reactive nodes.

**Inputs:**

- **floats**: Primary list of float values to visualize
- **floats_optional2**: (Optional) Second list of float values
- **floats_optional3**: (Optional) Third list of float values

**Parameters:**

- **title**: Graph title
- **x_label**: Label for the x-axis
- **y_label**: Label for the y-axis

**Outputs:**

- **visual_graph**: Image displaying the graph of the provided float sequences
"""
})

add_node_config("MaskToFloat", {
    "BASE_DESCRIPTION": """
Converts mask inputs into float values by computing the mean pixel value of each mask.

**Inputs:**

- **mask**: Mask input to compute the float value from

**Outputs:**

- **float**: Float representing the average value of the mask
"""
})
