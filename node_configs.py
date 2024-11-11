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

add_node_config("LoadAudioSeparationModel", {
    "BASE_DESCRIPTION": """
Downloads and loads audio separation model. If the model is not already available, it will be downloaded to ComfyUI/models/audio_separation_model/


**Parameter:**

- **model**: Audio Separation model to load, [HybridDemucs](https://github.com/facebookresearch/demucs) is the most accurate, fatest and lightweight, [OpenUnmix](https://github.com/sigsep/open-unmix-pytorch) is an alternative

**Output**:

- **audio_separation_model**: Loaded audio separation model, connect it to "Audio Analysis" or "Audio Remixer" Nodes

"""
})

add_node_config("AudioAnalysis", {
    "BASE_DESCRIPTION": """
Analyzes audio input to generate audio-reactive weights and visualizations.\n
It can extract specific elements from the audio, such as drums, vocals, bass, or analyze the full audio.\n
AI audio separator preprocess the audio, parameters allow manual control over the audio weights

**Inputs:**

- **audio_separation_model**: Load model from "Load Audio Separation Model" Node, currently support HybridDemucs and Open-Unmix
- **audio**: Input audio file
- **batch_size**: Number of frames to associates audio weights with
- **fps**: Frames per second for processing audio weights

**Parameters:**

- **analysis_mode**: Selects the audio component to analyze
- **threshold**: Only weights detected in the audio above this value pass through
- **multiply**: Amplifies the weights by this factor, applied before normalization

**Outputs:**

- **graph_audio**: Image displaying a graph of the audio weights over each frames
- **processed_audio**: The separated or processed audio (e.g., drums, vocals) used in the analysis
- **original_audio**: The original audio input without modifications
- **audio_weights**: A list of audio-reactive float weights based on the processed audio
"""
})

#Audio Peaks Detection here

add_node_config("AudioIPAdapterTransitions", {
    "BASE_DESCRIPTION": """
Receives "peaks_weights" from "Audio Peaks Detection" Node to control blending and switching between images based on audio peaks.\n
Returns images and associated weights to use with two IPAdapter batches, inspired by "IPAdapter Weights" from [IPAdapter_Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus).

**Inputs:**

- **images**: Batch of images for transitions; each switches on an audio peak, the inputs images loop to match the number of peaks
- **peaks_weights**: List of audio peaks from "Audio Peaks Detection" Node

**Parameters:**

- **blend_mode**: Blending timing function; each modes smooth weights in a different way
- **transitions_length**: Number of frames used to blend between images
- **min_IPA_weight**: Affect "weights" and "weights_invert" list; correspond to the min weights that IPA gonna applies on each scheduled frame
- **max_IPA_weight**: Same as "min_IPA_weight" but for max

**Outputs:**

- **image_1**: Starting image for a transition; connect to first IPAdapter batch image input
- **weights**: Blending weights for image transitions; connect to first IPAdapter batch weight input
- **image_2**: Ending image for a transition; connect to second IPAdapter batch image input
- **weights_invert**: Inversed weights; connect to second IPAdapter batch weight input
- **graph_transitions**: Visualization of weights transitions scheduled over frames
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

#Audio AnimateDiff Schedule here

add_node_config("AudioRemixer", {
    "BASE_DESCRIPTION": """
Enables modification of the input audio by adjusting the intensity of drums, bass, vocals, or other elements. The output is the audio with the applied modifications

**Inputs:**

- **audio_separation_model**: Load model from "Load Audio Separation Model" Node, currently support HybridDemucs and Open-Unmix
- **audio**: The audio file you want to modify. You can load either mono or stereo format

**Parameters:**

- **bass_volume**: Adjusts bass volume (-10 to mute, 10 = max value to amplify)
- **drums_volume**: Adjusts drums volume
- **others_volume**: Adjusts volume of others elements
- **vocals_volume**: Adjusts vocals volume

**Outputs**:

- **merged_audio**: A composition of four separated tracks (drums, bass, vocals, other), modified as specified

"""
})
#Repeat ImagesToCount here 

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