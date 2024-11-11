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
Load an audio separation model, If unavailable :<br>
downloads to `ComfyUI/models/audio_separation_model/

**Parameters:**

- **model**: Audio separation model to load
  - [HybridDemucs](https://github.com/facebookresearch/demucs): Most accurate fastest and lightweight
  - [OpenUnmix](https://github.com/sigsep/open-unmix-pytorch): Alternative model

**Outputs:**

- **audio_sep_model**: Loaded audio separation model<br>
Connect it to "Audio Analysis" or "Audio Remixer"
    """
})

add_node_config("AudioAnalysis", {
    "BASE_DESCRIPTION": """
Analyzes audio to generate reactive weights and graph<br>
Can extract specific elements like drums, vocals, bass<br>
Parameters allow manual control over audio weights

**Inputs:**

- **audio_sep_model**: Loaded model from "Load Audio Separation Model"
- **audio**: Input audio file
- **batch_size**: Number of frames to associate with audio weights
- **fps**: Frames per second for processing audio weights

**Parameters:**

- **analysis_mode**: Select audio component to analyze
- **threshold**: Minimum weight value to pass through
- **multiply**: Amplification factor for weights before normalization

**Outputs:**

- **graph_audio**: Graph image of audio weights over frames
- **processed_audio**: Separated or processed audio (e.g., drums vocals)
- **original_audio**: Original unmodified audio input
- **audio_weights**: List of audio-reactive weights based on processed audio
    """
})

add_node_config("AudioPeaksDetection", {
    "BASE_DESCRIPTION": """
Detects peaks in audio weights based on a threshold and minimum distance<br>
Identifies significant audio events to trigger visual changes or actions

**Inputs:**

- **audio_weights**: "audio_weights" from "Audio Analysis"

**Parameters:**

- **peaks_threshold**: Threshold for peak detection
- **min_peaks_distance**: Minimum frames between consecutive peaks<br>
help remove close unwanted peaks around big peaks

**Outputs:**

- **peaks_weights**: Binary list indicating peak presence (1 for peak 0 otherwise)
- **peaks_alternate_weights**: Alternating binary list based on detected peaks
- **peaks_index**: String of peak indices
- **peaks_count**: Total number of detected peaks
- **graph_peaks**: Visualization image of detected peaks over audio weights
    """
})

add_node_config("AudioIPAdapterTransitions", {
    "BASE_DESCRIPTION": """
Uses "peaks_weights" from "Audio Peaks Detection" to control image transitions based on audio peaks<br>
Outputs images and weights for two IPAdapter batches, logic from "IPAdapter Weights", [IPAdapter_Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)

**Inputs:**

- **images**: Batch of images for transitions, Loops images to match peak count
- **peaks_weights**: List of audio peaks from "Audio Peaks Detection"

**Parameters:**

- **blend_mode**: transition method applied to weights
- **transitions_length**: Frames used to blend between images
- **min_IPA_weight**: Minimum weight applied by IPAdapter per frame
- **max_IPA_weight**: Maximum weight applied by IPAdapter per frame

**Outputs:**

- **image_1**: Starting image for transition Connect to first IPAdapter batch "image"
- **weights**: Blending weights for transitions Connect to first IPAdapter batch "weight"
- **image_2**: Ending image for transition Connect to second IPAdapter batch "image"
- **weights_invert**: Inversed weights Connect to second IPAdapter batch "weight"
- **graph_transitions**: Visualization of weight transitions over frames
    """
})

add_node_config("AudioPromptSchedule", {
    "BASE_DESCRIPTION": """
Associates "prompts" with "peaks_index" into a scheduled format<br>
Connect output to "batch prompt schedule" of [Fizz Nodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)<br>
add an empty line between each individual prompts

**Inputs:**

- **peaks_index**: frames where peaks occurs from "Audio Peaks Detections" 
- **prompts**: Multiline string of prompts for each index

**Outputs:**

- **prompt_schedule**: String mapping each audio index to a prompt
    """
})

add_node_config("AudioAnimateDiffSchedule", {
    "BASE_DESCRIPTION": """
Smooths and rescales audio weights, Connect to "Multival [Float List]"<br>
from [AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) to schedule motion with audio

**Inputs:**

- **any_audio_weights**: audio weights from "Audio Peaks Detection"<br>
or "Audio Analysis", basically any *_weights audio

**Parameters:**

- **smooth**: Smoothing factor (0.0 to 1.0) Higher values result in smoother transitions
- **min_range**: Minimum value of the rescaled weights<br>
AnimateDiff multival works better between 0.9 and 1.3 range
- **max_range**: Maximum value of the rescaled weights

**Outputs:**

- **float_val**: Smoothed and rescaled audio weights<br>
connect it to AD multival to influence the motion with audio
    """
})

add_node_config("AudioRemixer", {
    "BASE_DESCRIPTION": """
Modify input audio by adjusting the intensity of drums bass vocals or others elements

**Inputs:**

- **audio_sep_model**: Loaded model from "Load Audio Separation Model"
- **audio**: Input audio file

**Parameters:**

- **bass_volume**: Adjusts bass volume
- **drums_volume**: Adjusts drums volume
- **others_volume**: Adjusts others elements' volume
- **vocals_volume**: Adjusts vocals volume

**Outputs:**

- **merged_audio**: Composition of separated tracks with applied modifications
    """
})

add_node_config("RepeatImageToCount", {
    "BASE_DESCRIPTION": """
 Repeats images N times, Cycles inputs if N > images
**Inputs:**

- **image**: Batch of input images to repeat
- **count**: Number of repetitions

**Outputs:**

- **images**: Batch of repeated images matching the specified count
    """
})

add_node_config("InvertFloats", {
    "BASE_DESCRIPTION": """
Inverts each value in a list of floats

**Inputs:**

- **floats**: List of float values to invert

**Outputs:**

- **inverted_floats**: Inverted list of float values
    """
})

add_node_config("FloatsVisualizer", {
    "BASE_DESCRIPTION": """
Generates a graph from floats for visual data comparison<br>
Useful to compare audio weights

**Inputs:**

- **floats**: Primary list of floats to visualize
- **floats_optional1**: (Optional) Second list of floats
- **floats_optional2**: (Optional) Third list of floats

**Parameters:**

- **title**: Graph title
- **x_label**: Label for the x-axis
- **y_label**: Label for the y-axis

**Outputs:**

- **visual_graph**: Visual graph of provided floats
    """
})

add_node_config("MaskToFloat", {
    "BASE_DESCRIPTION": """
Converts mask into float<br>
works with batch of mask
**Inputs:**

- **mask**: Mask input to convert

**Outputs:**

- **float**: Float value
    """
})

add_node_config("FloatsToWeightsStrategy", {
    "BASE_DESCRIPTION": """
Converts a list of floats into an IPAdapter weights strategy format<br>
Use with "IPAdapter Weights From Strategy" or "Prompt Schedule From Weights Strategy"<br>
to integrate output into [IPAdapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus) pipeline

**Inputs:**

- **floats**: List of float values to convert

**Outputs:**

- **WEIGHTS_STRATEGY**: Dictionary of the weights strategy
    """
})
