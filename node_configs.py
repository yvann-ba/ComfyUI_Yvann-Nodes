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
    "BASE_DESCRIPTION": "Loads an audio separation model for extracting drums, vocals, bass, and other audio components. If the model is not available locally, it will be automatically downloaded to ComfyUI/models/audio_separation_model/. Choose between HybridDemucs (most accurate, fastest, and lightweight) or OpenUnmix (alternative model). The output audio_sep_model should be connected to Audio Analysis or Audio Remixer nodes."
})

add_node_config("AudioAnalysis", {
    "BASE_DESCRIPTION": "Analyzes audio to generate reactive weights and visualization graphs. Can extract specific elements like drums, vocals, or bass from the audio. Input requires an audio_sep_model from Load Audio Separation Model, the audio file, batch_size (number of frames), and fps (frames per second). Choose an analysis_mode to select which audio component to analyze. The threshold parameter sets the minimum weight value to pass through (0.0 to 1.0), and multiply amplifies the weights before normalization. Outputs include audio_weights (list of reactive weights), processed_audio (separated component), original_audio (unchanged input), and graph_audio (visual representation)."
})

add_node_config("AudioPeaksDetection", {
    "BASE_DESCRIPTION": "Detects peaks in audio weights based on threshold and minimum distance between peaks. This node identifies significant audio events to trigger visual changes or actions. The peaks_threshold parameter controls the minimum height required for peak detection (0.0 to 1.0), while min_peaks_distance sets the minimum number of frames between consecutive peaks, helping remove unwanted close peaks around bigger peaks. Outputs include peaks_weights (binary list with 1 for peak, 0 otherwise), peaks_alternate_weights (alternating pattern), peaks_index (comma-separated frame numbers), peaks_count (total detected peaks), and graph_peaks (visual overlay showing detected peaks)."
})

add_node_config("AudioIPAdapterTransitions", {
    "BASE_DESCRIPTION": "Controls IPAdapter image transitions synchronized with audio peaks. Uses peaks_weights from Audio Peaks Detection to switch between images at detected peaks. Outputs images and blending weights for two IPAdapter Batch nodes. The images input accepts a batch of images which will loop if there are fewer images than peaks. Choose a transition_mode (linear, ease_in_out, ease_in, ease_out) to control the blending curve, and set transition_length to define how many frames the blend takes. The min_IPA_weight and max_IPA_weight parameters control the IPAdapter weight range per frame. Connect image_1 and weights to the first IPAdapter Batch, and image_2 and weights_invert to the second IPAdapter Batch for smooth audio-reactive transitions."
})

add_node_config("AudioPromptSchedule", {
    "BASE_DESCRIPTION": "Associates prompts with peaks_index into a scheduled format compatible with batch prompt schedule nodes. Input peaks_index comes from Audio Peaks Detection (comma-separated frame numbers), and prompts is a multiline string where each line is a different prompt. Prompts will cycle if there are fewer prompts than peaks. Add an empty line between each individual prompt for proper formatting. The output prompt_schedule maps each peak frame index to its corresponding prompt, ready to connect to batch prompt schedule from Fizz Nodes or similar schedulers."
})

add_node_config("EditAudioWeights", {
    "BASE_DESCRIPTION": "Smooths and rescales audio weights for animation control. Connect the output to Multival (Float List) from AnimateDiff-Evolved to schedule motion with audio, or to Latent Keyframe From List to schedule ControlNet Apply. The smooth parameter (0.0 to 1.0) controls the smoothing factor where higher values result in smoother transitions. The min_range and max_range parameters define the minimum and maximum values of the rescaled weights - AnimateDiff multival works best between 0.9 and 1.3 range. Outputs include process_weights (smoothed and rescaled audio weights ready to connect to AD multival) and graph_audio (visual graph of the processed weights)."
})

add_node_config("AudioRemixer", {
    "BASE_DESCRIPTION": "Modifies input audio by adjusting the intensity of drums, bass, vocals, or other audio elements. Requires audio_sep_model from Load Audio Separation Model and an audio input file. Use the volume parameters (drums_volume, vocals_volume, bass_volume, others_volume) to adjust each component's intensity. Values range from -10.0 (mute) to 10.0 (boost), with 0.0 being the original volume. The output merged_audio is the composition of all separated tracks with the applied volume modifications mixed together."
})

add_node_config("RepeatImageToCount", {
    "BASE_DESCRIPTION": "Repeats a batch of images to reach a specified count. If the count is greater than the number of input images, the node will cycle through the images repeatedly until reaching the target count. Input an image batch and specify the desired count (minimum 1). The output is a batch of images matching exactly the specified count, useful for synchronizing image batches with audio frame counts or other batch-based operations."
})

add_node_config("InvertFloats", {
    "BASE_DESCRIPTION": "Inverts each value in a list of floats relative to the range midpoint. Takes a list of float values and flips them around the midpoint between the minimum and maximum values in the list. For example, if the range is 0.0 to 1.0, a value of 0.3 becomes 0.7. Useful for inverting audio weights or other float-based animations. The output inverted_floats contains the inverted list of float values."
})

add_node_config("FloatsVisualizer", {
    "BASE_DESCRIPTION": "Generates a visual graph from one or more lists of floats for data comparison. Particularly useful for comparing different audio weights or analyzing float-based data. Input a primary floats list, and optionally add floats_optional1 and floats_optional2 for comparison (up to 3 lists total). Customize the graph with title, x_label, and y_label parameters. Each list is plotted with a different color and line style for easy distinction. The output visual_graph is an image showing all provided float lists plotted together."
})

add_node_config("MaskToFloat", {
    "BASE_DESCRIPTION": "Converts a batch of masks into a list of float values. Each mask in the batch is converted to a single float value by calculating the mean of all pixel values in that mask. Works with individual masks or batches of masks. Input a mask (single or batch), and the output floats will be a list containing one float value per mask, representing the average intensity of each mask. Useful for converting mask data into weights for animation or other float-based operations."
})

add_node_config("FloatsToWeightsStrategy", {
    "BASE_DESCRIPTION": "Converts a list of floats into an IPAdapter weights strategy format. This node formats float lists into a dictionary structure compatible with IPAdapter Weights From Strategy or Prompt Schedule From Weights Strategy nodes. Input a list of float values, and the output WEIGHTS_STRATEGY will be a properly formatted dictionary containing the weights, timing information, frame counts, and method settings required by IPAdapter pipeline nodes. This allows you to use custom float-based weights with the IPAdapter system."
})

add_node_config("FloatToInt", {
    "BASE_DESCRIPTION": "Converts a list of float values to integers by rounding each value to the nearest whole number. Takes any list of floats and outputs a list of integers (INTS type) with the same number of elements. Each float is rounded using standard rounding rules (0.5 rounds up). Useful for converting continuous weight values into discrete integer indices or counts for frame-based operations."
})
