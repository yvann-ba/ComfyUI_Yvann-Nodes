# 👁️ ComfyUI_Yvann-Nodes  [![Instagram](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=instagram&logoColor=E4405F)](https://www.instagram.com/yvann.mp4/) [![YouTube](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=youtube&logoColor=FF0000)](https://www.youtube.com/channel/yvann.mp4)


### **With this pack of nodes, you can analyze audio, extract drums, bass, vocal tracks, and use the scheduled masks and weights to create audio-reactive animations in ComfyUI**

- [Workflows](#Workflows)
- [Nodes](#Nodes)
- [Installation](#Installation), [Contributing](#Contributing)

--- 

## Workflows

**In Progress**

---
## Nodes

### Node: Audio Reactive 🔊

Analyzes audio input to generate **audio-reactive weights** and visualizations. It can extract specific elements from the audio, such as **drums**, **vocals**, **bass**, or analyze the **full audio**. Using AI-based audio separator [open-unmix](https://github.com/sigsep/open-unmix-pytorch), it separates these components from the input audio

![Audio Reactive Yvann](./assets/AudioReactive_node_preview.png)

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **batch_size**: The number of audio frames to process
>  - **fps**: Frames per second for processing audio weights, the output of your animation need to have the same fps to be correctly synchronized
>  - **audio**: Input audio file
>  - **analysis_mode**: Selects the audio component to analyze (**Drums Only**, **Full Audio**, **Vocals Only**, **Bass Only**, **Other Audio**). This analysis is performed using AI-based audio separation models (open-unmix)
>  - **threshold**: Filters the audio weights based on sound intensity (only values above the threshold pass through)
>  - **add**: Adds a constant value to all the weights
>  - **smooth**: Smoothing factor to reduce sharp transitions between weights
>  - **multiply**: Multiplication factor to amplify the weights
>  - **add_range**: Expands the range of the weights to control output dynamic range
>  - **invert_weights**: Inverts the audio weights
>
>  **Outputs**:
>  - **audio_weights**: A float list of audio-reactive weights based on the processed audio
>  - **processed_audio**: The separated or processed audio (e.g., drums, vocals) used in the analysis
>  - **original_audio**: The original audio input without modifications
>  - **audio_visualization**: An image displaying a graph of the audio weights over time, representing the variation in intensity across the analyzed frames
>
></details>

---

### Node: IPAdapter Audio Reactive 🎵

This node receives "audio-reactive weights" from the "Audio Reactive Node" to control the blending and switch between images based on audio peaks

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **audio_weights**: A list of audio-reactive weights used to control image blending
>  - **images**: A batch of images as a tensor, used as sources for transitions (required, type IMAGE)
>  - **timing**: Timing function for blending (choices: custom, linear, ease_in_out, ease_in, ease_out, random; default: linear)
>  - **transition_frames**: The number of frames over which to blend between images (type INT, default: 7, min: 1, step: 1)
>  - **threshold**: The minimum height required for a peak to be considered (type FLOAT, default: 0.5, min: 0.0, max: 1.0, step: 0.01)
>  - **distance**: The minimum number of frames between peaks (type INT, default: 1, min: 1, step: 1)
>  - **prominence**: The relative importance of a peak (type FLOAT, default: 0.1, min: 0.0, max: 1.0, step: 0.01)
>
>  **Outputs**:
>  - **weights**: The calculated blending weights for image transitions
>  - **weights_invert**: The inverse of the calculated blending weights
>  - **image_1**: The starting image for a transition
>  - **image_2**: The ending image for a transition
>  - **prompt_index**: The frame indices at which image transitions occur
>  - **visualization**: An image visualization of audio weights, detected peaks, and image transitions
>
></details>

---

### Node: Audio Prompt Schedule 📝

This node creates a structured prompt schedule for audio-reactive animations. It associates specific prompts with defined indices, allowing precise control over prompt changes in sync with audio cues

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **prompt_indices**: A list of indices where prompts will change (required, type FLOAT)
>  - **prompts**: A multiline string of prompts to be used at each index (type STRING, default: empty)
>
>  **Outputs**:
>  - **prompt_schedule**: A string representation of the prompt schedule. Each index is associated with a prompt from the provided list
>
></details>

---

### Node: Floats To Weights Strategy 🛠️

Converts a list of floats to IPAdapter weights strategy, useful to use "IPAdapter Weights From Strategy" & "Prompt Schedule From Weights Strategy" from any float list, this way you can give the audio_weights from my audio nodes to the [IPAdapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus) pipeline 

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **floats**: The list of float values to be converted into a weights strategy
>  - **batch_size**: The number of frames you want to proceed
>
>  **Outputs**:
>  - **WEIGHTS_STRATEGY**: A dictionary containing the weights strategy used by IPAdapter, including the weights and related parameters
>
></details>

---

### Node: Floats Visualizer 🛠️

This node generates a visualization of one or more list of floats. It plots the provided float lists on a graph, allowing you to visually analyze the data, Useful if you want to compare the audio weights of differents Audio Reactive instance, for example one with your drums weights, one with your vocal weights etc...

<details>
  <summary><i>Node Parameters</i></summary>

  - **floats**: The primary list of float values to visualize
  - **title**: Title of the graph
  - **x_label**: Label for the x-axis
  - **y_label**: Label for the y-axis
  - **floats_optional2**: (Optional) A second list of float values to include in the visualization
  - **floats_optional3**: (Optional) A third list of float values to include in the visualization

  **Outputs**:
  - **visual_graph**: An image displaying the graph of the provided float sequences

</details>

---

### Node: Mask To Float 🛠️

This node converts a mask input into a float value by computing the mean pixel value of the mask. It is useful for deriving numerical insights from mask data, which can then be applied in various processing scenarios

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **mask**: The mask input from which to compute the float value (required, type MASK)
>
>  **Outputs**:
>  - **float**: A float representing the average value of the mask. This output is especially useful for analyzing or comparing different mask areas quantitatively
>
></details>

---

### Node: Invert Floats 🛠️

This node inverts all the individuals values of a list of floats

<details>
  <summary><i>Node Parameters</i></summary>

  - **floats**: The list of float values to invert

  **Outputs**:
  - **floats_invert**: The inverted list of float values, where all the individual values have been inversed

</details>

---

## Installation
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Launch ComfyUI
3. Click on "🧩 Manager" -> "Custom Nodes Manager"
4. Search for `ComfyUI_Yvann-Nodes` in the manager and install it

---
## TO DO
- [x] Node FloatList to Graph Visualization
- [x] Node Combine graph visualization
## Contributing
Want to help make this project better? Feel free to:
- Open an issue
- Submit a pull request
- Reach out to me on [LinkedIn](https://www.linkedin.com/in/yvann-barbot/)
