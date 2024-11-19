# 👁️ ComfyUI_Yvann-Nodes  [![Instagram](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=instagram&logoColor=E4405F)](https://www.instagram.com/yvann.mp4/) [![YouTube](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=youtube&logoColor=FF0000)](https://www.youtube.com/channel/yvann.mp4)

#### Made with the help of [Lilien](https://x.com/Lilien_RIG) <a href="https://x.com/Lilien_RIG"><img src="https://github.com/user-attachments/assets/26a483b9-cfe6-4666-af0d-52d40ee65dcf" alt="unnamed" width="35"></a>

#### **With this pack of nodes, you can analyze audio, extract drums, bass, vocal tracks, and use the scheduled masks and weights to create AI-generated audio-reactive animations in ComfyUI**

#### **Works with IPAdapter, AnimateDiff, ControlNets, Prompts Schedules**

- [Workflows](#Workflows)
- [Nodes](#Nodes)
- [Installation](#Installation)

--- 

## Workflows


### Audio Reactive Images To Video :

##### Workflow File (click on the link, then on "download raw file" button at the right and drop the file into ComfyUI) 📜
[AudioReactive_ImagesToVideo_Yvann.json](WORKFLOW_AUDIO_REACTIVE/AudioReactive_ImagesToVideo_Yvann.json)

##### Youtube Tutorial (Click on the image)
[![Video](https://img.youtube.com/vi/O2s6NseXlMc/maxresdefault.jpg)](https://www.youtube.com/watch?v=O2s6NseXlMc)

##### Workflow Demo Render *(ENABLE VIDEO SOUND)* 🔊

https://github.com/user-attachments/assets/1e6590fc-e0d7-42d7-a205-433adf6c405c

><details>
>  <summary><i>AudioReactive VideoToVideo</i></summary>
>
>##### Workflow File (click on the link, then on "download raw file" button at the right and drop the file into ComfyUI) 📜
>[AudioReactive_VideoToVideo_Yvann.json](./WORKFLOW_AUDIO_REACTIVE/AudioReactive_VideoToVideo_Yvann.json)
>
>##### Workflow Demo Render *(ENABLE VIDEO SOUND)* 🔊
>
>https://github.com/user-attachments/assets/6b0aa544-aa20-4257-b6be-28673082c7ef
>
>##### Youtube Tutorial (Click on the image)
>[![Video](https://img.youtube.com/vi/BiQHWKP3q0c/maxresdefault.jpg)](https://www.youtube.com/watch?v=BiQHWKP3q0c)
>##### Workflow Preview
>![videotovideo](https://github.com/user-attachments/assets/62dd4443-2e7d-48b5-aa0a-6dd49e3f90ac)
>[CIVITAI Workflow Page](https://civitai.com/models/867298)
>
></details>

---

## Nodes

###  Audio Analysis 🔍

Analyzes audio to generate reactive weights and graph. Can extract specific elements like drums, vocals, bass. Parameters allow manual control over audio weights

![preview](https://github.com/user-attachments/assets/4959a654-d1d1-478a-ac42-8068de32d581)

><details>
>  <summary><i>Node Parameters</i></summary>
>
> - **audio_sep_model**: Loaded model from "Load Audio Separation Model"
> - **audio**: Input audio file
> - **batch_size**: Number of frames to associate with audio weights
> - **fps**: Frames per second for processing audio weights
> 
> **Parameters:**
> 
> - **analysis_mode**: Select audio component to analyze
> - **threshold**: Minimum weight value to pass through
> - **multiply**: Amplification factor for weights before normalization
> 
> **Outputs:**
> 
> - **graph_audio**: Graph image of audio weights over frames
> - **processed_audio**: Separated or processed audio (e.g., drums vocals)
> - **original_audio**: Original unmodified audio input
> - **audio_weights**: List of audio-reactive weights based on processed audio
>
></details>

---

###  Load Audio Separation Model 🎧

Load an audio separation model, If unavailable downloads to `ComfyUI/models/audio_separation_model/

![preview](https://github.com/user-attachments/assets/7fb58067-a79b-4a53-9ae5-524a04ed37b6)

><details>
>  <summary><i>Node Parameters</i></summary>
> 
>   - **model**: Audio separation model to load
>   - [HybridDemucs](https://github.com/facebookresearch/demucs): Most accurate fastest and lightweight
>   - [OpenUnmix](https://github.com/sigsep/open-unmix-pytorch): Alternative model
> 
> **Outputs:**
> 
>   - **audio_sep_model**: Loaded audio separation model<br>
>   Connect it to "Audio Analysis" or "Audio Remixer"

></details>

---
###  Audio Peaks Detection 📈

Detects peaks in audio weights based on a threshold and minimum distance. Identifies significant audio events to trigger visual changes or actions.

![preview](https://github.com/user-attachments/assets/e5f66608-bb91-443b-9478-707eba48e521)

><details>
>  <summary><i>Node Parameters</i></summary>
>
>   - **peaks_threshold**: Threshold for peak detection
>   - **min_peaks_distance**: Minimum frames between consecutive peaks help remove close unwanted peaks around big peaks
>   
>   **Outputs:**
>   
>   - **peaks_weights**: Binary list indicating peak presence (1 for peak 0 otherwise)
>   - **peaks_alternate_weights**: Alternating binary list based on detected peaks
>   - **peaks_index**: String of peak indices
>   - **peaks_count**: Total number of detected peaks
>   - **graph_peaks**: Visualization image of detected peaks over audio weights
>
></details>

---

###  Audio IP Adapter Transitions 🔄

Uses "peaks_weights" from "Audio Peaks Detection" to control image transitions based on audio peaks. Outputs images and weights for two IPAdapter batches, logic from "IPAdapter Weights", [IPAdapter_Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)

![preview](https://github.com/user-attachments/assets/60204704-5916-44a3-a33b-c99b1732f189)

><details>
>  <summary><i>Node Parameters</i></summary>
>   - **images**: Batch of images for transitions, Loops images to match peak count
>   - **peaks_weights**: List of audio peaks from "Audio Peaks Detection"
>   
>   **Parameters:**
>   
>   - **blend_mode**: transition method applied to weights
>   - **transitions_length**: Frames used to blend between images
>   - **min_IPA_weight**: Minimum weight applied by IPAdapter per frame
>   - **max_IPA_weight**: Maximum weight applied by IPAdapter per frame
>   
>   **Outputs:**
>   
>   - **image_1**: Starting image for transition Connect to first IPAdapter batch "image"
>   - **weights**: Blending weights for transitions Connect to first IPAdapter batch "weight"
>   - **image_2**: Ending image for transition Connect to second IPAdapter batch "image"
>   - **weights_invert**: Inversed weights Connect to second IPAdapter batch "weight"
>   - **graph_transitions**: Visualization of weight transitions over frames
></details>

---
###  Audio Prompt Schedule 📝

Associates "prompts" with "peaks_index" into a scheduled format. Connect output to "batch prompt schedule" of [Fizz Nodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) add an empty line between each individual prompts

![preview](https://github.com/user-attachments/assets/cec2ad2a-94c4-44df-a12a-f4d4509cefb1)

><details>
>  <summary><i>Node Parameters</i></summary>
>   - **peaks_index**: frames where peaks occurs from "Audio Peaks Detections" 
>   - **prompts**: Multiline string of prompts for each index
>   
>   **Outputs:**
>   
>   - **prompt_schedule**: String mapping each audio index to a prompt
></details>

---

###  Audio Remixer 🎛️

Modify input audio by adjusting the intensity of drums bass vocals or others elements

![preview](https://github.com/user-attachments/assets/ada877fa-baa8-447d-bbad-5c30ac6cdadb)

><details>
>  <summary><i>Node Parameters</i></summary>
>   - **audio_sep_model**: Loaded model from "Load Audio Separation Model"
>   - **audio**: Input audio file
>   
>   **Parameters:**
>   
>   - **bass_volume**: Adjusts bass volume
>   - **drums_volume**: Adjusts drums volume
>   - **others_volume**: Adjusts others elements' volume
>   - **vocals_volume**: Adjusts vocals volume
>   
>   **Outputs:**
>   
>   - **merged_audio**: Composition of separated tracks with applied modifications
></details>

---
###  Repeat Image To Count 🔁

Repeats images N times, Cycles inputs if N > images

![Preview](https://github.com/user-attachments/assets/3fa1059e-2aed-4375-b5d2-de850f6cd8c6)

><details>
>  <summary><i>Node Parameters</i></summary>
>   - **mask**: Mask input to convert
>   
>   **Outputs:**
>   
>   - **float**: Float value
></details>

---
###  Invert Floats 🔄

Inverts each value in a list of floats

![Preview](https://github.com/user-attachments/assets/bb90cc61-dbbc-42cd-bc26-55f25efbb6aa)

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **floats**: List of float values to invert.
>
>  **Outputs**:
>  - **inverted_floats**: Inverted list of float values.
>
></details>

---

###  Floats Visualizer 📈

Generates a graph from floats for visual data comparison<br>
Useful to compare audio weights

![preview](https://github.com/user-attachments/assets/615cf287-e7d6-4dce-92f9-3d691aae43af)

><details>
>  <summary><i>Node Parameters</i></summary>
>
>   - **floats**: Primary list of floats to visualize
>   - **floats_optional1**: (Optional) Second list of floats
>   - **floats_optional2**: (Optional) Third list of floats
>   
>   **Parameters:**
>   
>   - **title**: Graph title
>   - **x_label**: Label for the x-axis
>   - **y_label**: Label for the y-axis
>   
>   **Outputs:**
>   
>   - **visual_graph**: Visual graph of provided floats
></details>

---
###  Mask To Float 🎭

Converts mask into float works with batch of mask

![preview](https://github.com/user-attachments/assets/159f2a19-d8b3-4064-b416-07a17cc32ef0)

><details>
>  <summary><i>Node Parameters</i></summary>
>
>  - **mask**: Mask input to convert.
>
>  **Outputs**:
>  - **float**: Float value representing the average value of the mask.
>
></details>

---
###  Floats To Weights Strategy 🏋️

Converts a list of floats into an IPAdapter weights strategy format. Use with "IPAdapter Weights From Strategy" or "Prompt Schedule From Weights Strategy" to integrate output into [IPAdapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus) pipeline

![preview](https://github.com/user-attachments/assets/a9899ea9-c67f-42a2-8040-2af8a2744849)

><details>
>  <summary><i>Node Parameters</i></summary>
>   
>   **Inputs:**
>   
>   - **floats**: List of float values to convert
>   
>   **Outputs:**
>   
>   - **WEIGHTS_STRATEGY**: Dictionary of the weights strategy

---

### Installation
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Launch ComfyUI
3. Click on "🧩 Manager" -> "Custom Nodes Manager"
4. Search for `ComfyUI_Yvann-Nodes` in the manager and install it

---
#### *Giving a ⭐ to this repo is the best way to support us (:*
