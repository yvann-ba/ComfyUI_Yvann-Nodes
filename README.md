# 👁️ ComfyUI_Yvann-Nodes  [![Instagram](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=instagram&logoColor=E4405F)](https://www.instagram.com/yvann.mp4/) [![YouTube](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=youtube&logoColor=FF0000)](https://www.youtube.com/channel/yvann.mp4)


#### **With this pack of nodes, you can analyze audio, extract drums, bass, vocal tracks, and use the outputted masks and scheduled weights to create audio-reactive animations in ComfyUI**

---
# Table of contents
- [Workflows](#Workflows)
- [Nodes](#Nodes)
- [Installation](#Installation), [Contributing](#Contributing)

--- 

### Workflows

**In Progress**

---
### Nodes

#### Node: Audio Reactive

This node analyzes audio input to generate **audio-reactive weights** and visualizations. It can extract specific elements from the audio, such as **drums**, **vocals**, **bass**, or analyze the **full audio**. Using AI-based audio separator [open-unmix](https://github.com/sigsep/open-unmix-pytorch), it separates these components from the input audio

![Audio Reactive Yvann](./docs/audio-reactive-yvann.png)

The various parameters offer advanced control over how the audio input is transformed into weights, providing flexibility in shaping the audio-driven visual effects. 

The node parameters allow manual adjustment, offering fine-grained control over how the audio data is interpreted and converted into weights for reactive animations or visual effects.



<details>
  <summary><i>Parameters</i></summary>

  - **batch_size**: The number of audio frames to process
  - **fps**: Frames per second for processing audio weights, the output of your animation need to have the same fps to be correctly synchronized
  - **audio**: Input audio file
  - **analysis_mode**: Selects the audio component to analyze (**Drums Only**, **Full Audio**, **Vocals Only**, **Bass Only**, **Other Audio**). This analysis is performed using AI-based audio separation models (open-unmix)
  - **threshold**: Filters the audio weights based on sound intensity (only values above the threshold pass through)
  - **add**: Adds a constant value to all the weights
  - **smooth**: Smoothing factor to reduce sharp transitions between weights
  - **multiply**: Multiplication factor to amplify the weights
  - **add_range**: Expands the range of the weights to control output dynamic range
  - **invert_weights**: Inverts the audio weights

  **Outputs**:
  - **audio_weights**: A float list of audio-reactive weights based on the processed audio
  - **processed_audio**: The separated or processed audio (e.g., drums, vocals) used in the analysis
  - **original_audio**: The original audio input without modifications
  - **audio_visualization**: An image displaying a graph of the audio weights over time, representing the variation in intensity across the analyzed frames

</details>

---

### Installation
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Launch ComfyUI
3. Click on "🧩 Manager" -> "Custom Nodes Manager"
4. Search for `ComfyUI_Yvann-Nodes` in the manager and install it

---

### Contributing
Want to help make this project better? Feel free to:
- Open an issue
- Submit a pull request
- Reach out to me on [LinkedIn](https://www.linkedin.com/in/yvann-barbot/)
