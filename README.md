# 🔊 ComfyUI_Yvann-Nodes [![YouTube Tutorials](https://img.shields.io/badge/Workflows_Tutorial-white?style=for-the-badge&logo=youtube&logoColor=FF0000)](https://www.youtube.com/channel/yvann_ba)

#### Made with the help of [Lilien](https://x.com/Lilien_RIG) <a href="https://x.com/Lilien_RIG"><img src="https://github.com/user-attachments/assets/26a483b9-cfe6-4666-af0d-52d40ee65dcf" alt="unnamed" width="35"></a>

### **A pack of custom nodes that enable audio reactivity within [ComfyUI](https://github.com/comfyanonymous/ComfyUI), allowing you to generate AI-driven animations that sync with music**

---

## What Does This Do?

- **Create** Audio Reactive AI videos, enable controls over AI generations styles, content and composition with any audio
- **Simple**: Just Drop one of our [Workflows](/WORKFLOW_AUDIO_REACTIVE) in ComfyUI and specify your audio and visuals input
- **Flexible**: Works with existing ComfyUI AI tech and nodes (eg: IPAdapter, AnimateDiff, ControlNet, etc.)

---

## Quick Setup

- Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)

### Pick a Workflow (Images → Video or Video → Video)

1. **Images → Video**  
   - Takes a **set of images** plus an **audio** track.  
   - *Watch Tutorial*:  
     [![Images2Video](https://img.youtube.com/vi/O2s6NseXlMc/maxresdefault.jpg)](https://www.youtube.com/watch?v=O2s6NseXlMc)  
   - *Example Render (Sound On)*:
    [https://github.com/user-attachments/assets/1e6590fc-e0d7-42d7-a205-433adf6c405c](https://github.com/user-attachments/assets/1e6590fc-e0d7-42d7-a205-433adf6c405c)
2. **Video → Video**  
   - Takes a **source video** plus an **audio** track.  
   - *Watch Tutorial*:  
     [![Video2Video](https://img.youtube.com/vi/BiQHWKP3q0c/maxresdefault.jpg)](https://www.youtube.com/watch?v=BiQHWKP3q0c)  
   - *Example Render (Sound On)*:
   [https://github.com/user-attachments/assets/6b0aa544-aa20-4257-b6be-28673082c7ef](https://github.com/user-attachments/assets/6b0aa544-aa20-4257-b6be-28673082c7ef)


---

### Load Your Chosen Workflow in ComfyUI

1. **Download** the `.json` file for the workflow you picked:  
   - [AudioReactive_ImagesToVideo_Yvann.json](WORKFLOW_AUDIO_REACTIVE/AudioReactive_ImagesToVideo_Yvann.json)  
   - [AudioReactive_VideoToVideo_Yvann.json](WORKFLOW_AUDIO_REACTIVE/AudioReactive_VideoToVideo_Yvann.json)

2. **Drop** the `.json` file into the **ComfyUI window**.  

3. **Open the “🧩 Manager”** → **“Install Missing Custom Nodes”**  
   - Install each pack of nodes that appears.  
   - **Restart** ComfyUI if prompted.

4. **Set Your Inputs & Generate**  
   - Provide the inputs needed (everything explained [here](https://www.youtube.com/@yvann_ba)
   - Click **Queue** button to produce your **audio-reactive** animation!

**That’s it!** Have fun playing with the differents settings now !!
(if you have any questions or problems, check my [Youtube Tutorials](https://www.youtube.com/@yvann_ba)

---

## Advanced/Optional Node Details

<details>
  <summary><strong>Click to Expand: Node-by-Node Reference</strong></summary>

### Audio Analysis 🔍
Analyzes audio to generate reactive weights for each frame.  
<details>
  <summary><em>Node Parameters</em></summary>

- **audio_sep_model**: Model from "Load Audio Separation Model"  
- **audio**: Input audio file  
- **batch_size**: Frames to associate with audio weights  
- **fps**: Frame rate for the analysis  

**Parameters**:  
- **analysis_mode**: e.g., Drums Only, Vocals, Full Audio  
- **threshold**: Minimum weight pass-through  
- **multiply**: Amplification factor  

**Outputs**:
- **graph_audio** (image preview),  
- **processed_audio**, **original_audio**,  
- **audio_weights** (list of values).

</details>

---

### Load Audio Separation Model 🎧
Loads or downloads an audio separation model (e.g., HybridDemucs, OpenUnmix).  
<details>
  <summary><em>Node Parameters</em></summary>

- **model**: Choose between HybridDemucs / OpenUnmix.  
- **Outputs**: **audio_sep_model** (connect to Audio Analysis or Remixer).

</details>

---

### Audio Peaks Detection 📈
Identifies peaks in the audio weights to trigger transitions or events.  
<details>
  <summary><em>Node Parameters</em></summary>

- **peaks_threshold**: Sensitivity.  
- **min_peaks_distance**: Minimum gap in frames between peaks.  
- **Outputs**: Binary peak list, alternate list, peak indices/count, graph.

</details>

---

### Audio IP Adapter Transitions 🔄
Manages transitions between images based on peaks. Great for stable or style transitions.  
<details>
  <summary><em>Node Parameters</em></summary>

- **images**: Batch of images.  
- **peaks_weights**: From “Audio Peaks Detection”.  
- **blend_mode**, **transitions_length**, **min_IPA_weight**, etc.

</details>

---

### Audio Prompt Schedule 📝
Links text prompts to peak indices.  
<details>
  <summary><em>Node Parameters</em></summary>

- **peaks_index**: Indices from peaks detection.  
- **prompts**: multiline string.  
- **Outputs**: mapped schedule string.

</details>

---

### Audio Remixer 🎛️
Adjusts volume levels (drums, vocals, bass, others) in a track.  
<details>
  <summary><em>Node Parameters</em></summary>

- **drums_volume**, **vocals_volume**, **bass_volume**, **others_volume**  
- **Outputs**: single merged audio track.

</details>

---

### Repeat Image To Count 🔁
Repeats a set of images N times.  
<details>
  <summary><em>Node Parameters</em></summary>

- **mask**: Mask input.  
- **Outputs**: Repeated images.

</details>

---

### Invert Floats 🔄
Flips sign of float values.  
<details>
  <summary><em>Node Parameters</em></summary>

- **floats**: list of floats.  
- **Outputs**: inverted list.

</details>

---

### Floats Visualizer 📈
Plots float values as a graph.  
<details>
  <summary><em>Node Parameters</em></summary>

- **floats** (and optional second/third).  
- **Outputs**: visual graph image.

</details>

---

### Mask To Float 🎭
Converts a mask into a single float value.  
<details>
  <summary><em>Node Parameters</em></summary>

- **mask**: input.  
- **Outputs**: float.

</details>

---

### Floats To Weights Strategy 🏋️
Transforms float lists into an IPAdapter “weight strategy.”  
<details>
  <summary><em>Node Parameters</em></summary>

- **floats**: list of floats.  
- **Outputs**: dictionary with strategy info.

</details>

</details>

---

## 6. Thank You!

Please give a **⭐ on GitHub** it helps us enhance our Tool and it's Free for you !! (:  
