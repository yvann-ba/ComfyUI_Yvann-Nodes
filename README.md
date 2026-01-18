#### `pleaseee star⭐️ this repo, so I can keep improving it ((:`
---
# 🔊 ComfyUI Yvann Nodes [![Contact Me](https://img.shields.io/badge/Contact_Me-white?style=for-the-badge&logo=linkedin&logoColor=0A66C2)](https://www.linkedin.com/in/yvann-barbot/)


### **A pack of custom nodes that enable audio reactivity within [ComfyUI](https://github.com/comfyanonymous/ComfyUI), allowing you to generate AI-driven animations that sync with music**

---

## What Does This Do?

- **Create** Audio Reactive AI videos, enable controls over AI generations styles, content and composition with any audio
- **Simple**: Just Drop one of our [Workflows](/example_workflows) in ComfyUI and specify your audio and visuals input
- **Flexible**: Works with existing ComfyUI AI tech and nodes (eg: IPAdapter, AnimateDiff, ControlNet, etc.)

---

## Quick Setup

### 1. Install [ComfyUI](https://www.comfy.org/download) (Works on Mac, Windows, Linux)

### 2. Pick a Workflow ⬇️

---

#### 🖼️ Images to Video
Takes a **set of images** + **audio**

**Results:**
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/615394cd-c829-4ee0-94de-1ffd20d35b9d" width="100%" controls autoplay loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/d9a630d5-cd13-4cf4-a1da-282e6078cd49" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/cd6af84d-db51-47cc-897d-c880271f2971" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/f4b64874-5ca9-49ea-8d2c-40b377a5b5bd" width="100%" controls autoplay loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/8e75df33-6426-4d6e-98d1-f8288cc87b74" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9d179485-011d-4de2-a4fb-d8489f20a2cf" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=O2s6NseXlMc">
        <img src="https://img.youtube.com/vi/O2s6NseXlMc/maxresdefault.jpg" width="100%">
      </a>
    </td>
    <td width="50%" valign="middle">
      <h3>📺 <a href="https://www.youtube.com/watch?v=O2s6NseXlMc">Watch Tutorial</a></h3>
    </td>
  </tr>
</table>

📥 [Download ImagesToVideo Workflow](example_workflows/AudioReactive_ImagesToVideo_Yvann.json)

---

#### 🎬 Video to Video
Takes a **source video** + **audio**

**Results:**
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/c0450100-a61f-4707-9e14-0d4ca563a2b1" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/c0fa2ca0-6c0f-4687-b1c9-fe531278c58e" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=BiQHWKP3q0c">
        <img src="https://img.youtube.com/vi/BiQHWKP3q0c/maxresdefault.jpg" width="100%">
      </a>
    </td>
    <td width="50%" valign="middle">
      <h3>📺 <a href="https://www.youtube.com/watch?v=BiQHWKP3q0c">Watch Tutorial</a></h3>
    </td>
  </tr>
</table>

📥 [Download VideoToVideo Workflow](example_workflows/AudioReactive_VideoToVideo_Yvann.json)

---

#### ✍️ Text to Video
Takes a **text prompt** + **audio**

**Results:**
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;"> 
  <tr>
     <td>
          <video src="https://github.com/user-attachments/assets/bb2b2299-5423-4034-b7e5-121a3df7eb1a" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d5bc5607-d242-4c50-aadc-9ad313a80104" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

📥 [Download TextToVideo Workflow](example_workflows/AudioReactive_TextToVideo_Yvann.json)

---

### 3. Load & Run

1. **Drop** the downloaded `.json` file into the ComfyUI window
2. **Click "Install All"** on the popup
3. **Set your inputs** (images/video/text + audio)
4. **Download the models** listed on the note
5. **Click Queue** to generate your audio-reactive animation!

**That's it!** Have fun !!

**That's it!** Have fun !!
---

## Nodes Details

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
- **peaks_weights**: From "Audio Peaks Detection".  
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
Transforms float lists into an IPAdapter "weight strategy."  
<details>
  <summary><em>Node Parameters</em></summary>

- **floats**: list of floats.  
- **Outputs**: dictionary with strategy info.

</details>

</details>

---
<h3 align="center">
  Please give a ⭐ on GitHub it helps us enhance our Tool and it's Free !! (:
</h3>

#### Made with the help of [Lilien](https://www.linkedin.com/in/lilien-auger/) 😎

