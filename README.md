# ComfyUI_Yvann-Nodes 

[![Instagram](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=instagram&logoColor=E4405F)](https://instagram.com/yvann.mp4)
[![YouTube](https://img.shields.io/badge/yvann.mp4-white?style=for-the-badge&logo=youtube&logoColor=FF0000)](https://www.youtube.com/channel/yvann.mp4)



### With this pack, you can analyze audio, extract drum and vocal tracks, and use this information to create audio-reactive animations in your ComfyUI workflows

![Usage Example](./docs/example.gif)

## Audio Nodes

### Audio Analysis

This node analyzes the overall audio and generates weights based on RMS energy.

![Audio Analysis](./docs/audio_analysis.png)

### Audio Drums Analysis

Extracts and analyzes the drum track from the audio.

![Audio Drums Analysis](./docs/audio_drums_analysis.png)

### Audio Vocals Analysis

Extracts and analyzes the vocal track from the audio.

![Audio Vocals Analysis](./docs/audio_vocals_analysis.png)

Each analysis node provides:
- Audio weights
- Audio masks
- Weights graph

---

## Installation
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Launch ComfyUI
3. Click on "🧩 Manager" -> "Custom Nodes Manager"
4. Search for `ComfyUI_Yvann-Nodes` in the manager and install it

---

## Usage

Use these nodes like any other node in ComfyUI. You'll find them in the "Audio Nodes" category.

1. Connect your audio file to the input of the desired analysis node.
2. Use the outputs (weights, masks, graph) to influence other parts of your workflow.
3. Create audio-reactive animations by linking these outputs to image or video parameters.

---


## Contributing 🙌  
Want to help make this project better? Feel free to:

- Open an issue
- Submit a pull request
- Reach out to me on [LinkedIn](https://www.linkedin.com/in/yvann-barbot/) or via email at [barbot.yvann@gmail.com](mailto:barbot.yvann@gmail.com)
