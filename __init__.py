from .nodes.audio.Audio_Reactive_IPAdapter_YVANN import Audio_Reactive_IPAdapter_YVANN
from .nodes.audio.AudioAnalysis_Advanced_YVANN import AudioAnalysis_Advanced_YVANN
from .nodes.audio.AudioFrequencyAnalysis_YVANN import AudioFrequencyAnalysis_YVANN

NODE_CLASS_MAPPINGS = {
    "Audio Reactive IPAdapter | YVANN": Audio_Reactive_IPAdapter_YVANN,
    "Audio Analysis Advanced | YVANN": AudioAnalysis_Advanced_YVANN,
    "Audio Frequency Analysis | YVANN": AudioFrequencyAnalysis_YVANN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Reactive IPAdapter | YVANN": "Audio Reactive IPAdapter | YVANN",
    "AudioAnalysis_Advanced_YVANN": "Audio Analysis Advanced | YVANN",
   "AudioFrequencyAnalysis_YVANN": "Audio Frequency Analysis | YVANN",
}

__all__ = ['NODE_CLASS_MAPPINGS']