from .nodes.audio.AudioAnalysis_YVANN import AudioAnalysis_YVANN
from .nodes.audio.AudioAnalysis_Advanced_YVANN import AudioAnalysis_Advanced_YVANN
from .nodes.audio.AudioFrequencyAnalysis_YVANN import AudioFrequencyAnalysis_YVANN

NODE_CLASS_MAPPINGS = {
    "Audio Analysis | YVANN": AudioAnalysis_YVANN,
    "Audio Analysis Advanced | YVANN": AudioAnalysis_Advanced_YVANN,
    "Audio Frequency Analysis | YVANN": AudioFrequencyAnalysis_YVANN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalysis_YVANN": "Audio Analysis | YVANN",
    "AudioAnalysis_Advanced_YVANN": "Audio Analysis Advanced | YVANN",
   "AudioFrequencyAnalysis_YVANN": "Audio Frequency Analysis | YVANN",
}

__all__ = ['NODE_CLASS_MAPPINGS']