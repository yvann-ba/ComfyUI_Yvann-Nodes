from .nodes.audio.AudioAnalysis_YVANN import AudioAnalysis_YVANN
from .nodes.audio.AudioFrequencyAnalysis_YVANN import AudioFrequencyAnalysis_YVANN
from .nodes.text.ShowText_YVANN import ShowText_YVANN

NODE_CLASS_MAPPINGS = {
    "Audio Analysis | YVANN": AudioAnalysis_YVANN,
    "Audio Frequency Analysis | YVANN": AudioFrequencyAnalysis_YVANN,
    "Show Text | YVANN": ShowText_YVANN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalysis_YVANN": "Audio Analysis | YVANN",
    "AudioFrequencyAnalysis_YVANN": "Audio Frequency Analysis | YVANN",
    "ShowText_YVANN": "Show Text YVANN"
}

__all__ = ['NODE_CLASS_MAPPINGS']