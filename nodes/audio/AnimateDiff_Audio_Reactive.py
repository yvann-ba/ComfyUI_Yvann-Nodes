from ... import Yvann
import numpy as np

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class AnimateDiff_Audio_Reactive(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_signal": ("FLOAT", {"forceInput": True}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_range": ("FLOAT", {"default": 0.9, "min": 0.7, "max": 1.4}),
                "max_range": ("FLOAT", {"default": 1.2, "min": 0.8, "max": 1.5}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("processed_weights",)
    FUNCTION = "process_audio_signal"

    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

    def process_audio_signal(self, audio_signal, smooth, min_range, max_range):
        if not isinstance(audio_signal, (list, np.ndarray)):
            print("Invalid audio_signal input")
            return None

        audio_signal = np.array(audio_signal, dtype=np.float32)

        # Apply smoothing
        smoothed_signal = np.zeros_like(audio_signal)
        for i in range(len(audio_signal)):
            if i == 0:
                smoothed_signal[i] = audio_signal[i]
            else:
                smoothed_signal[i] = smoothed_signal[i-1] * smooth + audio_signal[i] * (1 - smooth)

        # Normalize the smoothed signal
        min_val = np.min(smoothed_signal)
        max_val = np.max(smoothed_signal)
        if max_val - min_val != 0:
            normalized_signal = (smoothed_signal - min_val) / (max_val - min_val)
        else:
            normalized_signal = smoothed_signal - min_val  # All values are the same

        # Rescale to specified range
        rescaled_signal = normalized_signal * (max_range - min_range) + min_range
        rescaled_signal.tolist()
        rounded_rescaled_signal = [round(elem, 3) for elem in rescaled_signal]

        return (rounded_rescaled_signal,)
