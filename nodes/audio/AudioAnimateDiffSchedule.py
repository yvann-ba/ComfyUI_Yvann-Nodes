from ... import Yvann
import numpy as np

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class AudioAnimateDiffSchedule(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_audio_weights": ("FLOAT", {"forceInput": True}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_range": ("FLOAT", {"default": 0.95, "min": 0.8, "max": 1.49, "step": 0.01}),
                "max_range": ("FLOAT", {"default": 1.25, "min": 0.81, "max": 1.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("processed_weights",)
    FUNCTION = "process_any_audio_weights"

    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

    def process_any_audio_weights(self, any_audio_weights, smooth, min_range, max_range):
        if not isinstance(any_audio_weights, (list, np.ndarray)):
            print("Invalid any_audio_weights input")
            return None

        any_audio_weights = np.array(any_audio_weights, dtype=np.float32)

        # Apply smoothing
        smoothed_signal = np.zeros_like(any_audio_weights)
        for i in range(len(any_audio_weights)):
            if i == 0:
                smoothed_signal[i] = any_audio_weights[i]
            else:
                smoothed_signal[i] = smoothed_signal[i-1] * smooth + any_audio_weights[i] * (1 - smooth)

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
        rounded_rescaled_signal = [round(elem, 6) for elem in rescaled_signal]

        return (rounded_rescaled_signal,)
