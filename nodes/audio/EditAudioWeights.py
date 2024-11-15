from ... import Yvann
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tempfile
from PIL import Image
import torch

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class EditAudioWeights(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_audio_weights": ("FLOAT", {"forceInput": True}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_range": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.99, "step": 0.01}),
                "max_range": ("FLOAT", {"default": 1, "min": 0.01, "max": 3, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("process_weights", "graph_audio")
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

        rounded_rescaled_signal = [round(float(elem), 6) for elem in rescaled_signal]

        # Plot the rescaled signal
        try:
            figsize = 12.0
            plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
            plt.plot(
                list(range(len(rounded_rescaled_signal))),
                rounded_rescaled_signal,
                label='Processed Weights',
                color='blue'
            )
            plt.xlabel('Frames')
            plt.ylabel('Weights')
            plt.title('Processed Audio Weights')
            plt.legend()
            plt.grid(True)

            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile.name, format='png', bbox_inches='tight')
                tmpfile_path = tmpfile.name
            plt.close()

            weights_graph = Image.open(tmpfile_path).convert("RGB")
            weights_graph = np.array(weights_graph).astype(np.float32) / 255.0
            weights_graph = torch.from_numpy(weights_graph).unsqueeze(0)
        except Exception as e:
            print(f"Error in creating weights graph: {e}")
            weights_graph = torch.zeros((1, 400, 300, 3))

        return (rounded_rescaled_signal, weights_graph)
