import torch
import os
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Audio_Reactive_Switch_Yvann(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_weights": ("FLOAT", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distance": ("INT", {"default": 1, "min": 1, "step": 1}),
                "prominence": ("FLOAT", {"default": 0.1, "min": 0.0,  "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT", "IMAGE")
    RETURN_NAMES = ("indices", "visualization")
    FUNCTION = "process_weights"

    def process_weights(self, audio_weights, threshold, distance, prominence):
        if not isinstance(audio_weights, list) and not isinstance(audio_weights, np.ndarray):
            print("Invalid audio_weights input")
            return None, None

        weights = np.array(audio_weights)

        # Normalize weights
        weights_normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) > 0 else weights

        # Detect peaks
        peaks, _ = find_peaks(weights_normalized, height=threshold, distance=distance, prominence=prominence)

        # Generate indices
        indices = []
        index_value = 0
        peak_set = set(peaks)
        for i in range(len(weights_normalized)):
            if i in peak_set:
                index_value += 1
            indices.append(index_value)

        # Generate visualization
        try:
            plt.figure(figsize=(10, 6), facecolor='white')
            plt.plot(range(len(weights_normalized)), weights_normalized, label='Normalized Weights', color='blue', alpha=0.5)
            plt.scatter(peaks, weights_normalized[peaks], color='red', label='Detected Peaks')
            plt.step(range(len(indices)), np.array(indices)/max(indices if max(indices) > 0 else 1), where='post', label='Indices (Normalized)', color='green')
            plt.xlabel('Frame Number')
            plt.ylabel('Normalized Weights / Indices')
            plt.title('Audio Weights and Detected Peaks')
            plt.legend()
            plt.grid(True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile, format='png')
                tmpfile_path = tmpfile.name
            plt.close()

            visualization = Image.open(tmpfile_path).convert("RGB")
            visualization = np.array(visualization)
            visualization = torch.tensor(visualization).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            visualization = visualization.permute(0, 2, 3, 1)
        except Exception as e:
            print(f"Error in creating visualization: {e}")
            visualization = None

        return indices, visualization
