import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Import for integer x-axis labels
import tempfile
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class AudioPeaksDetection(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_weights": ("FLOAT", {"forceInput": True}),
                "peaks_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_peaks_distance": ("INT", {"default": 1, "min": 1,})
            }
        }

    RETURN_TYPES = ("FLOAT",  "FLOAT", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("peaks_weights", "peaks_alternate_weights", "peaks_index", "peaks_count", "graph_peaks")
    FUNCTION = "detect_peaks"

    def detect_peaks(self, audio_weights, peaks_threshold, min_peaks_distance):
        if not isinstance(audio_weights, (list, np.ndarray)):
            print("Invalid audio_weights input")
            return None, None

        audio_weights = np.array(audio_weights, dtype=np.float32)

        peaks, _ = find_peaks(audio_weights, height=peaks_threshold, distance=min_peaks_distance)

        # Generate binary peaks array: 1 for peaks, 0 for non-peaks
        peaks_binary = np.zeros_like(audio_weights, dtype=int)
        peaks_binary[peaks] = 1
        
        actual_value = 0
        peaks_alternate = np.zeros_like(peaks_binary)
        for i in range (len(peaks_binary)):
            if peaks_binary[i] == 1:
                actual_value = 1 - actual_value
            peaks_alternate[i] = actual_value

        audio_peaks_index = np.array(peaks, dtype=int)
        audio_peaks_index = np.insert(audio_peaks_index, 0, 0)

        peaks_count = len(audio_peaks_index)
        str_peaks_index = ', '.join(map(str, audio_peaks_index))
        # Generate visualization
        try:
            figsize = 12.0
            plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
            plt.plot(range(0, len(audio_weights)), audio_weights, label='Audio Weights', color='blue', alpha=0.5)
            plt.scatter(peaks, audio_weights[peaks], color='red', label='Detected Peaks')

            plt.xlabel('Frame Number')
            plt.ylabel('Audio Weights')
            plt.title('Audio Weights and Detected Peaks')
            plt.legend()
            plt.grid(True)

            # Ensure x-axis labels are integers
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile.name, format='png')
                tmpfile_path = tmpfile.name
            plt.close()

            visualization = Image.open(tmpfile_path).convert("RGB")
            visualization = np.array(visualization).astype(np.float32) / 255.0
            visualization = torch.from_numpy(visualization).unsqueeze(0)  # Shape: [1, H, W, C]

        except Exception as e:
            print(f"Error in creating visualization: {e}")
            visualization = None

        return (peaks_binary.tolist(), peaks_alternate.tolist(), str_peaks_index, peaks_count, visualization)
