import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import io
from PIL import Image
import tempfile
import numpy as np

# Function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter to data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to extract low, mid, and high frequency bands from the waveform
def extract_frequency_bands(waveform, sample_rate):
    low_waveform = butter_bandpass_filter(waveform, 20, 250, sample_rate)
    mid_waveform = butter_bandpass_filter(waveform, 250, 4000, sample_rate)
    high_waveform = butter_bandpass_filter(waveform, 4000, 20000, sample_rate)
    return low_waveform, mid_waveform, high_waveform

class AudioFrequencyAnalysis_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
                "low_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mid_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "high_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("Low Weights Schedule", "Mid Weights Schedule", "High Weights Schedule", "Visual Frequency Weights Graph")
    FUNCTION = "compute_weights"

    def compute_weights(self, audio, video_frames, frame_rate, low_threshold, mid_threshold, high_threshold):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        num_frames, _, _, _ = video_frames.shape

        # Ensure waveform has the correct dimensions
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)

        # Extract low, mid, and high frequency bands
        low_waveform, mid_waveform, high_waveform = extract_frequency_bands(
            waveform.cpu().numpy(), sample_rate)
        low_waveform = torch.tensor(low_waveform)
        mid_waveform = torch.tensor(mid_waveform)
        high_waveform = torch.tensor(high_waveform)

        total_samples = waveform.shape[-1]
        samples_per_frame = total_samples // num_frames

        # Function to compute normalized weights for each frame with threshold
        def compute_weights(waveform, threshold):
            weights = []
            for i in range(num_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                frame_waveform = waveform[..., start:end]
                frame_energy = torch.sqrt(
                    torch.mean(frame_waveform ** 2)).item()
                weights.append(frame_energy)
            max_weight = max(weights)
            if max_weight > 0:
                weights = [weight / max_weight for weight in weights]
            weights = [weight if weight >= threshold else 0 for weight in weights]
            return weights

        # Compute weights for low, mid, and high frequency bands with thresholds
        low_weights = compute_weights(low_waveform, low_threshold)
        mid_weights = compute_weights(mid_waveform, mid_threshold)
        high_weights = compute_weights(high_waveform, high_threshold)

        # Format weights as strings with frame index starting from 1
        low_weights_str = [
            f"\"{i+1}:{round(weight, 2)}\"" for i, weight in enumerate(low_weights)]
        mid_weights_str = [
            f"\"{i+1}:{round(weight, 2)}\"" for i, weight in enumerate(mid_weights)]
        high_weights_str = [
            f"\"{i+1}:{round(weight, 2)}\"" for i, weight in enumerate(high_weights)]

        # Plot the weights
        frames = list(range(1, num_frames + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(frames, low_weights, label='Low Weights', color='blue')
        plt.plot(frames, mid_weights, label='Mid Weights', color='green')
        plt.plot(frames, high_weights, label='High Weights', color='red')
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Weights')
        plt.title('Normalized Weights for Low, Mid, and High Frequency Bands')
        plt.legend()
        plt.grid(True)
# Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile, format='png')
            tmpfile_path = tmpfile.name
        plt.close()

        # Load the image from the temporary file and convert to tensor
        weights_graph = Image.open(tmpfile_path).convert("RGB")
        weights_graph = np.array(weights_graph)
        weights_graph = torch.tensor(weights_graph).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Ensure the tensor has the correct shape [B, H, W, C]
        weights_graph = weights_graph.permute(0, 2, 3, 1)

        return ", ".join(low_weights_str), ", ".join(mid_weights_str), ", ".join(high_weights_str), weights_graph
