import torch
import folder_paths
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
import pandas as pd
from ... import Yvann
import random

class AudioNodeBase(Yvann):
    CATEGORY= "👁️ Yvann Nodes/🔊 Audio"
        
class Audio_Analysis_Yvann(AudioNodeBase):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 60, "step": 1}),
                "smoothing_factor": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "global_intensity": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "MASK", "IMAGE")
    RETURN_NAMES = ("Audio", "Audio Weights", "Audio Masks", "Weights Graph")
    FUNCTION = "process_audio"

    def _get_audio_frame(self, waveform, i, samples_per_frame):
        start = i * samples_per_frame
        end = start + samples_per_frame
        return waveform[..., start:end].cpu().numpy().squeeze()

    def _rms_energy(self, waveform, num_frames, samples_per_frame):
        return np.array([np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)) for i in range(num_frames)])

    def _smooth_weights(self, weights, smoothing_factor):
        if smoothing_factor <= 0.01:
            return weights
        kernel_size = max(3, int(smoothing_factor * 50))  # Ensure minimum kernel size of 3
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(weights, kernel, mode='same')

    def _normalize_weights(self, weights):
        min_val, max_val = np.min(weights), np.max(weights)
        if max_val > min_val:
            return (weights - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(weights)

    def adjust_weights(self, weights, global_intensity):
        factor = 1 + (global_intensity * 0.5)
        adjusted_weights = np.maximum(weights * factor, 0)
        adjusted_weights = np.round(adjusted_weights, 3)
        return adjusted_weights

    def generate_masks(self, input_values, width, height):
        # Ensure input_values is a list
        if isinstance(input_values, (float, int)):
            input_values = [input_values]
        elif isinstance(input_values, pd.Series):
            input_values = input_values.tolist()
        elif isinstance(input_values, list) and all(isinstance(item, list) for item in input_values):
            input_values = [item for sublist in input_values for item in sublist]

        # Generate a batch of masks based on the input_values
        masks = []
        for value in input_values:
            # Assuming value is a float between 0 and 1 representing the mask's intensity
            mask = torch.ones((height, width), dtype=torch.float32) * value
            masks.append(mask)
        masks_out = torch.stack(masks, dim=0)
    
        return masks_out

    def process_audio(self, audio, video_frames, frame_rate, smoothing_factor, global_intensity, prompt=None, filename_prefix="ComfyUI", extra_pnginfo=None):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        num_frames, height, width, _ = video_frames.shape

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0) 
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo if necessary

        total_samples = waveform.shape[-1]
        samples_per_frame = total_samples // num_frames

        processed_audio = {
            'waveform': waveform.cpu(),
            'sample_rate': sample_rate,
            'frame_rate': frame_rate
        }
        
        audio_weights = self._rms_energy(waveform, num_frames, samples_per_frame)
        audio_weights = self._smooth_weights(audio_weights, max(0.01, smoothing_factor))
        audio_weights = self._normalize_weights(audio_weights)
        audio_weights = [round(float(weight), 3) for weight in audio_weights]

        audio_weights = self.adjust_weights(np.array(audio_weights), global_intensity)

        # Plot the weights
        frames = list(range(1, num_frames + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(frames, audio_weights, label='Audio Weights', color='blue')
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Weights')
        plt.title('Normalized Weights for Audio (RMS Energy)')
        plt.legend()
        plt.grid(True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile, format='png')
            tmpfile_path = tmpfile.name
        plt.close()

        weights_graph = Image.open(tmpfile_path).convert("RGB")
        weights_graph = np.array(weights_graph)
        weights_graph = torch.tensor(weights_graph).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        weights_graph = weights_graph.permute(0, 2, 3, 1)

        # Generate masks from audio weights
        audio_masks = self.generate_masks(audio_weights, width, height)

        return (
            processed_audio,
            audio_weights.tolist(),
            audio_masks,
            weights_graph,
        )
