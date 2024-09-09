import torch
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY= "👁️ Yvann Nodes/🔊 Audio"
        
class Audio_Analysis_Yvann(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.6, "step": 0.01}),
                "gain": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 7.0, "step": 0.1}),
                "add": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "smooth": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        try:
            return np.array([np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)) for i in range(num_frames)])
        except Exception as e:
            print(f"Error in RMS energy(Audio Analysis algorithm) calculation: {e}")
            return np.zeros(num_frames)

    def _apply_audio_processing(self, weights, threshold, gain, add, smooth):
        # Normalize weights to 0-1 range
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        
        # Apply threshold
        weights = np.where(weights > threshold, weights, 0)
        
        # Apply gain (with increased effect)
        weights = np.power(weights, 1 / gain)  # This will boost lower values more aggressively
        
        # Apply add
        weights = np.clip(weights + add, 0, 1)
        
        # Apply smooth
        smoothed = np.zeros_like(weights)
        for i in range(len(weights)):
            if i == 0:
                smoothed[i] = weights[i]
            else:
                smoothed[i] = smoothed[i-1] * smooth + weights[i] * (1 - smooth)
        
        return smoothed

    def generate_masks(self, input_values, width, height):
        # Ensure input_values is a list
        if isinstance(input_values, (float, int)):
            input_values = [input_values]
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

    def process_audio(self, audio, video_frames, threshold, gain, add, smooth):
        if audio is None:
            print("No audio provided")
            return (None, None, None, None,)
        
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        if video_frames is None or not isinstance(video_frames, torch.Tensor) or video_frames.dim() != 4:
            print("Invalid video frames input")
            return (None, None, None, None,)
        
        if not isinstance(waveform, torch.Tensor):
            print("Waveform is not a torch.Tensor")
            return (None, None, None, None,)
        
        num_frames, height, width, _ = video_frames.shape

        # Ensure waveform is 3D (batch, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        if total_samples < num_frames:
            print("Audio is shorter than video")
            return None, None, None, None

        samples_per_frame = total_samples // num_frames

        audio_weights = self._rms_energy(waveform, num_frames, samples_per_frame)
        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            print("Invalid audio weights calculated")
            return None, None, None, None

        # Apply new audio processing
        audio_weights = self._apply_audio_processing(audio_weights, threshold, gain, add, smooth)

        # Plot the weights
        frames = list(range(1, num_frames + 1))
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(frames, audio_weights, label='Audio Weights', color='blue')
            plt.xlabel('Frame Number')
            plt.ylabel('Normalized Weights')
            plt.title('Processed Audio Weights')
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
        except Exception as e:
            print(f"Error in creating weights graph: {e}")
            weights_graph = None

        # Generate masks from audio weights
        audio_masks = self.generate_masks(audio_weights, width, height)

        if audio is None or audio_weights is None or audio_masks is None or weights_graph is None:
            print("One or more outputs are invalid")
            return None, None, None, None

        return audio, audio_weights.tolist(), audio_masks, weights_graph
