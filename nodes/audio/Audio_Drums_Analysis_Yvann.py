import torch
import os
import folder_paths
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY= "👁️ Yvann Nodes/🔊 Audio"

class Audio_Drums_Analysis_Yvann(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "smoothing_factor": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "global_intensity": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "MASK", "IMAGE")
    RETURN_NAMES = ("Drums Audio", "Drums Weights", "Drums Masks", "Weights Graph")
    FUNCTION = "process_audio"

    def download_and_load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        download_path = os.path.join(folder_paths.models_dir, "openunmix")
        os.makedirs(download_path, exist_ok=True)

        model_file = "umxl.pth"
        model_path = os.path.join(download_path, model_file)

        if not os.path.exists(model_path):
            print("Downloading umxhq model...")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device='cpu')
            torch.save(separator.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device='cpu')
            separator.load_state_dict(torch.load(model_path, map_location='cpu'))

        separator = separator.to(device)
        separator.eval()

        return separator

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
        elif isinstance(input_values, list) and all(isinstance(item, list) for item in input_values):
            input_values = [item for sublist in input_values for item in sublist]

        # Generate a batch of masks based on the input_values
        masks = []
        for value in input_values:
            mask = torch.ones((height, width), dtype=torch.float32) * value
            masks.append(mask)
        masks_out = torch.stack(masks, dim=0)
    
        return masks_out

    def process_audio(self, audio, video_frames, smoothing_factor, global_intensity):
        model = self.download_and_load_model()
        
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        num_frames, height, width, _ = video_frames.shape

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0) 
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo if necessary
            
        waveform = waveform.unsqueeze(0)    

        device = next(model.parameters()).device
        waveform = waveform.to(device)

        estimates = model(waveform)

        total_samples = waveform.shape[-1]
        samples_per_frame = total_samples // num_frames

        drums_waveform = estimates[:, 1, :, :]  # Shape: (1, 2, num_samples)
        drums_audio = {
            'waveform': drums_waveform.cpu(),  # Move back to CPU
            'sample_rate': sample_rate,
        }
        
        drums_weights = self._rms_energy(drums_waveform.squeeze(0), num_frames, samples_per_frame)
        drums_weights = self._smooth_weights(drums_weights, max(0.01, smoothing_factor))
        drums_weights = self._normalize_weights(drums_weights)
        drums_weights = [round(float(weight), 3) for weight in drums_weights]

        drums_weights = self.adjust_weights(np.array(drums_weights), global_intensity)

        # Generate masks from drums weights
        drums_masks = self.generate_masks(drums_weights, width, height)

        # Plot the weights
        frames = list(range(1, num_frames + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(frames, drums_weights, label='Drums Weights', color='red')
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Weights')
        plt.title('Normalized Weights for Drums (RMS Energy)')
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

        return (
            drums_audio,
            drums_weights.tolist(),
            drums_masks,
            weights_graph
        )
