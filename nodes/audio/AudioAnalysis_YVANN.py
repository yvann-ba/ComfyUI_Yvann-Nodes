import torch
import os
import folder_paths
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image

class AudioAnalysis_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["umxhq", "umxl"], {"default": "umxhq"}),
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            }
		}

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("audio", "drums_audio", "vocals_audio", "bass_audio", "other_audio", "audio_weights_str", "drums_weights_str", "vocals_weights_str", "bass_weights_str", "other_weights_str", "Visual Weights Graph")
    FUNCTION = "process_audio"

    def download_and_load_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        download_path = os.path.join(folder_paths.models_dir, "openunmix")
        os.makedirs(download_path, exist_ok=True)

        model_file = f"{model_name}.pth"
        model_path = os.path.join(download_path, model_file)

        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model...")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', model_name, device='cpu')
            torch.save(separator.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', model_name, device='cpu')
            separator.load_state_dict(torch.load(model_path, map_location='cpu'))

        separator = separator.to(device)
        separator.eval()

        return separator

    def process_audio(self, model_name, audio, video_frames, frame_rate):
        model = self.download_and_load_model(model_name)
        
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

        # Determine the device
        device = next(model.parameters()).device
        waveform = waveform.to(device)

        estimates = model(waveform)

        # Compute normalized audio weights for each frame
        total_samples = waveform.shape[-1]
        samples_per_frame = total_samples // num_frames

        def compute_weights(waveform):
            weights = []
            for i in range(num_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                frame_waveform = waveform[..., start:end]
                frame_energy = torch.sqrt(torch.mean(frame_waveform ** 2)).item()
                weights.append(frame_energy)
            max_weight = max(weights)
            if max_weight > 0:
                weights = [weight / max_weight for weight in weights]
            return weights

        # Create isolated audio objects for each target
        isolated_audio = {}
        target_indices = {'drums': 1, 'vocals': 0, 'bass': 2, 'other': 3}  # Corrected indices
        for target, index in target_indices.items():
            target_waveform = estimates[:, index, :, :]  # Shape: (1, 2, num_samples)
            
            isolated_audio[target] = {
                'waveform': target_waveform.cpu(),  # Move back to CPU
                'sample_rate': sample_rate,
                'frame_rate': frame_rate
            }
        
        # Compute normalized audio weights for each frame
        total_samples = waveform.shape[-1]
        samples_per_frame = total_samples // num_frames

        # Calculate the energy of the waveform in each frame
        audio_weights = []
        audio_weights_str = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            frame_waveform = waveform[..., start:end]
            frame_energy = torch.sqrt(torch.mean(frame_waveform ** 2)).item()
            audio_weights.append(frame_energy)
            audio_weights_str.append(f"\"{i}:{round(frame_energy, 2)}\"")

        # Normalize the audio weights to be between 0 and 1
        max_weight = max(audio_weights)
        if max_weight > 0:
            audio_weights = [weight / max_weight for weight in audio_weights]
            audio_weights_str = [f"\"{i}:{round(weight, 2)}\"" for i, weight in enumerate(audio_weights)]

        # Calculate and normalize weights for each isolated audio target
        target_weights = {}
        target_weights_str = {}
        for target, index in target_indices.items():
            target_waveform = isolated_audio[target]['waveform']
            target_weights[target] = []
            for i in range(num_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                frame_waveform = target_waveform[..., start:end]
                frame_energy = torch.sqrt(torch.mean(frame_waveform ** 2)).item()
                target_weights[target].append(frame_energy)
            
            max_target_weight = max(target_weights[target])
            if max_target_weight > 0:
                target_weights[target] = [weight / max_target_weight for weight in target_weights[target]]
                target_weights_str[target] = [f"\"{i}:{round(weight, 2)}\"" for i, weight in enumerate(target_weights[target])]

        # Plot the weights
        frames = list(range(1, num_frames + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(frames, audio_weights, label='Audio Weights', color='black')
        plt.plot(frames, target_weights['drums'], label='Drums Weights', color='red')
        plt.plot(frames, target_weights['vocals'], label='Vocals Weights', color='green')
        plt.plot(frames, target_weights['bass'], label='Bass Weights', color='blue')
        plt.plot(frames, target_weights['other'], label='Other Weights', color='orange')
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Weights')
        plt.title('Normalized Weights for Audio Components')
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

        return (
            audio,
            isolated_audio['drums'],
            isolated_audio['vocals'],
            isolated_audio['bass'],
            isolated_audio['other'],
            ", ".join(audio_weights_str),
            ", ".join(target_weights_str['drums']),
            ", ".join(target_weights_str['vocals']),
            ", ".join(target_weights_str['bass']),
            ", ".join(target_weights_str['other']),
            weights_graph
        )
