import torch
import os
import folder_paths
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
import librosa

class Audio_Reactive_IPAdapter_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT",),
                "weight_algorithm": (["rms_energy", "amplitude_envelope", "spectral_centroid", "onset_detection", "chroma_features"], {"default": "rms_energy"}),
                "smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "AUDIO", "FLOAT", "AUDIO", "FLOAT", "AUDIO", "FLOAT", "IMAGE")
    RETURN_NAMES = ("audio", "Audio Weights", "drums_audio", "Drums Weights", "vocals_audio", "Vocals Weights", "bass_audio", "Bass Weights", "Visual Weights Graph")
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

    def _amplitude_envelope(self, waveform, num_frames, samples_per_frame):
        return np.array([np.max(np.abs(self._get_audio_frame(waveform, i, samples_per_frame))) for i in range(num_frames)])

    def _rms_energy(self, waveform, num_frames, samples_per_frame):
        return np.array([np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)) for i in range(num_frames)])

    def _spectral_centroid(self, waveform, num_frames, samples_per_frame, sample_rate):
        return np.array([np.mean(librosa.feature.spectral_centroid(y=self._get_audio_frame(waveform, i, samples_per_frame), sr=sample_rate)[0]) for i in range(num_frames)])

    def _onset_detection(self, waveform, num_frames, samples_per_frame, sample_rate):
        return np.array([np.mean(librosa.onset.onset_strength(y=self._get_audio_frame(waveform, i, samples_per_frame), sr=sample_rate)) for i in range(num_frames)])

    def _chroma_features(self, waveform, num_frames, samples_per_frame, sample_rate):
        return np.array([np.mean(librosa.feature.chroma_stft(y=self._get_audio_frame(waveform, i, samples_per_frame), sr=sample_rate)) for i in range(num_frames)])

    def _smooth_weights(self, weights, smoothing_factor):
        kernel_size = max(3, int(smoothing_factor * 50))  # Ensure minimum kernel size of 3
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(weights, kernel, mode='same')

    def _normalize_weights(self, weights):
        min_val, max_val = np.min(weights), np.max(weights)
        if max_val > min_val:
            return (weights - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(weights)

    def process_audio(self, audio, video_frames, frame_rate, weight_algorithm, smoothing_factor):
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

        # Determine the device
        device = next(model.parameters()).device
        waveform = waveform.to(device)

        estimates = model(waveform)

        # Compute normalized audio weights for each frame
        total_samples = waveform.shape[-1]
        samples_per_frame = total_samples // num_frames

        # Création des isolated_audio avec normalisation
        isolated_audio = {}
        target_indices = {'drums': 1, 'vocals': 0, 'bass': 2}
        for target, index in target_indices.items():
            target_waveform = estimates[:, index, :, :]  # Shape: (1, 2, num_samples)
            
            # Normalisation du volume
            max_val = torch.max(torch.abs(target_waveform))
            if max_val > 0:
                target_waveform = target_waveform / max_val

            isolated_audio[target] = {
                'waveform': target_waveform.cpu(),  # Move back to CPU
                'sample_rate': sample_rate,
                'frame_rate': frame_rate
            }
        
        # Apply the selected weight algorithm
        weight_function = getattr(self, f"_{weight_algorithm}")
        if weight_algorithm in ['spectral_centroid', 'onset_detection', 'chroma_features']:
            audio_weights = weight_function(waveform.squeeze(0), num_frames, samples_per_frame, sample_rate)
        else:
            audio_weights = weight_function(waveform.squeeze(0), num_frames, samples_per_frame)

        # Apply smoothing to audio weights
        audio_weights = self._smooth_weights(audio_weights, smoothing_factor)
        audio_weights = self._normalize_weights(audio_weights)

        audio_weights = [round(float(weight), 3) for weight in audio_weights]

        # Calculate and normalize weights for each isolated audio target
        target_weights = {}
        for target, index in target_indices.items():
            target_waveform = isolated_audio[target]['waveform'].squeeze(0)
            if weight_algorithm in ['spectral_centroid', 'onset_detection', 'chroma_features']:
                target_weights[target] = weight_function(target_waveform, num_frames, samples_per_frame, sample_rate)
            else:
                target_weights[target] = weight_function(target_waveform, num_frames, samples_per_frame)
            
            # Apply smoothing to target weights
            target_weights[target] = self._smooth_weights(target_weights[target], smoothing_factor)
            target_weights[target] = self._normalize_weights(target_weights[target])
            
            target_weights[target] = [round(float(weight), 3) for weight in target_weights[target]]

        # Plot the weights
        frames = list(range(1, num_frames + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(frames, audio_weights, label='Audio Weights', color='black')
        plt.plot(frames, target_weights['drums'], label='Drums Weights', color='red')
        plt.plot(frames, target_weights['vocals'], label='Vocals Weights', color='green')
        plt.plot(frames, target_weights['bass'], label='Bass Weights', color='blue')
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Weights')
        plt.title(f'Normalized Weights for Audio Components ({weight_algorithm})')
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
            audio_weights,
            isolated_audio['drums'],
            target_weights['drums'],
            isolated_audio['vocals'],
            target_weights['vocals'],
            isolated_audio['bass'],
            target_weights['bass'],
            weights_graph
        )
