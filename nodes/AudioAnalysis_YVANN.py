import librosa
import torch
import os
import folder_paths


class DownloadOpenUnmixModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["umxl", "umxhq"], {"default": "umxl"}),
            }
        }

    RETURN_TYPES = ("OPEN_UNMIX_MODEL",)
    FUNCTION = "download_and_load_model"
    CATEGORY = "RyanOnTheInside/Audio"

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

        return (separator,)

class AudioAnalysis_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OPEN_UNMIX_MODEL",),
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            }
		}

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "drums_audio", "vocals_audio", "bass_audio", "other_audio", "audio_weights")
    FUNCTION = "process_audio"

    def process_audio(self, model, audio, video_frames, frame_rate):
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
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            frame_waveform = waveform[..., start:end]
            frame_energy = torch.sqrt(torch.mean(frame_waveform ** 2)).item()
            audio_weights.append(frame_energy)

        # Normalize the audio weights to be between 0 and 1
        max_weight = max(audio_weights)
        if max_weight > 0:
            audio_weights = [weight / max_weight for weight in audio_weights]

        return (
            audio,
            isolated_audio['drums'],
            isolated_audio['vocals'],
            isolated_audio['bass'],
            isolated_audio['other'],
            audio_weights
        )









# import numpy as np
# import librosa
# from PIL import Image, ImageDraw

# class AudioAnalysis_YVANN:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "audio": ("AUDIO", {"forceInput": True}),
#                 "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 100.0, "step": 1.0}),
#                 "high_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "kick_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "mid_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "low_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "snare_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
#             }
#         }

#     RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE")
#     RETURN_NAMES = ("high_values", "kick_values", "mid_values", "low_values", "snare_values", "visualization")
#     FUNCTION = "analyze_audio"
#     CATEGORY = "Audio/Analysis"

#     def analyze_audio(self, audio, frame_rate, high_threshold, kick_threshold, mid_threshold, low_threshold, snare_threshold):
#         waveform = audio['waveform'].numpy()
#         sr = audio['sample_rate']

#         # Calculer la longueur de saut (hop_length) pour correspondre au frame_rate désiré
#         hop_length = int(sr / frame_rate)

#         # Calculer le spectrogramme
#         n_fft = 2048  # Choisir une taille de fenêtre appropriée
#         D = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
#         S = np.abs(D)

#         # Définir les bandes de fréquences
#         freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#         num_bins = S.shape[0]
#         high_band = (freqs > 5000)[:num_bins]
#         mid_band = ((freqs > 500) & (freqs <= 5000))[:num_bins]
#         low_band = (freqs <= 500)[:num_bins]

#         # Extraire l'énergie pour chaque bande
#         high_energy = np.sum(S[high_band, :], axis=0)
#         mid_energy = np.sum(S[mid_band, :], axis=0)
#         low_energy = np.sum(S[low_band, :], axis=0)

#         # Normaliser les énergies
#         def normalize(x):
#             return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)

#         high_values = normalize(high_energy)
#         mid_values = normalize(mid_energy)
#         low_values = normalize(low_energy)

#         # Détecter les kicks et snares
#         onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
#         onset_env_normalized = normalize(onset_env)

#         kick_values = np.where(onset_env_normalized > kick_threshold, onset_env_normalized, 0.0)
#         snare_values = np.where(onset_env_normalized > snare_threshold, onset_env_normalized, 0.0)


#         def values_to_text(values):
#             # Convertir en liste pour s'assurer que nous manipulons des scalaires
#             values_list = values.tolist() if isinstance(values, np.ndarray) else values
#             return "\n".join([f"{i}: {value:.2f}" for i, value in enumerate(values_list)])


#         return (
#             values_to_text(high_values),
#             values_to_text(kick_values),
#             values_to_text(mid_values),
#             values_to_text(low_values),
#             values_to_text(snare_values)
#         )