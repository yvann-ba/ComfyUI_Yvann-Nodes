import numpy as np
import librosa

class AudioAnalysis_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "mid_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mid_values", "audio")
    FUNCTION = "analyze_audio"
    CATEGORY = "Audio/Analysis"

    def analyze_audio(self, audio_path, frame_rate, mid_threshold):
        # Load audio using librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate the mid-range frequencies (e.g., 300-5000 Hz)
        mid_frequencies = librosa.effects.harmonic(y)
        
        # Calculate the mid values based on the threshold
        mid_values = mid_frequencies[mid_frequencies > mid_threshold]

        # Convert the mid values to a string for output
        mid_values_str = ",".join(map(str, mid_values.tolist()))

        return (mid_values_str, audio_path)









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