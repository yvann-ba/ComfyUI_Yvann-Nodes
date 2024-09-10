import torch
import os
import folder_paths
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
from ... import Yvann


class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"


class AudioToVid_Audio_Reactivity_Yvann(AudioNodeBase):
    analysis_modes = ["audio", "drums only", "vocals only"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Audio": ("AUDIO",),
                "Analysis Mode": (cls.analysis_modes,),
                "Threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.6, "step": 0.01}),
                "Gain": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "Add": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "Smooth": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "AUDIO", "IMAGE")
    RETURN_NAMES = ("Audio Weights", "Processed Audio", "Audio Visualization")
    FUNCTION = "process_audio"

    def process_audio(self, audio, analysis_mode, threshold, gain, add, smooth):
        if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
            print("Invalid audio input")
            return None, None, None

        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        # Définir un taux d'analyse fixe (par exemple, 30 segments par seconde)
        analysis_rate = 30
        audio_duration = waveform.shape[-1] / sample_rate
        num_segments = int(audio_duration * analysis_rate)

        # Traitement de l'audio selon le mode d'analyse
        if analysis_mode != "audio":
            processed_waveform = self._apply_separation(waveform, analysis_mode)
        else:
            processed_waveform = waveform

        processed_audio = {
            'waveform': processed_waveform.cpu(),
            'sample_rate': sample_rate,
        }
        
        audio_weights = self._rms_energy(processed_waveform.squeeze(0), num_segments, waveform.shape[-1] // num_segments)
        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            print("Invalid audio weights calculated")
            return None, None, None

        audio_weights = self._apply_audio_processing(audio_weights, threshold, gain, add, smooth)

        visualization = self._create_visualization(audio_weights, analysis_mode)

        return audio_weights.tolist(), processed_audio, visualization

    def _apply_audio_processing(self, weights, threshold, gain, add, smooth):
        # Normalize weights to 0-1 range
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        
        # Apply threshold
        weights = np.where(weights > threshold, weights, 0)
        
        # Apply gain
        effective_gain = (6 - gain)  # Inverse la plage de 1-5 à 5-1
        weights = np.power(weights, effective_gain)
        
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

    def _apply_separation(self, waveform, mode):
        # Implémentez ici la séparation audio pour "drums only" ou "vocals only"
        # Utilisez le modèle de séparation que vous avez déjà
        pass

    def _rms_energy(self, waveform, num_segments, samples_per_segment):
        # Calculez les poids RMS pour chaque segment
        pass

    def _create_visualization(self, weights, mode):
        # Créez et retournez une visualisation des poids audio
        pass
