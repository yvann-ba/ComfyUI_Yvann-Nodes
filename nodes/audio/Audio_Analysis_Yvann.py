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


class Audio_Analysis_Yvann(AudioNodeBase):
    analysis_modes = ["audio", "drums only", "vocals only"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "analysis_mode": (cls.analysis_modes,),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.6, "step": 0.01}),
                "gain": ("FLOAT", {"default": 3.5, "min": 0.01, "max": 7.0, "step": 0.01}),
                "add": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "smooth": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "MASK", "IMAGE")
    RETURN_NAMES = ("Processed Audio", "Audio Weights",
                    "Audio Masks", "Weights Graph")
    FUNCTION = "process_audio"

    def download_and_load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        download_path = os.path.join(folder_paths.models_dir, "openunmix")
        os.makedirs(download_path, exist_ok=True)

        model_file = "umxl.pth"
        model_path = os.path.join(download_path, model_file)

        if not os.path.exists(model_path):
            print("Downloading umxl model...")
            separator = torch.hub.load(
                'sigsep/open-unmix-pytorch', 'umxl', device='cpu')
            torch.save(separator.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
            separator = torch.hub.load(
                'sigsep/open-unmix-pytorch', 'umxl', device='cpu')
            separator.load_state_dict(
                torch.load(model_path, map_location='cpu'))

        separator = separator.to(device)
        separator.eval()

        return separator

    def _get_audio_frame(self, waveform, i, samples_per_frame):
        start = i * samples_per_frame
        end = start + samples_per_frame
        return waveform[..., start:end].cpu().numpy().squeeze()

    def _rms_energy(self, waveform, num_frames, samples_per_frame):
        try:
            return np.array([np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)) for i in range(num_frames)])
        except Exception as e:
            print(f"Error in RMS energy calculation: {e}")
            return np.zeros(num_frames)

    def _apply_audio_processing(self, weights, threshold, gain, add, smooth):
        # Normalize weights to 0-1 range
        weights = (weights - np.min(weights)) / \
            (np.max(weights) - np.min(weights))

        # Apply threshold
        weights = np.where(weights > threshold, weights, 0)

        # Apply gain (with safety check for zero or near-zero values)
        if gain > 0.01:  # Avoid division by very small numbers
            weights = np.power(weights, 1 / gain)
        elif gain < -0.01:  # Handle negative gain
            weights = 1 - np.power(1 - weights, 1 / abs(gain))
        else:  # For gain very close to zero, don't modify weights
            pass  # weights remain unchanged

        # Apply add
        weights = np.clip(weights + add, 0, 1)

        # Apply smooth
        smoothed = np.zeros_like(weights)
        for i in range(len(weights)):
            if i == 0:
                smoothed[i] = weights[i]
            else:
                smoothed[i] = smoothed[i-1] * \
                    smooth + weights[i] * (1 - smooth)

        return smoothed

    def _apply_threshold(weights, threshold):
        # Normalisation
        weights = (weights - np.min(weights)) / \
            (np.max(weights) - np.min(weights))

        # Application du threshold avec une fonction de transfert
        return np.where(weights > threshold,
                        (weights - threshold) / (1 - threshold),
                        0)

    def generate_masks(self, input_values, width, height):
        if isinstance(input_values, (float, int)):
            input_values = [input_values]
        elif isinstance(input_values, list) and all(isinstance(item, list) for item in input_values):
            input_values = [
                item for sublist in input_values for item in sublist]

        masks = []
        for value in input_values:
            mask = torch.ones((height, width), dtype=torch.float32) * value
            masks.append(mask)
        masks_out = torch.stack(masks, dim=0)

        return masks_out

    def process_audio(self, audio, video_frames, analysis_mode, threshold, gain, add, smooth):
        if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
            print("Invalid audio input")
            return None, None, None, None

        if video_frames is None or not isinstance(video_frames, torch.Tensor) or video_frames.dim() != 4:
            print("Invalid video frames input")
            return None, None, None, None

        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        if not isinstance(waveform, torch.Tensor):
            print("Waveform is not a torch.Tensor")
            return None, None, None, None

        num_frames, height, width, _ = video_frames.shape

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)

        waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        if total_samples < num_frames:
            print("Audio is shorter than video")
            return None, None, None, None

        samples_per_frame = total_samples // num_frames

        if analysis_mode in ["drums only", "vocals only"]:
            try:
                model = self.download_and_load_model()
                device = next(model.parameters()).device
                waveform = waveform.to(device)
                estimates = model(waveform)
                if analysis_mode == "drums only":
                    processed_waveform = estimates[:, 1, :, :]
                else:  # vocals only
                    processed_waveform = estimates[:, 0, :, :]
            except Exception as e:
                print(f"Error in model processing: {e}")
                return None, None, None, None
        else:  # audio (full mix)
            processed_waveform = waveform

        processed_audio = {
            'waveform': processed_waveform.cpu(),
            'sample_rate': sample_rate,
        }

        audio_weights = self._rms_energy(
            processed_waveform.squeeze(0), num_frames, samples_per_frame)
        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            print("Invalid audio weights calculated")
            return None, None, None, None

        audio_weights = self._apply_audio_processing(
            audio_weights, threshold, gain, add, smooth)

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(1, num_frames + 1)), audio_weights,
                     label=f'{analysis_mode.capitalize()} Weights', color='blue')
            plt.xlabel('Frame Number')
            plt.ylabel('Normalized Weights')
            plt.title(f'Processed Audio Weights ({analysis_mode.capitalize()})')
            plt.legend()
            plt.grid(True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile, format='png')
                tmpfile_path = tmpfile.name
            plt.close()

            weights_graph = Image.open(tmpfile_path).convert("RGB")
            weights_graph = np.array(weights_graph)
            weights_graph = torch.tensor(weights_graph).permute(
                2, 0, 1).unsqueeze(0).float() / 255.0
            weights_graph = weights_graph.permute(0, 2, 3, 1)
        except Exception as e:
            print(f"Error in creating weights graph: {e}")
            weights_graph = None

        audio_masks = self.generate_masks(audio_weights, width, height)

        if processed_audio is None or audio_weights is None or audio_masks is None or weights_graph is None:
            print("One or more outputs are invalid")
            return None, None, None, None

        return processed_audio, audio_weights.tolist(), audio_masks, weights_graph
