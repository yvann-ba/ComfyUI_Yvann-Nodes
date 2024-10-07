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


class Audio_Reactive_Yvann(AudioNodeBase):
    analysis_modes = ["Drums Only", "Full Audio",
                      "Vocals Only", "Bass Only", "Other Audio"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
                "audio": ("AUDIO", {"forceInput": True}),
                "analysis_mode": (cls.analysis_modes,),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.01}),
                "add": ("FLOAT", {"default": 0.0, "min": -1, "max": 1, "step": 0.01}),
                "smooth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "multiply": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
                "weights_range": ("FLOAT", {"default": 0, "min": 0, "max": 3, "step": 0.1}),
                "invert_weights": ("BOOLEAN", {"default": False}),
            }
        }
        
    RETURN_TYPES = ("FLOAT", "AUDIO", "AUDIO", "IMAGE")
    RETURN_NAMES = ("audio_weights", "processed_audio", "original_audio", "audio_visualization")
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
        # Extract a single frame of audio from the waveform
        start = i * samples_per_frame
        end = start + samples_per_frame
        return waveform[..., start:end].cpu().numpy().squeeze()

    def _rms_energy(self, waveform, batch_size, samples_per_frame):
        # Calculate the RMS energy for each audio frame
        try:
            return np.array([round(np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)), 4) for i in range(batch_size)])
        except Exception as e:
            print(f"Error in RMS energy calculation: {e}")
            return np.zeros(batch_size)

    def _apply_audio_processing(self, weights, threshold, add, smooth, multiply):
        # Normalize weights to 0-1 range
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) > 0 else weights

        # Apply threshold
        weights = np.where(weights > threshold, weights, 0)

        # Apply Add
        weights = np.clip(weights + add, 0, (1))

        # Apply smoothing
        smoothed = np.zeros_like(weights)
        for i in range(len(weights)):
            if i == 0:
                smoothed[i] = weights[i]
            else:
                smoothed[i] = smoothed[i-1] * smooth + weights[i] * (1 - smooth)

        # Apply final multiplication
        smoothed = smoothed * multiply

        return smoothed

    def _apply_threshold(self, weights, threshold):
        # Apply threshold to weights with normalization
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) > 0 else weights

        thresholded = np.where(weights > threshold,(weights - threshold) / (1 - threshold),0)
        return thresholded

    def process_audio(self, audio, batch_size, fps, analysis_mode, threshold, add, smooth, multiply, weights_range, invert_weights):
        # Main function to process audio and generate weights
        if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
            print("Invalid audio input")
            return None, None, None, None

        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        if not isinstance(waveform, torch.Tensor):
            print("Waveform is not a torch.Tensor")
            return None, None, None, None

        # Calculate total audio duration needed
        audio_duration = batch_size / fps
        total_samples_needed = int(audio_duration * sample_rate)

        # Crop or pad audio to match required duration
        if waveform.shape[-1] > total_samples_needed:
            waveform = waveform[..., :total_samples_needed]
        elif waveform.shape[-1] < total_samples_needed:
            pad_length = total_samples_needed - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        samples_per_frame = total_samples_needed // batch_size

        # Apply audio separation if needed
        if analysis_mode in ["Drums Only", "Vocals Only", "Bass Only", "Other Audio"]:
            try:
                model = self.download_and_load_model()
                device = next(model.parameters()).device
                waveform = waveform.to(device)
                estimates = model(waveform)
                if analysis_mode == "Drums Only":
                    processed_waveform = estimates[:, 1, :, :]
                elif analysis_mode == "Vocals Only":
                    processed_waveform = estimates[:, 0, :, :]
                elif analysis_mode == "Bass Only":
                    processed_waveform = estimates[:, 2, :, :]
                elif analysis_mode == "Other Audio":
                    processed_waveform = estimates[:, 3, :, :]
            except Exception as e:
                print(f"Error in model processing: {e}")
                return None, None, None, None
        else:  # audio (full mix)
            processed_waveform = waveform

        processed_audio = {
            'waveform': processed_waveform.cpu(),
            'sample_rate': sample_rate,
        }
        original_audio = {
            'waveform': waveform.cpu(),
            'sample_rate': sample_rate,
        }

        # Calculate audio weights
        audio_weights = self._rms_energy(
            processed_waveform.squeeze(0), batch_size, samples_per_frame)
        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            print("Invalid audio weights calculated")
            return None, None, None, None

        audio_weights = self._apply_audio_processing(
            audio_weights, threshold, add, smooth, multiply)

        audio_weights = np.clip(audio_weights, 0, 1)
        scale_audio_weights = audio_weights + weights_range
        scale_audio_weights = np.round(scale_audio_weights, 3)


        if (invert_weights == True):
            audio_weights_inverted = 1.0 - np.array(audio_weights)
            audio_weights_inverted = np.clip(audio_weights_inverted, 0, 1)
            scale_audio_weights_inverted = audio_weights_inverted + weights_range
            scale_audio_weights_inverted = np.round(scale_audio_weights_inverted, 3)

        # Generate visualization
        try:
            plt.figure(figsize=(10, 6), facecolor='white')
            if (invert_weights == False):
                plt.plot(list(range(1, len(scale_audio_weights) + 1)), scale_audio_weights, 
                         label=f'{analysis_mode.capitalize()} Weights', color='blue')

            if (invert_weights == True):
                plt.plot(list(range(1, len(scale_audio_weights_inverted) + 1)), scale_audio_weights_inverted,
                         label='Inverted Weights', color='red', linestyle='--')

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
            return None, None, None, None

        if (invert_weights == True):
            scale_audio_weights = scale_audio_weights_inverted

        if processed_audio is None or scale_audio_weights is None or weights_graph is None:
            print("One or more outputs are invalid")
            return None, None, None, None
        scale_audio_weights = scale_audio_weights.tolist()
        rounded_scale_audio_weights = [round(elem, 3) for elem in scale_audio_weights]
        
        return (rounded_scale_audio_weights, processed_audio, original_audio, weights_graph)
