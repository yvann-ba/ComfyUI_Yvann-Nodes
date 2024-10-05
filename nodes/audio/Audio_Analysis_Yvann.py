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
                "num_frames": ("INT", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
                "audio": ("AUDIO",),
                "analysis_mode": (cls.analysis_modes,),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.6, "step": 0.01}),
                "gain": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "add": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "smooth": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "multiply_by": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "AUDIO", "AUDIO", "IMAGE")
    RETURN_NAMES = ("audio_weights", "audio_weights_inverted", "processed_audio", "original_audio", "audio_visualization")
    FUNCTION = "process_audio"

    def download_and_load_model(self):
        # Download and load the OpenUnmix model for audio separation
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

    def _rms_energy(self, waveform, num_frames, samples_per_frame):
        # Calculate the RMS energy for each audio frame
        try:
            return np.array([round(np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)), 4) for i in range(num_frames)])
        except Exception as e:
            print(f"Error in RMS energy calculation: {e}")
            return np.zeros(num_frames)

    def _apply_audio_processing(self, weights, threshold, gain, add, smooth, multiply_by):
        # Normalize weights to 0-1 range
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) > 0 else weights
        weights = np.round(weights, 4)

        # Apply threshold
        weights = np.where(weights > threshold, weights, 0)
        weights = np.round(weights, 4)

        # Apply gain
        if gain > 0.01:
            weights = np.power(weights, 1 / gain)
        elif gain < -0.01:
            weights = 1 - np.power(1 - weights, 1 / abs(gain))
        weights = np.round(weights, 4)
        
        # Apply addition and clip to 0-1.25 range
        weights = np.clip(weights + add, 0, 1.25)
        weights = np.round(weights, 4)

        # Apply smoothing
        smoothed = np.zeros_like(weights)
        for i in range(len(weights)):
            if i == 0:
                smoothed[i] = weights[i]
            else:
                smoothed[i] = smoothed[i-1] * smooth + weights[i] * (1 - smooth)
        smoothed = np.round(smoothed, 4)

        # Apply final multiplication
        smoothed = smoothed * multiply_by
        smoothed = np.round(smoothed, 4)

        return smoothed

    def _apply_threshold(self, weights, threshold):
        # Apply threshold to weights with normalization
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) > 0 else weights
        weights = np.round(weights, 4)

        thresholded = np.where(weights > threshold,
                        (weights - threshold) / (1 - threshold),
                        0)
        return np.round(thresholded, 4)

    def generate_masks(self, input_values, width, height):
        # Generate compressed masks from input values
        if isinstance(input_values, (float, int)):
            input_values = [round(input_values, 4)]
        elif isinstance(input_values, list) and all(isinstance(item, list) for item in input_values):
            input_values = [round(item, 4) for sublist in input_values for item in sublist]
        else:
            input_values = [round(item, 4) for item in input_values]

        # Compress the mask resolution
        compressed_width = max(32, width // 8)  # Minimum width of 32 pixels
        compressed_height = max(32, height // 8)  # Minimum height of 32 pixels

        masks = []
        for value in input_values:
            # Create a small mask
            small_mask = torch.ones((compressed_height, compressed_width), dtype=torch.float32) * value
            # Resize the mask to original dimensions
            mask = torch.nn.functional.interpolate(small_mask.unsqueeze(0).unsqueeze(0), 
                                                   size=(height, width), 
                                                   mode='nearest').squeeze(0).squeeze(0)
            masks.append(mask)
        masks_out = torch.stack(masks, dim=0)

        return masks_out

    def process_audio(self, audio, num_frames, fps, analysis_mode, threshold, gain, add, smooth, multiply_by):
        # Main function to process audio and generate weights and masks
        
        # Input validation
        if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
            print("Invalid audio input")
            return None, None, None, None

        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        if not isinstance(waveform, torch.Tensor):
            print("Waveform is not a torch.Tensor")
            return None, None, None, None

        # Calculate total audio duration needed
        audio_duration = num_frames / fps
        total_samples_needed = int(audio_duration * sample_rate)

        # Crop or pad audio to match required duration
        if waveform.shape[-1] > total_samples_needed:
            waveform = waveform[..., :total_samples_needed]
        elif waveform.shape[-1] < total_samples_needed:
            pad_length = total_samples_needed - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Calculate samples per frame
        samples_per_frame = total_samples_needed // num_frames

        # Apply audio separation if needed
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
        original_audio = {
            'waveform': waveform.cpu(),
            'sample_rate': sample_rate,
        }

        # Calculate audio weights
        audio_weights = self._rms_energy(
            processed_waveform.squeeze(0), num_frames, samples_per_frame)
        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            print("Invalid audio weights calculated")
            return None, None, None, None

        # Apply audio processing
        audio_weights = self._apply_audio_processing(
            audio_weights, threshold, gain, add, smooth, multiply_by)

        # Ensure audio_weights are within [0, 1] range
        audio_weights = np.clip(audio_weights, 0, 1)

        # Calculate inverted weights
        audio_weights_inverted = 1.0 - np.array(audio_weights)
        
        # Ensure inverted weights are also within [0, 1] range
        audio_weights_inverted = np.clip(audio_weights_inverted, 0, 1)

        # Generate visualization
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(1, len(audio_weights) + 1)), audio_weights,
                     label=f'{analysis_mode.capitalize()} Weights', color='blue')
            plt.plot(list(range(1, len(audio_weights_inverted) + 1)), audio_weights_inverted,
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

        # Final validation
        if processed_audio is None or audio_weights is None or weights_graph is None:
            print("One or more outputs are invalid")
            return None, None, None, None

        return (audio_weights.tolist(), audio_weights_inverted.tolist(), processed_audio, original_audio, weights_graph)