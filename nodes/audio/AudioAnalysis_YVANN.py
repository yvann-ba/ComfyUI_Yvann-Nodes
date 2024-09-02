import torch
import os
import folder_paths
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_frequency_bands(waveform, sample_rate):
    low_waveform = butter_bandpass_filter(waveform, 20, 250, sample_rate)
    mid_waveform = butter_bandpass_filter(waveform, 250, 4000, sample_rate)
    high_waveform = butter_bandpass_filter(waveform, 4000, 20000, sample_rate)
    return low_waveform, mid_waveform, high_waveform

class AudioAnalysis_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["umxl", "umxhq"], {"default": "umxl"}),
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            }
		}

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "drums_audio", "vocals_audio", "bass_audio", "other_audio", "mid_audio", "low_audio", "high_audio", "audio_weights_str", "drums_weights_str", "vocals_weights_str", "bass_weights_str", "other_weights_str", "mid_weights_str", "low_weights_str", "high_weights_str")
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

        # Extract frequency bands
        low_waveform, mid_waveform, high_waveform = extract_frequency_bands(waveform.cpu().numpy(), sample_rate)
        low_waveform = torch.tensor(low_waveform).to(device)
        mid_waveform = torch.tensor(mid_waveform).to(device)
        high_waveform = torch.tensor(high_waveform).to(device)

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

        mid_weights = compute_weights(mid_waveform)
        low_weights = compute_weights(low_waveform)
        high_weights = compute_weights(high_waveform)

        mid_weights_str = [f"\"{i}:{round(weight, 2)}\"" for i, weight in enumerate(mid_weights)]
        low_weights_str = [f"\"{i}:{round(weight, 2)}\"" for i, weight in enumerate(low_weights)]
        high_weights_str = [f"\"{i}:{round(weight, 2)}\"" for i, weight in enumerate(high_weights)]

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
            audio_weights_str = [f"\"{i}:{round(weight / max_weight, 2)}\"" for i, weight in enumerate(audio_weights)]

        # Calculate and normalize weights for each isolated audio target
        target_weights_str = {}
        for target, index in target_indices.items():
            target_waveform = isolated_audio[target]['waveform']
            target_weights = []
            for i in range(num_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                frame_waveform = target_waveform[..., start:end]
                frame_energy = torch.sqrt(torch.mean(frame_waveform ** 2)).item()
                target_weights.append(frame_energy)
            
            max_target_weight = max(target_weights)
            if max_target_weight > 0:
                target_weights_str[target] = [f"\"{i}:{round(weight / max_target_weight, 2)}\"" for i, weight in enumerate(target_weights)]

        # Extract mid, low, and high frequencies from the input audio
        mid_audio = {'waveform': [], 'sample_rate': sample_rate, 'frame_rate': frame_rate}
        low_audio = {'waveform': [], 'sample_rate': sample_rate, 'frame_rate': frame_rate}
        high_audio = {'waveform': [], 'sample_rate': sample_rate, 'frame_rate': frame_rate}
        
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            frame_waveform = waveform[..., start:end]
            
            # Append the frame waveform to the respective frequency band audio
            mid_audio['waveform'].append(mid_waveform[..., start:end].cpu())
            low_audio['waveform'].append(low_waveform[..., start:end].cpu())
            high_audio['waveform'].append(high_waveform[..., start:end].cpu())

        # Convert lists to tensors
        mid_audio['waveform'] = torch.cat(mid_audio['waveform'], dim=-1)
        low_audio['waveform'] = torch.cat(low_audio['waveform'], dim=-1)
        high_audio['waveform'] = torch.cat(high_audio['waveform'], dim=-1)

        return (
            audio,
            isolated_audio['drums'],
            isolated_audio['vocals'],
            isolated_audio['bass'],
            isolated_audio['other'],
            mid_audio,
            low_audio,
            high_audio,
            ", ".join(audio_weights_str),
            ", ".join(target_weights_str['drums']),
            ", ".join(target_weights_str['vocals']),
            ", ".join(target_weights_str['bass']),
            ", ".join(target_weights_str['other']),
            ", ".join(mid_weights_str),
            ", ".join(low_weights_str),
            ", ".join(high_weights_str)
        )
