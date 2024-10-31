import torch
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tempfile
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict

from ... import Yvann
import comfy.model_management as mm
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Audio_Analysis(AudioNodeBase):
    analysis_modes = ["Drums Only", "Full Audio", "Vocals Only", "Bass Only", "Other Audio"]



    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
                "audio": ("AUDIO", {"forceInput": True}),
                "analysis_mode": (cls.analysis_modes,),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.01}),
                "multiply": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "AUDIO", "FLOAT",)
    RETURN_NAMES = ("graph_audio", "processed_audio", "original_audio", "audio_weights")
    FUNCTION = "process_audio"



    def download_and_load_model(self):
        device = mm.get_torch_device()
        try:
            print("Loading Hybrid Demucs model...")
            bundle = HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model().to(device)
            model.sample_rate = bundle.sample_rate
            model.sources = ['bass', 'drums', 'other', 'vocals']
            model.eval()
            print("Hybrid Demucs model loaded successfully")
        except Exception as e:
            print(f"Error in loading Hybrid Demucs model: {e}")
            raise RuntimeError("Error in loading Hybrid Demucs model")
        return model



    def _get_audio_frame(self, waveform: torch.Tensor, i: int, samples_per_frame: int) -> np.ndarray:
        start = i * samples_per_frame
        end = start + samples_per_frame
        return waveform[..., start:end].cpu().numpy().squeeze()



    def _rms_energy(self, waveform: torch.Tensor, batch_size: int, samples_per_frame: int) -> np.ndarray:
        try:
            rms_values = []
            for i in range(batch_size):
                frame = self._get_audio_frame(waveform, i, samples_per_frame)
                if frame.size == 0:
                    rms = 0.0
                else:
                    rms = np.sqrt(np.mean(frame ** 2))
                    rms = round(rms, 4)
                rms_values.append(rms)
            return np.array(rms_values)
        except Exception as e:
            print(f"Error in RMS energy calculation: {e}")
            return np.zeros(batch_size)



    def process_audio(self, audio: Dict[str, torch.Tensor], batch_size: int, fps: float, analysis_mode: str, threshold: float, multiply: float,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[float]]:
        
        if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
            raise ValueError("Invalid audio input")

        # Get and normalize initial waveform
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        print(f"Initial waveform shape after normalization: {waveform.shape}")

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
        if analysis_mode != "Full Audio":
            try:
                model = self.download_and_load_model()
                device = next(model.parameters()).device
                waveform = waveform.to(device)
                input_sample_rate = sample_rate
                model_sample_rate = model.sample_rate

                # Resample if necessary
                if input_sample_rate != model_sample_rate:
                    resample = torchaudio.transforms.Resample(
                        orig_freq=input_sample_rate,
                        new_freq=model_sample_rate
                    ).to(device)
                    waveform = resample(waveform)

                model_input = waveform
                print(f"Model input shape: {model_input.shape}")

                with torch.no_grad():
                    estimates = model(model_input)
                print(f"Model output shape: {estimates.shape}")

                #inverted because otherwise the output of bass was drums, quick fix
                source_name_mapping = {
                    "Bass Only": "drums",
                    "Drums Only": "bass",
                    "Other Audio": "other",
                    "Vocals Only": "vocals"
                }

                source_name = source_name_mapping.get(analysis_mode)
                if source_name is not None:
                    source_index = model.sources.index(source_name)
                    # Extract correct source and ensure 2D shape
                    processed_waveform = estimates[:, source_index]
                    processed_waveform = processed_waveform
                else:
                    raise ValueError(f"Analysis mode '{analysis_mode}' is not valid.")
            except Exception as e:
                print(f"Error in model processing: {e}")
                raise
        else:  # Full Audio
            processed_waveform = waveform.clone()

        original_waveform = waveform

        # Convert to Mono by averaging channels
        #processed_waveform = processed_waveform.mean(dim=0, keepdim=True)
        #original_waveform = original_waveform.mean(dim=0, keepdim=True)

        # Store the sample rate used for processed audio
        final_sample_rate = model_sample_rate if 'model_sample_rate' in locals() else sample_rate


        # Prepare audio outputs with explicit shape checks
        processed_audio = {
            'waveform': processed_waveform.cpu().detach(),
            'sample_rate': final_sample_rate
        }
        print(f"Processed audio output shape: {processed_audio['waveform'].shape}")

        original_audio = {
            'waveform': original_waveform.cpu().detach(),
            'sample_rate': final_sample_rate
        }
        print(f"Original audio output shape: {original_audio['waveform'].shape}")


        # Calculate audio weights using mean across channels if multi-channel
        waveform_for_rms = processed_waveform.squeeze(0).squeeze(0)
        audio_weights = self._rms_energy(waveform_for_rms, batch_size, samples_per_frame)

        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            raise ValueError("Invalid audio weights calculated")


        # Normalize and process weights
        min_weight = np.min(audio_weights)
        max_weight = np.max(audio_weights)
        if max_weight - min_weight != 0:
            audio_weights_normalized = (audio_weights - min_weight) / (max_weight - min_weight)
        else:
            audio_weights_normalized = audio_weights - min_weight

        audio_weights_thresholded = np.where(audio_weights_normalized > threshold, audio_weights_normalized, 0)
        audio_weights_processed = np.clip(audio_weights_thresholded * multiply, 0, 1)


        # Generate visualization
        try:
            figsize = 12.0
            plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
            plt.plot(
                list(range(0, len(audio_weights_processed))),
                audio_weights_processed,
                label=f'{analysis_mode} Weights',
                color='blue'
            )

            plt.xlabel('Frame Number')
            plt.ylabel('Weights')
            plt.title(f'Processed Audio Weights ({analysis_mode})')
            plt.legend()
            plt.grid(True)

            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile.name, format='png')
                tmpfile_path = tmpfile.name
            plt.close()

            weights_graph = Image.open(tmpfile_path).convert("RGB")
            weights_graph = np.array(weights_graph).astype(np.float32) / 255.0
            weights_graph = torch.from_numpy(weights_graph).permute(2, 0, 1).unsqueeze(0).permute(0, 2, 3, 1)
        except Exception as e:
            print(f"Error in creating weights graph: {e}")
            weights_graph = torch.zeros((1, 400, 300, 3))

        rounded_audio_weights = [round(float(x), 3) for x in audio_weights_processed]
        
        return (weights_graph, processed_audio, original_audio, rounded_audio_weights)
