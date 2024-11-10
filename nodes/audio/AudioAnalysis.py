import torch
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
from termcolor import colored
from torchaudio.transforms import Fade, Resample

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class AudioAnalysis(AudioNodeBase):
    analysis_modes = ["Drums Only", "Full Audio", "Vocals Only", "Bass Only", "Others Audio"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_separation_model": ("AUDIO_SEPARATION_MODEL", {"forceInput": True}),
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
            model.sources = ['bass', 'drums', 'others', 'vocals']
            model.eval()
            print("Hybrid Demucs model loaded successfully")
        except Exception as e:
            print(f"Error in loading Hybrid Demucs model: {e}")
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

    def prepare_audio_and_device(self, audio: Dict[str, torch.Tensor]) -> Tuple[torch.device, torch.Tensor]:
        """Prepares the device (GPU or CPU) and sets up the audio waveform."""
        device = mm.get_torch_device()
        waveform = audio['waveform'].squeeze(0).to(device)
        self.audio_sample_rate = audio['sample_rate']
        return device, waveform

    def apply_model_and_extract_sources(self, model, waveform: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, list[str]]:
        """Applies the model and extracts audio sources, handling both Open-Unmix and GDemucs cases."""
        sources, sources_list = None, []
        

        if isinstance(model, torch.nn.Module):  # Open-Unmix model
            print(colored("Applying Open_Unmix model on audio", 'green'))
            self.model_sample_rate = model.sample_rate

            if self.audio_sample_rate != self.model_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate).to(device)
                waveform = resampler(waveform)
            sources = model(waveform.unsqueeze(0)).squeeze(0)
            sources_list = ['bass', 'drums', 'others', 'vocals']

        elif hasattr(model, "get_model"):  # GDemucs model
            print(colored("Applying GDemucs model on audio", 'green'))
            self.model_sample_rate = model.sample_rate
            model = model.get_model().to(device)

            if self.audio_sample_rate != self.model_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate).to(device)
                waveform = resampler(waveform)
            ref = waveform.mean(0)
            waveform = (waveform - ref.mean()) / ref.std()
            sources = self.separate_sources(model, waveform[None], segment=10.0, overlap=0.1, device=device)[0]
            sources = sources * ref.std() + ref.mean()
            
            sources_list = getattr(model, 'sources', ['bass', 'drums', 'others', 'vocals'])

        else:
            print(colored("Unrecognized model type", 'red'))
            return None, []

        return sources, sources_list

    def separate_sources(self, model, mix, segment=10.0, overlap=0.1, device=None,
    ):
        """
        Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

        Args:
            segment (int): segment length in seconds
            device (torch.device, str, or None): if provided, device on which to
                execute the computation, otherwise `mix.device` is assumed.
                When `device` is different from `mix.device`, only local computations will
                be on `device`, while the entire tracks will be stored on `mix.device`.
        """
        if device is None:
            device = mix.device
        else:
            device = torch.device(device)

        batch, channels, length = mix.shape
        sample_rate = self.model_sample_rate

        chunk_len = int(sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = overlap * sample_rate
        fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final

    def adjust_waveform_dimensions(self, waveform):

        # Ensure waveform is at least 2D (channels, frames)
        if waveform.ndim == 1:
            # If waveform is 1D (frames,), add channel dimension
            waveform = waveform.unsqueeze(0)

        # Ensure waveform has batch dimension
        if waveform.ndim == 2:
            # Add batch dimension
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            # Waveform already has batch dimension
            pass
        else:
            raise ValueError(f"Waveform has unexpected dimensions: {waveform.shape}")

        return waveform


    def process_audio(self, audio_separation_model, audio: Dict[str, torch.Tensor], batch_size: int, fps: float, analysis_mode: str, threshold: float, multiply: float,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[float]]:
        if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
            raise ValueError("Invalid audio input")

        model = audio_separation_model
        waveform = audio['waveform']
        self.audio_waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        original_sample_rate = audio['sample_rate']
        self.audio_sample_rate = original_sample_rate

        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate
        if batch_size == 0:
            batch_size = int(audio_duration * fps)
        else:
            audio_duration = batch_size / fps
        total_samples_needed = int(audio_duration * sample_rate)


        samples_per_frame = total_samples_needed // batch_size



        """
        here we do that
        
        """
        if analysis_mode != "Full Audio":
            try:
                device, waveform = self.prepare_audio_and_device(audio)

                with torch.no_grad():
                    estimates, estimates_list = self.apply_model_and_extract_sources(model, waveform, device)

                if isinstance(model, torch.nn.Module):
                    model.sources = ['bass', 'drums', 'others', 'vocals']
                elif hasattr(model, "get_model"):
                    estimates_list = ['drums', 'bass', 'others', 'vocals']

                if isinstance(model, torch.nn.Module):
                    source_name_mapping = {
                        "Others Audio": "vocals",
                        "Bass Only": "others",
                        "Drums Only": "drums",
                        "Vocals Only": "bass"
                    }
                elif hasattr(model, "get_model"):
                    source_name_mapping = {
                        "Drums Only": "drums",
                        "Bass Only": "bass",
                        "Others Audio": "others",
                        "Vocals Only": "vocals"
                    }
    
                source_name = source_name_mapping.get(analysis_mode)
                if source_name is not None:
                    try:
                        source_index = estimates_list.index(source_name)
                        processed_waveform = estimates[source_index]
                        print(colored("Checking sources in processed_waveform:", 'blue'))
                    except ValueError:
                        raise ValueError(f"Source '{source_name}' is not available in the model's provided sources.")
                else:
                    raise ValueError(f"Analysis mode '{analysis_mode}' is invalid.")
            except Exception as e:
                print(f"Error in model processing: {e}")
                raise
        else:
            processed_waveform = waveform.clone()

        #--------------------------------------------------#
        #--------------------------------------------------#
        #--------------------------------------------------#


        if waveform.shape[-1] > total_samples_needed:
            waveform = waveform[..., :total_samples_needed]
        elif waveform.shape[-1] < total_samples_needed:
            pad_length = total_samples_needed - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # if processed_waveform.shape[-1] > total_samples_needed:
        #     processed_waveform = processed_waveform[..., :total_samples_needed]
        # elif processed_waveform.shape[-1] < total_samples_needed:
        #     pad_length = total_samples_needed - processed_waveform.shape[-1]
        #     processed_waveform = torch.nn.functional.pad(processed_waveform, (0, pad_length))

        processed_waveform = self.adjust_waveform_dimensions(processed_waveform)
        original_waveform = self.adjust_waveform_dimensions(waveform.clone())


        final_sample_rate = self.model_sample_rate if hasattr(self, 'model_sample_rate') else sample_rate
        if (analysis_mode != "Full Audio"):

            
            print(f"Resampling processed audio from {final_sample_rate} Hz back to original sample rate {original_sample_rate} Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=final_sample_rate, new_freq=original_sample_rate).to(processed_waveform.device)
            processed_waveform = processed_waveform.squeeze(0)
            processed_waveform = resampler(processed_waveform)
            processed_waveform = processed_waveform.unsqueeze(0)
            
        expected_num_samples = original_waveform.shape[-1]
        actual_num_samples = processed_waveform.shape[-1]
        original_audio_samples = original_waveform.shape[-1]
        if actual_num_samples > expected_num_samples:
            processed_waveform = processed_waveform[..., :expected_num_samples]
        elif actual_num_samples < expected_num_samples:
            pad_length = expected_num_samples - actual_num_samples
            processed_waveform = torch.nn.functional.pad(processed_waveform, (0, pad_length))
            
        final_sample_rate = original_sample_rate

        processed_audio = {
            'waveform': processed_waveform.cpu().detach(),
            'sample_rate': self.audio_sample_rate
        }
        original_audio = {
            'waveform': original_waveform.cpu().detach(),
            'sample_rate': self.audio_sample_rate
        }

        #--------------------------------------------------#
        #--------------------------------------------------#
        #--------------------------------------------------#

        waveform_for_rms = processed_waveform.squeeze(0).squeeze(0)
        audio_weights = self._rms_energy(waveform_for_rms, batch_size, samples_per_frame)

        
        if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
            raise ValueError("Invalid audio weights calculated")

        min_weight = np.min(audio_weights)
        max_weight = np.max(audio_weights)
        if max_weight - min_weight != 0:
            audio_weights_normalized = (audio_weights - min_weight) / (max_weight - min_weight)
        else:
            audio_weights_normalized = audio_weights - min_weight

        audio_weights_thresholded = np.where(audio_weights_normalized > threshold, audio_weights_normalized, 0)
        audio_weights_processed = np.clip(audio_weights_thresholded * multiply, 0, 1)

        try:
            figsize = 12.0
            plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
            plt.plot(
                list(range(0, (len(audio_weights_processed)))),
                audio_weights_processed,
                label=f'{analysis_mode} Weights',
                color='blue'
            )
            plt.xlabel(f'Frame Number (Batch Size = {batch_size})')
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
