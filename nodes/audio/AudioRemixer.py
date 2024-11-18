import torch
import torchaudio

# from IPython.display import Audio
# from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset

from typing import Dict, Tuple, Any
from torchaudio.transforms import Fade, Resample
from termcolor import colored

from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class AudioRemixer(AudioNodeBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_sep_model": ("AUDIO_SEPARATION_MODEL", {"forceInput": True}),
                "audio": ("AUDIO", {"forceInput": True}),
                "drums_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.1}),
                "vocals_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.1}),
                "bass_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.1}),
                "others_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("merged_audio",)
    FUNCTION = "main"


    def main(self, audio_sep_model, audio: Dict[str, torch.Tensor], drums_volume: float, vocals_volume: float, bass_volume: float, others_volume: float) -> tuple[torch.Tensor]:
    
        model = audio_sep_model
        # 1. Prepare audio and device
        device, waveform = self.prepare_audio_and_device(audio)

        # 2. Apply model and extract sources
        sources, sources_list = self.apply_model_and_extract_sources(model, waveform, device)

        if sources is None:
            return None  # Return if the model is unrecognized

        # 3. Adjust volumes and merge sources
        merge_audio = self.process_and_merge_audio(sources, sources_list, drums_volume, vocals_volume, bass_volume, others_volume)

        return (merge_audio,)


    def prepare_audio_and_device(self, audio: Dict[str, torch.Tensor]) -> Tuple[torch.device, torch.Tensor]:
        """Prepares the device (GPU or CPU) and sets up the audio waveform."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        waveform = audio['waveform'].squeeze(0).to(device)
        self.audio_sample_rate = audio['sample_rate']
        return device, waveform


    def apply_model_and_extract_sources(self, model, waveform: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, list[str]]:
        """Applies the model and extracts audio sources, handling both Open-Unmix and GDemucs cases."""
        sources, sources_list = None, []

        if isinstance(model, torch.nn.Module):  # Open-Unmix model
            print(colored("Applying Open_Unmix model on audio.", 'green'))
            self.model_sample_rate = int(model.sample_rate)

            if self.audio_sample_rate != self.model_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate).to(device)
                waveform = resampler(waveform)
            sources = model(waveform.unsqueeze(0)).squeeze(0)
            sources_list = ['bass', 'drums', 'other', 'vocals']

        elif "demucs" in model and model["demucs"]:  # GDemucs model
            print(colored("Applying GDemucs model on audio", 'green'))
            self.model_sample_rate = int(model["sample_rate"])
            model = model["model"]

            if self.audio_sample_rate != self.model_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate).to(device)
                waveform = resampler(waveform)
            ref = waveform.mean(0)
            waveform = (waveform - ref.mean()) / ref.std()
            sources = self.separate_sources(model, waveform[None], segment=10.0, overlap=0.1, device=device)[0]
            sources = sources * ref.std() + ref.mean()
            sources_list = model.sources

        else:
            print(colored("Unrecognized model type", 'red'))
            return None, []

        return sources, sources_list


    def process_and_merge_audio(self, sources: torch.Tensor, sources_list: list[str], drums_volume: float, vocals_volume: float, bass_volume: float, others_volume: float) -> torch.Tensor:
        """Adjusts source volumes and merges them into a single audio output"""
        required_sources = ['bass', 'drums', 'other', 'vocals']
        for source in required_sources:
            if source not in sources_list:
                print(colored(f"Warning: '{source}' not found in sources_list", 'yellow'))

        # Adjust volume levels
        drums_volume = self.adjust_volume_range(drums_volume)
        vocals_volume = self.adjust_volume_range(vocals_volume)
        bass_volume = self.adjust_volume_range(bass_volume)
        others_volume = self.adjust_volume_range(others_volume)

        # Convert to tuple and blend
        audios = self.sources_to_tuple(drums_volume, vocals_volume, bass_volume, others_volume, dict(zip(sources_list, sources)))
        return self.blend_audios([audios[0]["waveform"], audios[1]["waveform"], audios[2]["waveform"], audios[3]["waveform"]])

    def adjust_volume_range(self, value):
        if value <= -10:
                return 0
        elif value >= 10:
            return 10
        elif value <= 0:
            return (value + 10) / 10
        else:
            return 1 + (value / 10) * 9

    def blend_audios(self, audio_tensors):
        blended_audio = sum(audio_tensors)

        if blended_audio.dim() == 2:
            blended_audio = blended_audio.unsqueeze(0)

        return {
            "waveform": blended_audio.cpu(),
            "sample_rate": self.model_sample_rate,
        }

    def sources_to_tuple(self, drums_volume, vocals_volume, bass_volume, others_volume, sources: Dict[str, torch.Tensor]) -> Tuple[Any, Any, Any, Any]:

        threshold = 0.00

        output_order = ["bass", "drums", "other", "vocals"]
        outputs = []

        for source in output_order:
            if source not in sources:
                raise ValueError(f"Missing source {source} in the output")
            outputs.append(
                {
                    "waveform": sources[source].cpu().unsqueeze(0),
                    "sample_rate": self.model_sample_rate,
                }
            )

        for i, volume in enumerate([bass_volume, drums_volume, others_volume, vocals_volume]):
            waveform = outputs[i]["waveform"]
            mask = torch.abs(waveform) > threshold
            outputs[i]["waveform"] = waveform * volume * mask.float() + waveform * (1 - mask.float())

        return tuple(outputs)

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