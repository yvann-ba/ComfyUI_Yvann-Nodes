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

class Audio_Remixer(AudioNodeBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "audio": ("AUDIO", {"forceInput": True}),
                "Bass_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.01}),
                "Drums_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.01}),
                "Other_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.01}),
                "Vocals_volume": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO","AUDIO")
    RETURN_NAMES = ("base", "drums", "other", "vocal", "merge_audio")
    FUNCTION = "main"


    def main(self, model, audio: Dict[str, torch.Tensor], Drums_volume: float, Vocals_volume: float, Bass_volume: float, Other_volume: float) -> tuple[Any, Any, Any, Any]:

        if model is None:
            print(colored("Model not set.", 'red'))
            return None

        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        waveform: torch.Tensor = audio['waveform']
        waveform = waveform.squeeze(0).to(device)
        self.audio_sample_rate: int = audio['sample_rate']
        self.model = model

        sources_list: list[str] = []

        if isinstance(self.model, torch.nn.Module):
            print(colored("Applying Open_Unmix model on audio.", 'green'))
            self.model_sample_rate = self.model.sample_rate
            if hasattr(self.model, 'sample_rate') and self.audio_sample_rate != self.model_sample_rate:
                print(colored(f"Resampling from {self.audio_sample_rate} to {self.model_sample_rate}", 'yellow'))
                waveform = torchaudio.transforms.Resample(orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate)(waveform)
            waveform = waveform.unsqueeze(0)
            sources = model(waveform)
            sources = sources.squeeze(0)
            print(f"Shape of sources tensor: {sources.shape}")
            model.sources = ['bass', 'drums', 'other', 'vocals']

            sources_list = model.sources
            print(colored(f"Sources list after Open-Unmix: {sources_list}", 'cyan'))

        elif hasattr(self.model, "get_model"):
            print(colored("Applying GDemucs model on audio.", 'green'))
            self.model_sample_rate: int = model.sample_rate
            model = model.get_model()
            model.to(device)

            if self.audio_sample_rate != self.model_sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate)(waveform)

            # Normalisation
            ref: torch.Tensor = waveform.mean(0)
            waveform = (waveform - ref.mean()) / ref.std()

            sources: torch.Tensor = self.separate_sources(model, waveform[None], segment=10.0, overlap=0.1, device=device)[0]
            print(f"Shape of sources tensor: {sources.shape}")
            sources = sources * ref.std() + ref.mean() 

            sources_list: list[str] = model.sources
            sources: list[torch.Tensor] = list(sources)

        else:
            print(colored("Unrecognized model type.", 'red'))
            return None

        required_sources = ['bass', 'drums', 'other', 'vocals']
        for source in required_sources:
            if source not in sources_list:
                print(colored(f"Warning: '{source}' not found in sources_list.", 'yellow'))
        
        Drums_volume: float = self.adjust_volume_range(Drums_volume)
        Vocals_volume: float = self.adjust_volume_range(Vocals_volume)
        Bass_volume: float = self.adjust_volume_range(Bass_volume)
        Other_volume: float = self.adjust_volume_range(Other_volume)

        audios: Tuple[Any, Any, Any, Any] = self.sources_to_tuple(Drums_volume, Vocals_volume, Bass_volume, Other_volume, dict(zip(sources_list, sources)))

        print(colored(f"Audios: {audios}", 'magenta'))

        merge_audio: torch.Tensor = self.blend_audios([audios[0]["waveform"], audios[1]["waveform"], audios[2]["waveform"], audios[3]["waveform"]])
        
        return audios[0], audios[1], audios[2], audios[3], merge_audio





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
        
        #max_val = torch.max(torch.abs(blended_audio))
        #if max_val > 1.0:
        #    blended_audio = blended_audio / max_val

        if blended_audio.dim() == 2:
            blended_audio = blended_audio.unsqueeze(0)

        return {
            "waveform": blended_audio.cpu(),
            "sample_rate": self.model_sample_rate,
        }

    def sources_to_tuple(self, Drums_volume, Vocals_volume, Bass_volume, Other_volume, sources: Dict[str, torch.Tensor]) -> Tuple[Any, Any, Any, Any]:

        threshold = 0.00

        # Define the expected output order
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

        # print(f"--Type of \"TUPLE[audio]\": {type(outputs[0])}")
        # print(f"--Type of \"TUPLE[audio]\": {type(outputs[1])}")
        # print(f"--Type of \"TUPLE[audio]\": {type(outputs[2])}")
        # print(f"--Type of \"TUPLE[audio]\": {type(outputs[3])}")
        # print(Bass_volume, Drums_volume, Other_volume, Vocals_volume)

        for i, volume in enumerate([Bass_volume, Drums_volume, Other_volume, Vocals_volume]):
            waveform = outputs[i]["waveform"]  # Get the waveform and remove the extra dimension
            mask = torch.abs(waveform) > threshold  # Create a boolean mask for samples above the threshold
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