import os
import folder_paths
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from typing import Any
from ... import Yvann
import comfy.model_management as mm


class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class LoadAudioSeparationModel(AudioNodeBase):
    audio_models = ["Hybrid Demucs", "Open-Unmix"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "model": (cls.audio_models,),
            }
        }

    RETURN_TYPES = ("AUDIO_SEPARATION_MODEL",)
    RETURN_NAMES = ("audio_sep_model",)
    FUNCTION = "main"
    
    def load_OpenUnmix(self, model):
        device = mm.get_torch_device()
        download_path = os.path.join(folder_paths.models_dir, "openunmix")
        os.makedirs(download_path, exist_ok=True)

        model_file = "umxl.pth"
        model_path = os.path.join(download_path, model_file)

        if not os.path.exists(model_path):
            print(f"Downloading {model} model...")
            try:
                separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device='cpu')
            except RuntimeError as e:
                print(f"Failed to download model : {e}")
                return None
            torch.save(separator.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device='cpu')
            separator.load_state_dict(torch.load(model_path, map_location='cpu'))

        separator = separator.to(device)
        separator.eval()

        return (separator,)
    
    def load_HDemucs(self):
        
        device = mm.get_torch_device()
        bundle: Any = HDEMUCS_HIGH_MUSDB_PLUS
        print("Hybrid Demucs model is loaded")
        model_info = {
            "demucs": True,
            "model": bundle.get_model().to(device),
            "sample_rate": bundle.sample_rate
        }
        return (model_info,)


    def main(self, model):

        if model == "Open-Unmix":
            return (self.load_OpenUnmix(model))
        else:
            return (self.load_HDemucs())