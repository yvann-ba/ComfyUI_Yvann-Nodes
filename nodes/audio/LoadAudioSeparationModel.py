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
    RETURN_NAMES = ("model",)
    FUNCTION: str = "main"
    
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
                print(f"Error during download model : {e}")
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
    
        device: torch.device = mm.get_torch_device()

        bundle: Any = HDEMUCS_HIGH_MUSDB_PLUS
        # model: torch.nn.Module = bundle.get_model()
        # model.to(device)
        # self.model_sample_rate: int = bundle.sample_rate
        print("Hybrid Demucs model is loaded")
        return (bundle,)


    def main(self, model) -> None:

        if model == "Open-Unmix":
            print("Open-Unmix selected")
            return (self.load_OpenUnmix(model))
             
        elif model == "Hybrid Demucs":
            print("Hybrid Demucs selected")
            return (self.load_HDemucs())
        else:
            print("Invalid selection")

