import os
import folder_paths
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from typing import Any
from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Load_Audio_Separation_Model(AudioNodeBase):
    audio_models = ["Open-Unmix", "GDemucs"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "model": (cls.audio_models,),
            }
        }

    RETURN_TYPES: tuple[str] = ("MODEL",)
    FUNCTION: str = "main"
    
    def load_OpenUnmix(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        download_path = os.path.join(folder_paths.models_dir, "openunmix")
        os.makedirs(download_path, exist_ok=True)

        model_file = "umxhq.pth"
        model_path = os.path.join(download_path, model_file)

        if not os.path.exists(model_path):
            print(f"Downloading {model} model...")
            try:
                separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', device='cpu')
            except RuntimeError as e:
                print(f"Error during download model : {e}")
                return None
            torch.save(separator.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', model, device='cpu')
            separator.load_state_dict(torch.load(model_path, map_location='cpu'))

        separator = separator.to(device)
        separator.eval()

        return (separator,)
    
    def load_GDemucs(self):
    
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bundle: Any = HDEMUCS_HIGH_MUSDB_PLUS
        model: torch.nn.Module = bundle.get_model()
        model.to(device)
        self.model_sample_rate: int = bundle.sample_rate
        print("HDemucs model is loaded")
        return (model,)


    def main(self, model) -> None:

        if model == "Open-Unmix":
            print("Open-Unmix selected")
            return (self.load_OpenUnmix(model))
             
        elif model == "GDemucs":
            print("GDemucs selected")
            return (self.load_GDemucs())
        else:
            print("Invalid selection")

