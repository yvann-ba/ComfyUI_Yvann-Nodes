import os
import subprocess
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# requirements_path = os.path.join(current_dir, 'requirements.txt')

# def install_requirements():
#     if os.path.isfile(requirements_path):
#         subprocess.check_call([sys.executable, "-s", "-m", "pip", "install", "-r", requirements_path])

# install_requirements()

from .nodes.AudioAnalysis_YVANN import AudioAnalysis_YVANN
from .nodes.ShowText_YVANN import ShowText_YVANN
from .nodes.AudioAnalysis_YVANN import DownloadOpenUnmixModel

NODE_CLASS_MAPPINGS = {
    ""
    "Audio Analysis YVANN": AudioAnalysis_YVANN,
    "Show Text YVANN": ShowText_YVANN,
    "Download Open Unmix Model": DownloadOpenUnmixModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalysis_YVANN": "Audio Analysis YVANN",
    "ShowText_YVANN": "Show Text YVANN"
}

# ascii_art = """
# """
# print(ascii_art)

#WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS']