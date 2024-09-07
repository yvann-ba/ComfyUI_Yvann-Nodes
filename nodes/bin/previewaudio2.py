import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import io
import json
import struct
import random
import hashlib
from comfy.cli_args import args
from comfy_extras.nodes_audio import SaveAudio

class PreviewAudio2(SaveAudio):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }