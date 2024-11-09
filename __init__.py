# Thanks to RyanOnTheInside, KJNodes, MTB, Fill, Akatz, Matheo, their works helped me a lot
from pathlib import Path
from aiohttp import web
from .node_configs import CombinedMeta
from collections import OrderedDict
from server import PromptServer

class Yvann(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        footer = "\n\n"
        footer = "#### 🐙 Docs, Workflows and Code: [Yvann-Nodes GitHub](https://github.com/yvann-ba/ComfyUI_Yvann-Nodes) "
        footer += " 👁️ Tutorials: [Yvann Youtube](https://www.youtube.com/@yvann.mp4)\n"

        desc = ""

        if hasattr(cls, 'DESCRIPTION'):
            desc += f"{cls.DESCRIPTION}\n\n{footer}"
            return desc

        if hasattr(cls, 'TOP_DESCRIPTION'):
            desc += f"{cls.TOP_DESCRIPTION}\n\n"

        if hasattr(cls, "BASE_DESCRIPTION"):
            desc += cls.BASE_DESCRIPTION + "\n\n"

        additional_info = OrderedDict()
        for c in cls.mro()[::-1]:
            if hasattr(c, 'ADDITIONAL_INFO'):
                info = c.ADDITIONAL_INFO.strip()
                additional_info[c.__name__] = info

        if additional_info:
            desc += "\n\n".join(additional_info.values()) + "\n\n"

        if hasattr(cls, 'BOTTOM_DESCRIPTION'):
            desc += f"{cls.BOTTOM_DESCRIPTION}\n\n"

        desc += footer
        return desc

from .nodes.audio.AudioAnalysis import AudioAnalysis
from .nodes.audio.AudioIPAdapterTransitions import AudioIPAdapterTransitions
from .nodes.audio.AudioPromptSchedule import AudioPromptSchedule
from .nodes.audio.AudioPeaksDetection import AudioPeaksDetection
from .nodes.audio.AudioAnimateDiffSchedule import AudioAnimateDiffSchedule
from .nodes.audio.AudioControlNetSchedule import AudioControlNetSchedule
from .nodes.audio.AudioRemixer import AudioRemixer
from .nodes.audio.LoadAudioSeparationModel import LoadAudioSeparationModel

from .nodes.utils.InvertFloats import InvertFloats
from .nodes.utils.FloatsVisualizer import FloatsVisualizer

from .nodes.convert.FloatsToWeightsStrategy import FloatsToWeightsStrategy
from .nodes.convert.MaskToFloat import MaskToFloat



NODE_CLASS_MAPPINGS = {
    "Audio Analysis": AudioAnalysis,
    "Audio IPAdapter Transitions": AudioIPAdapterTransitions,
    "Audio Prompt Schedule": AudioPromptSchedule,
    "Audio Peaks Detection": AudioPeaksDetection,
    "Audio AnimateDiff Schedule": AudioAnimateDiffSchedule,
    "Audio ControlNet Schedule": AudioControlNetSchedule,
    "Audio Remixer": AudioRemixer,
    "Load Audio Separation Model": LoadAudioSeparationModel,
    
    "Invert Floats": InvertFloats,
    "Floats Visualizer": FloatsVisualizer,

    "Mask To Float": MaskToFloat,
    "Floats To Weights Strategy": FloatsToWeightsStrategy,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Analysis": "Audio Analysis",
    "Audio IPAdapter Transitions": "Audio IPAdapter Transitions",
    "Audio Remixer": "Audio Remixer",
    "Load Audio Separation Model": "Load Audio Separation Model",
    "Audio Prompt Schedule": "Audio Prompt Schedule",
    "Audio Peaks Detection": "Audio Peaks Detection",
    
    "Audio AnimateDiff Schedule": "Audio AnimateDiff Schedule",
    "Audio ControlNet Schedule": "Audio ControlNet Schedule",
    
    "Invert Floats": "Invert Floats",
    "Floats Visualizer": "Floats Visualizer",

    "Floats To Weights Strategy": "Floats To Weights Strategy",
    "Mask To Float":  "Mask To Float",
}

Yvann_Print = """
🔊 Yvann Audio Reactive & Utils Node"""

print("\033[38;5;195m" + Yvann_Print +
      "\033[38;5;222m" + " : Loaded\n" + "\033[0m")


if hasattr(PromptServer, "instance"):
    # NOTE: we add an extra static path to avoid comfy mechanism
    # that loads every script in web.
    PromptServer.instance.app.add_routes(
        [web.static("/yvann_web_async",
                    (Path(__file__).parent.absolute() / "yvann_web_async").as_posix())]
    )

for node_name, node_class in NODE_CLASS_MAPPINGS.items():
    if hasattr(node_class, 'get_description'):
        desc = node_class.get_description()
        node_class.DESCRIPTION = desc

__all__ = ["NODE_CLASS_MAPPINGS",
           "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
