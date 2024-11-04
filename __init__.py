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

from .nodes.audio.Audio_Analysis import Audio_Analysis
from .nodes.audio.Audio_Reactive_v112 import Audio_Reactive_v112
from .nodes.audio.IPAdapter_Audio_Transitions import IPAdapter_Audio_Transitions
from .nodes.audio.Prompt_Audio_Schedule import Prompt_Audio_Schedule
from .nodes.audio.Audio_Peaks_Detection import Audio_Peaks_Detection
from .nodes.audio.AnimateDiff_Audio_Reactive import AnimateDiff_Audio_Reactive
from .nodes.audio.ControlNet_Audio_Reactive import ControlNet_Audio_Reactive

from .nodes.utils.Floats_To_Weights_Strategy import Floats_To_Weights_Strategy
from .nodes.utils.Invert_Floats import Invert_Floats
from .nodes.utils.Floats_Visualizer import Floats_Visualizer
from .nodes.utils.Mask_To_Float import Mask_To_Float


NODE_CLASS_MAPPINGS = {
    "Audio Analysis": Audio_Analysis,
    "Audio_Reactive_v112": Audio_Reactive_v112,
    "IPAdapter Audio Transitions": IPAdapter_Audio_Transitions,
    "Prompt Audio Schedule": Prompt_Audio_Schedule,
    "Audio Peaks Detection": Audio_Peaks_Detection,
    "AnimateDiff Audio Reactive": AnimateDiff_Audio_Reactive,
    "ControlNet Audio Reactive": ControlNet_Audio_Reactive,
    "Floats To Weights Strategy": Floats_To_Weights_Strategy,
    "Invert Floats": Invert_Floats,
    "Floats Visualizer": Floats_Visualizer,
    "Mask To Float": Mask_To_Float,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Analysis": "Audio Analysis",
    "Audio_Reactive_v112": "Audio_Reactive_v112",
    "IPAdapter Audio Transitions": "IPAdapter Audio Transitions",
    "Prompt Audio Schedule": "Prompt Audio Schedule",
    "Audio Peaks Detection": "Audio Peaks Detection",
    "AnimateDiff Audio Reactive": "AnimateDiff Audio Reactive",
    "ControlNet Audio Reactive": "ControlNet Audio Reactive",
    "Floats To Weights Strategy": "Floats To Weights Strategy",
    "Invert Floats": "Invert Floats",
    "Floats Visualizer": "Floats Visualizer",
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
