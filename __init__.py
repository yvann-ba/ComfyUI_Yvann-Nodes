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
        footer = "#### 🐙 Docs, Workflows and Code: [Yvann-Nodes GitHub](https://github.com/yvann-ba/ComfyUI-Nodes) "
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
from .nodes.audio.IPAdapter_Audio_Transitions import IPAdapter_Audio_Transitions
from .nodes.audio.Audio_PromptSchedule import Audio_PromptSchedule
from .nodes.utils.Floats_To_Weights_Strategy import Floats_To_Weights_Strategy
from .nodes.utils.Invert_Floats import Invert_Floats
from .nodes.utils.Floats_Visualizer import Floats_Visualizer
from .nodes.utils.Mask_To_Float import Mask_To_Float


NODE_CLASS_MAPPINGS = {
    "Audio Analysis": Audio_Analysis,
    "IPAdapter Audio Transitions": IPAdapter_Audio_Transitions,
    "Audio Prompt Schedule": Audio_PromptSchedule,
    "Floats To Weights Strategy": Floats_To_Weights_Strategy,
    "Invert Floats": Invert_Floats,
    "Floats Visualizer": Floats_Visualizer,
    "Mask To Float": Mask_To_Float,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Analysis": "Audio Analysis",
    "IPAdapter Audio Transitions": "IPAdapter Audio Transitions",
    "Audio Prompt Schedule": "Audio Prompt Schedule",
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
