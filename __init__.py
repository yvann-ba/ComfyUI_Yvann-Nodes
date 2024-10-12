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
        footer = "#### 🐙 Documentation, Workflows and Code Source: [Yvann-Nodes GitHub](https://github.com/yvann-ba/ComfyUI_Yvann-Nodes)\n"
        footer += "#### 👁️ Tutorials: [Yvann Youtube](https://www.youtube.com/@yvann.mp4)\n"

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

from .nodes.audio.Audio_Reactive_Yvann import Audio_Reactive_Yvann
from .nodes.audio.Audio_Reactive_IPAdapter_Yvann import Audio_Reactive_IPAdapter_Yvann
from .nodes.utils.Floats_To_Weights_Strategy_Yvann import Floats_To_Weights_Strategy_Yvann
from .nodes.utils.Invert_Floats_Yvann import Invert_Floats_Yvann
from .nodes.utils.Floats_Visualizer_Yvann import Floats_Visualizer_Yvann


NODE_CLASS_MAPPINGS = {
    "Audio Reactive | Yvann": Audio_Reactive_Yvann,
    "Audio Reactive IPAdapter | Yvann": Audio_Reactive_IPAdapter_Yvann,
    "Floats to Weights Strategy | Yvann": Floats_To_Weights_Strategy_Yvann,
    "Invert Floats | Yvann": Invert_Floats_Yvann,
    "Floats Visualizer | Yvann": Floats_Visualizer_Yvann,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Reactive | Yvann": "Audio Reactive | Yvann",
    "Audio Reactive IPAdapter | Yvann": "Audio Reactive IPAdapter | Yvann",
    "Floats to Weights Strategy | Yvann": "Floats to Weights Strategy | Yvann",
    "Invert Floats | Yvann": "Invert Floats | Yvann",
    "Floats Visualizer | Yvann": "Floats Visualizer | Yvann",
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
