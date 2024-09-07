from .node_configs import CombinedMeta
from collections import OrderedDict

# credit to RyanOnTheInside, KJNodes, MTB, Fill, Akatz, their works helped me a lot


#allows for central management and inheritance of class variables for help documentation
from collections import OrderedDict

class Yvann(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        footer = "Code : [Yvann GitHub](https://github.com/yvann-ba)\n"
        footer += "Tutorials and Workflows: [Yvann Youtube](https://www.youtube.com/@yvann.mp4)\n\n"
        
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
   
from .nodes.audio.Audio_Drums_Analysis_Yvann import Audio_Drums_Analysis_Yvann
from .nodes.audio.Audio_Vocals_Analysis_Yvann import Audio_Vocals_Analysis_Yvann
from .nodes.audio.Audio_Analysis_Yvann import Audio_Analysis_Yvann
from .nodes.audio.previewaudio2 import PreviewAudio2
from .nodes.audio.imagemask import ImageAndMaskPreview2

 
NODE_CLASS_MAPPINGS = {
    "Audio Drums Analysis | Yvann": Audio_Drums_Analysis_Yvann,
    "Audio Vocals Analysis | Yvann": Audio_Vocals_Analysis_Yvann,
    "Audio Analysis | Yvann": Audio_Analysis_Yvann,
    "Preview Audio | Yvann": PreviewAudio2,
    "Image And Mask Preview | Yvann": ImageAndMaskPreview2,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Drums Analysis | Yvann": "Audio Drums Analysis | Yvann",
    "Audio Vocals Analysis | Yvann": "Audio Vocals Analysis | Yvann",
    "Audio Analysis | Yvann": "Audio Analysis | Yvann",
    "Preview Audio | Yvann": "Preview Audio | Yvann",
    "Image And Mask Preview | Yvann": "Image And Mask Preview | Yvann",
}

Yvann_Print = """
🔊 Yvann Audio Reactive Nodes"""

print("\033[38;5;195m" + Yvann_Print + "\033[38;5;222m" + " : Loaded\n" + "\033[0m")




from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):

    # NOTE: we add an extra static path to avoid comfy mechanism
    # that loads every script in web. 

    PromptServer.instance.app.add_routes(
        [web.static("/yvann_web_async", (Path(__file__).parent.absolute() / "yvann_web_async").as_posix())]
    )



for node_name, node_class in NODE_CLASS_MAPPINGS.items():
    if hasattr(node_class, 'get_description'):
        desc = node_class.get_description()
        node_class.DESCRIPTION = desc

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
