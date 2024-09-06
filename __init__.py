from .node_configs import CombinedMeta
from collections import OrderedDict

# credit to RyanOnTheInside, KJNodes, MTB, Akatz, their works helped me a lot


#allows for central management and inheritance of class variables for help documentation
class Yvann(metaclass=CombinedMeta):
    @classmethod
    def get_description(cls):
        
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(cls.__name__, cls.__name__)
        footer = "For more information, visit [RyanOnTheInside GitHub](https://github.com/ryanontheinside).\n\n"
        footer += "For tutorials and example workflows visit [RyanOnTheInside Civitai](https://civitai.com/user/ryanontheinside).\n\n"
        display_name = display_name.replace(" | Yvann", "")
        
        desc = f"# {display_name}\n\n"
        
        if hasattr(cls, 'DESCRIPTION'):
            desc += f"{cls.DESCRIPTION}\n\n{footer}"
            return desc

        if hasattr(cls, 'TOP_DESCRIPTION'):
            desc += f"### {cls.TOP_DESCRIPTION}\n\n"
        
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
 
NODE_CLASS_MAPPINGS = {
    "Audio Drums Analysis | Yvann": Audio_Drums_Analysis_Yvann,
    "Audio Vocals Analysis | Yvann": Audio_Vocals_Analysis_Yvann,
    "Audio Analysis | Yvann": Audio_Analysis_Yvann,
}

WEB_DIRECTORY = "./web/js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Drums Analysis | Yvann": "Audio Drums Analysis | Yvann",
    "Audio Vocals Analysis | Yvann": "Audio Vocals Analysis | Yvann",
    "Audio Analysis | Yvann": "Audio Analysis | Yvann",
}

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
