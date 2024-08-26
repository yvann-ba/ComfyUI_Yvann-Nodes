class ShowText_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text" : ("STRING", {"forceInput": True})
            },
        }
        
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    CATEGORY = "text"
    FUNCTION = "show_text"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    
    def show_text(self, text):
        return {"ui": {"text": text}, "result": (text,)}
