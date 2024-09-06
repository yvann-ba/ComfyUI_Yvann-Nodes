class ShowFloat_YVANN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float" : ("FLOAT", {"forceInput": True})
            },
        }
        
    INPUT_IS_LIST = True
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)

    CATEGORY = "float"
    FUNCTION = "show_float"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    
    def show_float(self, float):
        return {"ui": {"text": float}, "result": (float,)}