from ... import Yvann

class ConvertNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔄 Convert"
    
class BboxListToBboxListOfTuples(ConvertNodeBase) :
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("BBOX")
    RETURN_NAMES = ("bboxes")
    FUNCTION = "bbox_list_to_bbox_list_of_tuples"
    
    def bbox_list_to_bbox_list_of_tuples(self, bboxes):
        tuple_list = [tuple(x) for x in bboxes]
        
        return (tuple_list,)