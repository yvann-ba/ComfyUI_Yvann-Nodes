# import numpy as np
# from ... import Yvann


# class UtilsBase(Yvann):
#     CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"


# class Math_Float_List(UtilsBase):
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "floats_list": ("FLOAT", {"forceInput": True}),
#                 "multiply_by": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
#             }
#         }

#     RETURN_TYPES = ("FLOAT",)
#     # RETURN_NAMES = ("float_multiplied")
#     FUNCTION = "multiply_float"

#     def multiply_float(self, floats_list,  multiply_by):

#         multiplied_list = [i * multiply_by for i in floats_list]

#         return (multiplied_list,)
