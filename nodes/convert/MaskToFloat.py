from ... import Yvann

# To do the opposite (Float To Mask), you can install the node pack "ComfyUI-KJNodes"
class ConvertNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🔄 Convert"

class MaskToFloat(ConvertNodeBase):
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK", {"forceInput": True}),
			},
		}
	RETURN_TYPES = ("FLOATS",)
	RETURN_NAMES = ("floats",)
	FUNCTION = "mask_to_float"

	def mask_to_float(self, mask):
		import torch

		# Ensure mask is a torch.Tensor
		if not isinstance(mask, torch.Tensor):
			raise ValueError("Input 'mask' must be a torch.Tensor")

		# Handle case where mask may have shape [H, W] instead of [B, H, W]
		if mask.dim() == 2:
			mask = mask.unsqueeze(0)  # Add batch dimension

		# mask has shape [B, H, W]
		batch_size = mask.shape[0]
		output_values = []

		for i in range(batch_size):
			single_mask = mask[i]  # shape [H, W]
			mean_value = round(single_mask.mean().item(), 6)  # Compute mean pixel value
			output_values.append(mean_value)

		return (output_values,)
