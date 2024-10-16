import torch
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
from ... import Yvann

class UtilsNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"

class Floats_Visualizer_Yvann(UtilsNodeBase):
	# Define class variables for line styles and colors
	line_styles = ["-", "--", "-."]
	line_colors = ["blue", "green", "red"]

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"floats": ("FLOAT", {"forceInput": True}),
				"title": ("STRING", {"default": "Graph"}),
				"x_label": ("STRING", {"default": "X-Axis"}),
				"y_label": ("STRING", {"default": "Y-Axis"}),
			},
			"optional": {
				"floats_optional2": ("FLOAT", {"forceInput": True}),
				"floats_optional3": ("FLOAT", {"forceInput": True}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("visual_graph",)
	FUNCTION = "floats_to_graph"

	def floats_to_graph(self, floats, title="Graph", x_label="X-Axis", y_label="Y-Axis",
						floats_optional1=None, floats_optional2=None):

		try:
			# Collect all floats into a list
			floats_list = [floats]
			if floats_optional1 is not None:
				floats_list.append(floats_optional1)
			if floats_optional2 is not None:
				floats_list.append(floats_optional2)

			# Convert all floats to NumPy arrays and ensure they are the same length
			processed_floats_list = []
			min_length = None
			for floats in floats_list:
				if isinstance(floats, list):
					floats_array = np.array(floats)
				elif isinstance(floats, torch.Tensor):
					floats_array = floats.cpu().numpy()
				else:
					raise ValueError("Unsupported type for 'floats' input")
				if min_length is None or len(floats_array) < min_length:
					min_length = len(floats_array)
				processed_floats_list.append(floats_array)

			# Truncate all arrays to the minimum length to match x-axis
			processed_floats_list = [floats_array[:min_length] for floats_array in processed_floats_list]

			# Create the plot
			figsize = 12.0
			plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')

			x_values = range(min_length)  # Use the minimum length

			for idx, floats_array in enumerate(processed_floats_list):
				color = self.line_colors[idx % len(self.line_colors)]
				style = self.line_styles[idx % len(self.line_styles)]
				label = f'floats {idx+1}'
				plt.plot(x_values, floats_array, label=label, color=color, linestyle=style)

			plt.title(title)
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.grid(True)
			plt.legend()

			# Save the plot to a temporary file
			with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
				plt.savefig(tmpfile.name, format='png', bbox_inches='tight')
				tmpfile_path = tmpfile.name
			plt.close()

			# Load the image and convert to tensor
			visualization = Image.open(tmpfile_path).convert("RGB")
			visualization = np.array(visualization).astype(np.float32) / 255.0
			visualization = torch.from_numpy(visualization).unsqueeze(0)  # Shape: [1, H, W, C]

		except Exception as e:
			print(f"Error in creating visualization: {e}")
			visualization = None

		return (visualization,)
