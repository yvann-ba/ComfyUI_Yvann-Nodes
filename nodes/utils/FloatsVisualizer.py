import tempfile

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ..base import UtilsNodeBase


class FloatsVisualizer(UtilsNodeBase):
    # Define class variables for line styles and colors
    line_styles = ["-", "--", "-."]
    line_colors = ["blue", "green", "red"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS", {
                    "forceInput": True,
                    "tooltip": "Primary list of floats to visualize"
                }),
                "title": ("STRING", {
                    "default": "Graph",
                    "tooltip": "Title displayed at top of graph"
                }),
                "x_label": ("STRING", {
                    "default": "X-Axis",
                    "tooltip": "Label for the horizontal axis"
                }),
                "y_label": ("STRING", {
                    "default": "Y-Axis",
                    "tooltip": "Label for the vertical axis"
                }),
            },
            "optional": {
                "floats_optional1": ("FLOATS", {
                    "forceInput": True,
                    "tooltip": "Second list of floats to compare (green dashed)"
                }),
                "floats_optional2": ("FLOATS", {
                    "forceInput": True,
                    "tooltip": "Third list of floats to compare (red dash-dot)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visual_graph",)
    FUNCTION = "floats_to_graph"

    def floats_to_graph(self, floats, title="Graph", x_label="X-Axis", y_label="Y-Axis",
                        floats_optional1=None, floats_optional2=None):

        try:
            # Create a list of tuples containing (label, data)
            floats_list = [("floats", floats)]
            if floats_optional1 is not None:
                floats_list.append(("floats_optional1", floats_optional1))
            if floats_optional2 is not None:
                floats_list.append(("floats_optional2", floats_optional2))

            # Convert all floats to NumPy arrays and ensure they are the same length
            processed_floats_list = []
            min_length = None
            for label, floats_data in floats_list:
                if isinstance(floats_data, list):
                    floats_array = np.array(floats_data)
                elif isinstance(floats_data, torch.Tensor):
                    floats_array = floats_data.cpu().numpy()
                else:
                    raise ValueError(f"Unsupported type for '{label}' input")
                if min_length is None or len(floats_array) < min_length:
                    min_length = len(floats_array)
                processed_floats_list.append((label, floats_array))

            # Truncate all arrays to the minimum length to match x-axis
            processed_floats_list = [
                (label, floats_array[:min_length]) for label, floats_array in processed_floats_list
            ]

            # Create the plot
            figsize = 12.0
            plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')

            x_values = range(min_length)  # Use the minimum length

            for idx, (label, floats_array) in enumerate(processed_floats_list):
                color = self.line_colors[idx % len(self.line_colors)]
                style = self.line_styles[idx % len(self.line_styles)]
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
