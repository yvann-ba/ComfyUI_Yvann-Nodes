import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tempfile
import numpy as np
import math
from PIL import Image
from ... import Yvann

class AudioNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class AudioIPAdapterTransitions(AudioNodeBase):
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"forceInput": True}),
				"peaks_weights": ("FLOAT", {"forceInput": True}),
				"blend_mode": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
				"transitions_length": ("INT", {"default": 5, "min": 1, "step": 2}),
				"min_IPA_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.9}),
				"max_IPA_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
			}
		}

	RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "FLOAT", "IMAGE")
	RETURN_NAMES = ("image_1", "weights", "image_2", "weights_invert", "graph_transitions")
	FUNCTION = "process_transitions"

	def process_transitions(self, images, peaks_weights, blend_mode, transitions_length, min_IPA_weight, max_IPA_weight):

		if not isinstance(peaks_weights, (list, np.ndarray)):
			print("Invalid peaks_weights input")
			return None, None, None, None, None

		# Convert peaks_weights to numpy array and ensure it's binary (0 or 1)
		peaks_binary = np.array(peaks_weights, dtype=int)
		total_frames = len(peaks_binary)

		# Generate switch indices by incrementing index at each peak
		switch_indices = []
		index_value = 0
		for peak in peaks_binary:
			if peak == 1:
				index_value += 1
			switch_indices.append(index_value)

		# images is a batch of images: tensor of shape [B, H, W, C]
		if images.dim() == 3:
			images = images.unsqueeze(0)  # Add batch dimension if missing

		num_images = images.shape[0]
		if num_images < 2:
			print("At least two images are required for transitions.")
			return None, None, None, None, None

		unique_indices = sorted(set(switch_indices))
		num_indices = len(unique_indices)

		# Map indices to image indices (cycling through images if necessary)
		image_indices = [i % num_images for i in unique_indices]

		# Create a mapping from switch index to image
		image_mapping = {idx: images[image_idx] for idx, image_idx in zip(unique_indices, image_indices)}

		# Initialize blending_weights, images1, images2
		blending_weights = np.zeros(total_frames, dtype=np.float32)
		images1 = [image_mapping[switch_indices[i]] for i in range(total_frames)]
		images2 = images1.copy()

		# Identify frames where index changes
		change_frames = [i for i in range(1, total_frames) if switch_indices[i] != switch_indices[i - 1]]

		# For each transition, compute blending weights
		for change_frame in change_frames:
			start = max(0, change_frame - transitions_length // 2)
			end = min(total_frames, change_frame + (transitions_length + 1) // 2)
			n = end - start - 1
			idx_prev = switch_indices[change_frame - 1] if change_frame > 0 else switch_indices[change_frame]
			idx_next = switch_indices[change_frame]

			for i in range(start, end):
				t = (i - start) / n if n > 0 else 1.0

				# Compute blending weight based on blend_mode
				if blend_mode == "linear":
					blending_weight = t
				elif blend_mode == "ease_in_out":
					blending_weight = (1 - math.cos(t * math.pi)) / 2
				elif blend_mode == "ease_in":
					blending_weight = math.sin(t * math.pi / 2)
				elif blend_mode == "ease_out":
					blending_weight = 1 - math.cos(t * math.pi / 2)
				else:
					blending_weight = t

				blending_weight = min(max(blending_weight, 0.0), 1.0)

				# Update blending_weights
				blending_weights[i] = blending_weight

				# Update images1 and images2
				images1[i] = image_mapping[idx_prev]
				images2[i] = image_mapping[idx_next]

		# Now, blending_weights correspond to image_2
		blending_weights_raw = blending_weights.copy()  # Keep the raw weights for internal use

		# Apply custom range to weights
		blending_weights = blending_weights * (max_IPA_weight - min_IPA_weight) + min_IPA_weight
		blending_weights = [round(w, 6) for w in blending_weights]
		weights_invert = [(max_IPA_weight + min_IPA_weight) - w for w in blending_weights]
		weights_invert = [round(w, 6) for w in weights_invert]

		# Convert lists to tensors
		images1 = torch.stack(images1)
		images2 = torch.stack(images2)
		blending_weights_tensor = torch.tensor(blending_weights_raw, dtype=images1.dtype).view(-1, 1, 1, 1)

		# Ensure blending weights are compatible with image dimensions
		blending_weights_tensor = blending_weights_tensor.to(images1.device)

		# Generate visualization of transitions
		try:
			figsize = 12.0
			plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')

			blending_weights_array = np.array(blending_weights_raw)
			plt.plot(range(0, len(blending_weights_array)), blending_weights_array, label='Blending Weights', color='green', alpha=0.5)
			plt.scatter(change_frames, blending_weights_array[change_frames], color='red', label='Transitions')

			plt.xlabel('Frames')
			plt.title('Image Transitions with Blending Weights')
			plt.legend()
			plt.grid(True)

			# Remove Y-axis labels
			plt.yticks([])

			# Ensure x-axis labels are integers
			ax = plt.gca()
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))

			with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
				plt.savefig(tmpfile.name, format='png')
				tmpfile_path = tmpfile.name
			plt.close()

			visualization = Image.open(tmpfile_path).convert("RGB")
			visualization = np.array(visualization).astype(np.float32) / 255.0
			visualization = torch.from_numpy(visualization).unsqueeze(0)  # Shape: [1, C, H, W]

		except Exception as e:
			print(f"Error in creating visualization: {e}")
			visualization = None

		# Return values with adjusted weights and images
		return images2, blending_weights, images1, weights_invert, visualization
