import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Import for integer x-axis labels
import tempfile
import numpy as np
import math
from PIL import Image
from scipy.signal import find_peaks
from ... import Yvann

class AudioNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class IPAdapter_Audio_Transitions(AudioNodeBase):
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {"forceInput": True}),
				"audio_weights": ("FLOAT", {"forceInput": True}),
				"timing": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
				"transition_frames": ("INT", {"default": 4, "min": 1, "step": 1}),
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
			}
		}

	RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "FLOAT", "IMAGE")
	RETURN_NAMES = ("image_1", "weights", "image_2", "weights_invert", "graph_audio_index")
	FUNCTION = "process_weights"

	def process_weights(self, images, audio_weights, timing, transition_frames, threshold):

		if not isinstance(audio_weights, list) and not isinstance(audio_weights, np.ndarray):
			print("Invalid audio_weights input")
			return None, None, None, None, None

		if images is None or not isinstance(images, torch.Tensor):
			print("Invalid or empty images input")
			return None, None, None, None, None

		# Normalize audio_weights
		audio_weights = np.array(audio_weights, dtype=np.float32)
		min_weight = np.min(audio_weights)
		max_weight = np.max(audio_weights)
		weights_range = max_weight - min_weight
		weights_normalized = (audio_weights - min_weight) / weights_range if weights_range > 0 else audio_weights - min_weight

		# Detect peaks
		peaks, _ = find_peaks(weights_normalized, height=threshold)

		# Generate indices based on peaks
		indices = []
		index_value = 0
		peak_set = set(peaks)
		for i in range(len(weights_normalized)):
			if i in peak_set:
				index_value += 1
			indices.append(index_value)

		total_frames = len(indices)

		# images is a batch of images: tensor of shape [B, H, W, C]
		if images.dim() == 3:
			images = images.unsqueeze(0)  # Add batch dimension if missing

		num_images = images.shape[0]
		unique_indices = sorted(set(indices))
		num_indices = len(unique_indices)

		image_indices = [i % num_images for i in range(num_indices)]
		
		# Create a mapping from index to image
		image_mapping = {idx: images[image_idx] for idx, image_idx in zip(unique_indices, image_indices)}

		# Implement the 'alternate batches' method
		blending_weights = []
		images1 = []
		images2 = []

		current_image = image_mapping[indices[0]]
		next_image = None
		transition_counter = 0
		in_transition = False

		for i in range(total_frames):
			if i == 0:
				blending_weight = 0.0
			else:
				if indices[i] != indices[i-1]:
					# Start of transition
					in_transition = True
					transition_counter = 0
					current_image = image_mapping[indices[i-1]]
					next_image = image_mapping[indices[i]]

			if in_transition:
				# Generate blending weight using timing function
				n = transition_frames - 1
				t = transition_counter / n if n > 0 else 1.0

				if timing == "linear":
					blending_weight = t
				elif timing == "ease_in_out":
					blending_weight = (1 - math.cos(t * math.pi)) / 2
				elif timing == "ease_in":
					blending_weight = math.sin(t * math.pi / 2)
				elif timing == "ease_out":
					blending_weight = 1 - math.cos(t * math.pi / 2)
				else:
					blending_weight = t

				blending_weight = min(max(blending_weight, 0.0), 1.0)
				images1.append(current_image)
				images2.append(next_image)
				blending_weights.append(blending_weight)
				transition_counter += 1
				if transition_counter >= transition_frames:
					in_transition = False
					current_image = next_image
			else:
				blending_weight = 0.0
				images1.append(current_image)
				images2.append(current_image)
				blending_weights.append(blending_weight)

		# Now, blending_weights correspond to image_2
		blending_weights = [round(w, 3) for w in blending_weights]
		weights_invert = [round(1.0 - w, 3) for w in blending_weights]

		# Convert lists to tensors
		images1 = torch.stack(images1)
		images2 = torch.stack(images2)
		blending_weights_tensor = torch.tensor(blending_weights, dtype=images1.dtype).view(-1, 1, 1, 1)

		# Ensure blending weights are compatible with image dimensions
		blending_weights_tensor = blending_weights_tensor.to(images1.device)

		# Generate visualization
		try:
			figsize = 12.0
			plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
			plt.plot(range(len(weights_normalized)), weights_normalized, label='Audio Weights', color='blue', alpha=0.5)
			plt.scatter(peaks, weights_normalized[peaks], color='red', label='Detected Peaks')
			indices_array = np.array(indices, dtype=np.float32)
			max_index = max(indices) if indices else 1
			indices_normalized = indices_array / (max_index if max_index > 0 else 1)
			plt.step(range(len(indices)), indices_normalized, where='post', label='Images Switch', color='green')
			plt.xlabel('Frame Number')
			plt.ylabel('Normalized Values')
			plt.title('Audio Weights and Detected Peaks')
			plt.legend()
			plt.grid(True)

			# Ensure x-axis labels are integers
			ax = plt.gca()
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))

			with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
				plt.savefig(tmpfile.name, format='png')
				tmpfile_path = tmpfile.name
			plt.close()

			visualization = Image.open(tmpfile_path).convert("RGB")
			visualization = np.array(visualization).astype(np.float32) / 255.0
			visualization = torch.from_numpy(visualization).unsqueeze(0)  # Shape: [1, H, W, C]

		except Exception as e:
			print(f"Error in creating visualization: {e}")
			visualization = None

		# Return values with weights and images as originally
		return images2, blending_weights, images1, weights_invert, visualization
