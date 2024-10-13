import torch
import os
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import math
from PIL import Image
from scipy.signal import find_peaks
import random
from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Audio_Reactive_IPAdapter_Yvann(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_weights": ("FLOAT", {"forceInput": True}),
                "images": ("IMAGE", {"forceInput": True}),
                "timing": (["custom", "linear", "ease_in_out", "ease_in", "ease_out", "random"], {"default": "linear"}),
                "transition_frames": ("INT", {"default": 10, "min": 1, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distance": ("INT", {"default": 1, "min": 1, "step": 1}),
                "prominence": ("FLOAT", {"default": 0.1, "min": 0.0,  "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("weights", "weights_invert", "image_1", "image_2", "visualization")
    FUNCTION = "process_weights"

    def process_weights(self, audio_weights, images, timing, transition_frames, threshold, distance, prominence):
        import random

        if not isinstance(audio_weights, list) and not isinstance(audio_weights, np.ndarray):
            print("Invalid audio_weights input")
            return None, None, None, None, None, None, None

        if images is None or not isinstance(images, torch.Tensor):
            print("Invalid or empty images input")
            return None, None, None, None, None, None, None

        # Normalize audio_weights
        audio_weights = np.array(audio_weights)
        weights_range = np.max(audio_weights) - np.min(audio_weights)
        weights_normalized = (audio_weights - np.min(audio_weights)) / weights_range if weights_range > 0 else audio_weights

        # Detect peaks
        peaks, _ = find_peaks(weights_normalized, height=threshold, distance=distance, prominence=prominence)

        # Generate indices based on peaks
        indices = []
        index_value = 0
        peak_set = set(peaks)
        for i in range(len(weights_normalized)):
            if i in peak_set:
                index_value += 1
            indices.append(index_value)

        total_frames = len(indices)

        # Use images in the order they are provided
        # images is a batch of images: tensor of shape [B, H, W, C]
        if len(images.shape) == 3:
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
                elif timing == "random":
                    blending_weight = random.uniform(0, 1)
                else:  # custom or default
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

        blending_weights = [round(w, 3) for w in blending_weights]
        weights_invert = [round(1.0 - w, 3) for w in blending_weights]
        
        # Convert lists to tensors
        images1 = torch.stack(images1)
        images2 = torch.stack(images2)
        blending_weights_tensor = torch.tensor(blending_weights).view(-1, 1, 1, 1)

        # Ensure blending weights are compatible with image dimensions
        blending_weights_tensor = blending_weights_tensor.to(images1.device).type_as(images1)

        # Generate visualization
        try:
            figsize = 12.0
            plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
            plt.plot(range(len(weights_normalized)), weights_normalized, label='Audio Weights', color='blue', alpha=0.5)
            plt.scatter(peaks, weights_normalized[peaks], color='red', label='Detected Peaks')
            indices_normalized = np.array(indices) / max(indices if max(indices) > 0 else 1)
            plt.step(range(len(indices)), indices_normalized, where='post', label='Indices (Normalized)', color='green')
            plt.xlabel('Frame Number')
            plt.ylabel('Normalized Values')
            plt.title('Audio Weights and Detected Peaks')
            plt.legend()
            plt.grid(True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.savefig(tmpfile, format='png')
                tmpfile_path = tmpfile.name
            plt.close()

            visualization = Image.open(tmpfile_path).convert("RGB")
            visualization = np.array(visualization).astype(np.float32) / 255.0
            visualization = torch.from_numpy(visualization).unsqueeze(0)  # Shape: [1, H, W, C]

        except Exception as e:
            print(f"Error in creating visualization: {e}")
            visualization = None

        return blending_weights, weights_invert, images1, images2, visualization
