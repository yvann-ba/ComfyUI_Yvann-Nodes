import torch
import os
import folder_paths
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Import for integer x-axis labels
import tempfile
import numpy as np
from PIL import Image
from ... import Yvann

class AudioNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Audio_Analysis(AudioNodeBase):
	analysis_modes = ["Drums Only", "Full Audio", "Vocals Only", "Bass Only", "Other Audio"]

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"batch_size": ("INT", {"forceInput": True}),
				"fps": ("FLOAT", {"forceInput": True}),
				"audio": ("AUDIO", {"forceInput": True}),
				"analysis_mode": (cls.analysis_modes,),
				"threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.01}),
				"smooth": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
				"multiply": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
				"min_range": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
				"max_range": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.1, "step": 0.01}),
			}
		}

	RETURN_TYPES = ("IMAGE", "AUDIO", "AUDIO", "FLOAT",)
	RETURN_NAMES = ("graph_audio", "processed_audio", "original_audio", "audio_weights")
	FUNCTION = "process_audio"

	def download_and_load_model(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		download_path = os.path.join(folder_paths.models_dir, "openunmix")
		os.makedirs(download_path, exist_ok=True)

		model_file = "umxl.pth"
		model_path = os.path.join(download_path, model_file)

		if not os.path.exists(model_path):
			print("Downloading umxl model...")
			separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device='cpu')
			torch.save(separator.state_dict(), model_path)
			print(f"Model saved to: {model_path}")
		else:
			print(f"Loading model from: {model_path}")
			separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device='cpu')
			separator.load_state_dict(torch.load(model_path, map_location='cpu'))

		separator = separator.to(device)
		separator.eval()

		return separator

	def _get_audio_frame(self, waveform, i, samples_per_frame):
		# Extract a single frame of audio from the waveform
		start = i * samples_per_frame
		end = start + samples_per_frame
		return waveform[..., start:end].cpu().numpy().squeeze()

	def _rms_energy(self, waveform, batch_size, samples_per_frame):
		# Calculate the RMS energy for each audio frame
		try:
			return np.array([round(np.sqrt(np.mean(self._get_audio_frame(waveform, i, samples_per_frame)**2)), 4) for i in range(batch_size)])
		except Exception as e:
			print(f"Error in RMS energy calculation: {e}")
			return np.zeros(batch_size)

	def _apply_audio_processing(self, weights, threshold, smooth, multiply):
		# Normalize weights to 0-1 range
		weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) - np.min(weights) > 0 else weights

		# Apply threshold
		weights = np.where(weights > threshold, weights, 0)

		# Apply smoothing
		smoothed = np.zeros_like(weights)
		for i in range(len(weights)):
			if i == 0:
				smoothed[i] = weights[i]
			else:
				smoothed[i] = smoothed[i-1] * smooth + weights[i] * (1 - smooth)

		# Apply multiply before rescaling
		smoothed = smoothed * multiply
		smoothed = np.clip(smoothed, 0, 1)

		return smoothed

	def _rescale_weights(self, weights, min_range, max_range):
		# Rescale weights to the specified range
		return weights * (max_range - min_range) + min_range

	def process_audio(self, audio, batch_size, fps, analysis_mode, threshold, smooth, multiply, min_range, max_range):
		# Main function to process audio and generate weights
		if audio is None or 'waveform' not in audio or 'sample_rate' not in audio:
			print("Invalid audio input")
			return None, None, None, None

		waveform = audio['waveform']
		sample_rate = audio['sample_rate']

		if not isinstance(waveform, torch.Tensor):
			print("Waveform is not a torch.Tensor")
			return None, None, None, None

		# Calculate total audio duration needed
		audio_duration = batch_size / fps
		total_samples_needed = int(audio_duration * sample_rate)

		# Crop or pad audio to match required duration
		if waveform.shape[-1] > total_samples_needed:
			waveform = waveform[..., :total_samples_needed]
		elif waveform.shape[-1] < total_samples_needed:
			pad_length = total_samples_needed - waveform.shape[-1]
			waveform = torch.nn.functional.pad(waveform, (0, pad_length))

		samples_per_frame = total_samples_needed // batch_size

		# Apply audio separation if needed
		if analysis_mode in ["Drums Only", "Vocals Only", "Bass Only", "Other Audio"]:
			try:
				model = self.download_and_load_model()
				device = next(model.parameters()).device
				waveform = waveform.to(device)
				estimates = model(waveform)
				if analysis_mode == "Drums Only":
					processed_waveform = estimates[:, 1, :, :]
				elif analysis_mode == "Vocals Only":
					processed_waveform = estimates[:, 0, :, :]
				elif analysis_mode == "Bass Only":
					processed_waveform = estimates[:, 2, :, :]
				elif analysis_mode == "Other Audio":
					processed_waveform = estimates[:, 3, :, :]
			except Exception as e:
				print(f"Error in model processing: {e}")
				return None, None, None, None
		else:  # Full Audio
			processed_waveform = waveform

		processed_audio = {
			'waveform': processed_waveform.cpu(),
			'sample_rate': sample_rate,
		}
		original_audio = {
			'waveform': waveform.cpu(),
			'sample_rate': sample_rate,
		}

		# Calculate audio weights
		audio_weights = self._rms_energy(processed_waveform.squeeze(0), batch_size, samples_per_frame)
		if np.isnan(audio_weights).any() or np.isinf(audio_weights).any():
			print("Invalid audio weights calculated")
			return None, None, None, None

		audio_weights = self._apply_audio_processing(audio_weights, threshold, smooth, multiply)
		audio_weights = self._rescale_weights(audio_weights, min_range, max_range)
		audio_weights = np.round(audio_weights, 3)

		# Generate visualization
		try:
			figsize = 12.0
			plt.figure(figsize=(figsize, figsize * 0.6), facecolor='white')
			plt.plot(list(range(1, len(audio_weights) + 1)), audio_weights, label=f'{analysis_mode} Weights', color='blue')

			plt.xlabel('Frame Number')
			plt.ylabel('Weights')
			plt.title(f'Processed Audio Weights ({analysis_mode})')
			plt.legend()
			plt.grid(True)

			# Ensure x-axis labels are integers
			ax = plt.gca()
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))

			with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
				plt.savefig(tmpfile, format='png')
				tmpfile_path = tmpfile.name
			plt.close()

			weights_graph = Image.open(tmpfile_path).convert("RGB")
			weights_graph = np.array(weights_graph)
			weights_graph = torch.tensor(weights_graph).permute(2, 0, 1).unsqueeze(0).float() / 255.0
			weights_graph = weights_graph.permute(0, 2, 3, 1)
		except Exception as e:
			print(f"Error in creating weights graph: {e}")
			weights_graph = None
			return None, None, None, None

		if processed_audio is None or audio_weights is None or weights_graph is None:
			print("One or more outputs are invalid")
			return None, None, None, None

		audio_weights = audio_weights.tolist()
		rounded_audio_weights = [round(elem, 3) for elem in audio_weights]

		return (weights_graph, processed_audio, original_audio, rounded_audio_weights)
