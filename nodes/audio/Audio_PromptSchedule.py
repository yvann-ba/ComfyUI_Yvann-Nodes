from ... import Yvann

class AudioNodeBase(Yvann):
	CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Audio_PromptSchedule(AudioNodeBase):
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"audio_switch_weights": ("INT", {"forceInput": True}),
				"prompts": ("STRING", {"default": "", "multiline": True}),
			}
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("prompt_schedule",)
	FUNCTION = "create_prompt_schedule"

	def create_prompt_schedule(self, audio_switch_weights, prompts=""):
		switch_index = audio_switch_weights
		if isinstance(switch_index, int):
			switch_index = [int(switch_index)]
		else:
			switch_index = [int(idx) for idx in switch_index]

		# Parse the prompts, split by newline, and remove empty lines
		prompts_list = [p.strip() for p in prompts.split("\n") if p.strip() != ""]

		# Ensure the number of prompts matches the number of indices by looping prompts
		num_indices = len(switch_index)
		num_prompts = len(prompts_list)

		if num_prompts > 0:
			# Loop prompts to match the number of indices
			extended_prompts = []
			while len(extended_prompts) < num_indices:
				for p in prompts_list:
					extended_prompts.append(p)
					if len(extended_prompts) == num_indices:
						break
			prompts_list = extended_prompts
		else:
			# If no prompts provided, fill with empty strings
			prompts_list = [""] * num_indices

		# Create the formatted prompt schedule string
		out = ""
		for idx, frame in enumerate(switch_index):
			out += f"\"{frame}\": \"{prompts_list[idx]}\",\n"

		return (out,)
