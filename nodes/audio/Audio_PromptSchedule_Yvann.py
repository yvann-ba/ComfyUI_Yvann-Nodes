from ... import Yvann

class AudioNodeBase(Yvann):
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"

class Audio_PromptSchedule_Yvann(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_index": ("FLOAT", {"forceInput": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_schedule",)
    FUNCTION = "create_prompt_schedule"

    def create_prompt_schedule(self, prompt_index, prompt=""):
        # Ensure prompt_index is a list of integers (frame indices)
        if isinstance(prompt_index, float) or isinstance(prompt_index, int):
            prompt_index = [int(prompt_index)]
        else:
            prompt_index = [int(idx) for idx in prompt_index]

        # Parse the prompts, split by newline, and remove empty lines
        prompt_list = [p.strip() for p in prompt.split("\n") if p.strip() != ""]

        # Ensure the number of prompts matches the number of indices
        num_indices = len(prompt_index)
        num_prompts = len(prompt_list)

        if num_prompts > num_indices:
            # Truncate prompts if there are more prompts than indices
            prompt_list = prompt_list[:num_indices]
        elif num_prompts < num_indices:
            # Extend prompts by repeating the last prompt if fewer prompts than indices
            prompt_list += [prompt_list[-1]] * (num_indices - num_prompts)

        # Create the formatted prompt schedule string
        out = ""
        for idx, frame in enumerate(prompt_index):
            out += f"\"{frame}\": \"{prompt_list[idx]}\",\n"

        return (out,)
