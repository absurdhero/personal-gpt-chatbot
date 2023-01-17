from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Model:
    def __init__(self, cache_dir=None):
        if not torch.cuda.is_available():
            raise RuntimeError("torch cuda support not installed. See pytorch.org for installation details.")

        self.device = torch.device('cuda:0')
        dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1", cache_dir=cache_dir,
                                                       low_memory=True)
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1",
                                                     cache_dir=cache_dir,
                                                     torch_dtype=dtype,
                                                     low_cpu_mem_usage=True)

        self.model = model.to(self.device, dtype=dtype)

    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

    def generate(self, input_ids, temperature=0.9, max_new_tokens=128):
        return self.model.generate(input_ids,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   do_sample=True,
                                   temperature=temperature,
                                   max_new_tokens=max_new_tokens,
                                   use_cache=True)

    def generate_text(self, prompt, temperature=0.9, max_new_tokens=128):
        """ The whole flow. Reads a prompt and outputs generated text. """
        input_ids = self.tokenize(prompt)
        gen_tokens = self.generate(input_ids, temperature, max_new_tokens)
        return self.tokenizer.batch_decode(gen_tokens)[0]
