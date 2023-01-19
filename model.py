from transformers import AutoTokenizer, AutoModelForCausalLM, BlenderbotForConditionalGeneration
import torch


class CausalModel:

    chat_delimiter = '###'

    def __init__(self, model_name="togethercomputer/GPT-JT-6B-v1", cache_dir=None):
        if not torch.cuda.is_available():
            raise RuntimeError("torch cuda support not installed. See pytorch.org for installation details.")

        self.device = torch.device('cuda:0')
        dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                                       low_memory=True)

        self.chat_delimiter_id = self.tokenize(self.chat_delimiter)[0].item()

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     eos_token_id=self.chat_delimiter_id,
                                                     cache_dir=cache_dir,
                                                     torch_dtype=dtype,
                                                     low_cpu_mem_usage=True)

        self.model = model.to(self.device, dtype=dtype)

    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

    def generate(self, input_ids, temperature=None, max_new_tokens=None):
        if not temperature:
            temperature = 0.9
        if not max_new_tokens:
            max_new_tokens = 128
        return self.model.generate(input_ids,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   do_sample=True,
                                   temperature=temperature,
                                   max_new_tokens=max_new_tokens,
                                   )

    def generate_text(self, prompt, temperature=None, max_new_tokens=None):
        """ The whole flow. Reads a prompt and outputs generated text. """
        if not temperature:
            temperature = 0.9
        if not max_new_tokens:
            max_new_tokens = 128
        input_ids = self.tokenize(prompt)
        gen_tokens = self.generate(input_ids, temperature, max_new_tokens)
        return self.tokenizer.batch_decode(gen_tokens)[0]


class BlenderbotModel:
    def __init__(self, model_name="facebook/blenderbot-400M-distill", cache_dir=None):
        if not torch.cuda.is_available():
            raise RuntimeError("torch cuda support not installed. See pytorch.org for installation details.")

        self.device = torch.device('cuda:0')
        dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                                       low_memory=True)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name,
                                                                   cache_dir=cache_dir,
                                                                   torch_dtype=dtype,
                                                                   encoder_no_repeat_ngram_size=0,
                                                                   low_cpu_mem_usage=True)

        self.model = model.to(self.device, dtype=dtype)

    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

    def generate(self, input_ids, temperature=None, max_new_tokens=None):
        if not temperature:
            temperature = 0.9
        if not max_new_tokens:
            max_new_tokens = 128
        return self.model.generate(input_ids,
                                   temperature=temperature,
                                   max_new_tokens=max_new_tokens,
                                   )

    def generate_text(self, prompt, temperature=None, max_new_tokens=None):
        """ The whole flow. Reads a prompt and outputs generated text. """
        if not temperature:
            temperature = 0.9
        if not max_new_tokens:
            max_new_tokens = 128
        input_ids = self.tokenize(prompt)
        gen_tokens = self.generate(input_ids, temperature, max_new_tokens)
        return self.tokenizer.batch_decode(gen_tokens)[0]


def entry():
    """
    Run input from stdin through the model. Terminate input by writing a single period on a new line.
    """

    import argparse

    parser = argparse.ArgumentParser(description='Run the model with an input and receive an output')
    parser.add_argument('--cache-dir', default='HOME/.cache', metavar='PATH', help='model storage location')
    parser.add_argument('--max-new-tokens', default=None, type=int, metavar='N',
                        help='max tokens to generate after input')
    parser.add_argument('--model', default="togethercomputer/GPT-JT-6B-v1")

    args = parser.parse_args()

    model = CausalModel(args.model, args.cache_dir)

    while True:
        print("ready for input:")
        lines = []
        while True:
            line = input()
            if line != '.':
                lines.append(line)
            else:
                print("\nresult:\n")
                break
        print(model.generate_text('\n'.join(lines), max_new_tokens=args.max_new_tokens))
        print('\n')


if __name__ == '__main__':
    entry()
