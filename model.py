import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class CausalModel:

    chat_delimiter = '###'

    def __init__(self, model_name, cache_dir=None, eight_bit_mode=False):
        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            print("warning: torch cuda support not installed. Running on the CPU.")
            print("See pytorch.org for installation details.\n")
            if eight_bit_mode:
                print("error: cannot use 8bit mode without cuda.")
                raise RuntimeError("8bit mode used without GPU acceleration")

        if has_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                                       low_memory=True)
        self.chat_delimiter_id = self.tokenize(self.chat_delimiter)[0].item()

        if not model_name:
            if has_cuda:
                model_name = "togethercomputer/GPT-JT-6B-v1"
            else:
                # Use a smaller, faster running model on CPUs
                model_name = "EleutherAI/gpt-neo-1.3B"

        if eight_bit_mode:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              eos_token_id=self.chat_delimiter_id,
                                                              cache_dir=cache_dir,
                                                              load_in_8bit=True,
                                                              device_map ="auto",
                                                              low_cpu_mem_usage=True)
        else:
            if has_cuda:
                dtype = torch.float16
            else:
                dtype = torch.float32

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


def entry():
    """
    Run input from stdin through the model. Terminate input by writing a single period on a new line.
    """

    import argparse

    parser = argparse.ArgumentParser(description='Run the model with an input and receive an output')
    parser.add_argument('--cache-dir', metavar='PATH', help='model storage location. default: HOME/~.cache')
    parser.add_argument('--max-new-tokens', default=None, type=int, metavar='N',
                        help='max tokens to generate after input')
    parser.add_argument('--model', default="togethercomputer/GPT-JT-6B-v1")
    parser.add_argument('--8bit',
                        default=False,
                        dest='eight_bit',
                        help='use half the memory by converting weights to 8-bit numbers.'
                             'Requires the bitsandbytes library.',
                        action='store_true')

    args = parser.parse_args()

    model = CausalModel(args.model, args.cache_dir, eight_bit_mode=args.eight_bit)

    while True:
        print("ready for input:", file=sys.stderr)
        lines = []
        while True:
            line = input()
            if line != '.':
                lines.append(line)
            else:
                print("\nresult:\n", file=sys.stderr)
                break
        try:
            print(model.generate_text('\n'.join(lines), max_new_tokens=args.max_new_tokens))
        except RuntimeError as e:
            print(e, file=sys.stderr)
        print('\n', file=sys.stderr)


if __name__ == '__main__':
    entry()
