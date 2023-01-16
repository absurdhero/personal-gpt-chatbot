import cmd
import re

from datetime import datetime

def initialize_model():
    global cuda_device
    global tokenizer
    global model

    from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM, pipeline, Conversation
    import torch

    if not torch.cuda.is_available():
        print("torch cuda support not installed. See pytorch.org for installation details.")
        exit(1)

    cuda_device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1", cache_dir="D:\gpt\.cache", low_memory=True)
    #model = AutoModelForCausalLM.from_pretrained(
    #        #"togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float16, low_cpu_mem_usage=True
    #        "./GPT-JT-6B-v1", low_cpu_mem_usage=True, low_memory=True, torch_dtype=torch.float32
    #        )
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1",
            cache_dir="D:\gpt\.cache",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True)

    model = model.to(cuda_device, dtype=torch.float16)
    #pipe = pipeline('text-generation', model, tokenizer=tokenizer)

    #prompt = "hello computer, how are you?"
    #input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cuda_device)

    #gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    #gen_text = tokenizer.batch_decode(gen_tokens)[0]
    #print(gen_text)

class GPTShell(cmd.Cmd):
    intro = 'Welcome!'
    prompt = '[user]: '
    debug = False

    temperature = 0.9
    max_length = 512 
    user_prefix = '[user]'
    gpt_prefix = '[robot]'
    preamble = ''
    history = []

    def __init__(self, preamble = None):
        super(GPTShell, self).__init__()
        if preamble:
            self.preamble = preamble
        else:
            user_prefix = self.user_prefix
            gpt_prefix = self.gpt_prefix
            self.preamble = """This is a discussion between a {user_prefix} and a {gpt_prefix}. 
The {gpt_prefix} is very nice and empathetic.

{user_prefix}: Hello nice to meet you.
{gpt_prefix}: Nice to meet you too.
###
{user_prefix}: How is it going today?
{gpt_prefix}: Not so bad, thank you! How about you?
###
{user_prefix}: I am ok, but I am a bit sad...
{gpt_prefix}: Oh? Why is that?
###
{user_prefix}: I caught a cold and couldn't go out to a special dinner with my friends.
{gpt_prefix}: I'm so sorry to hear that. I hope you feel better soon. Do you want to talk about it?
###
   """ 
        self.preamble = self.preamble.format(user_prefix=self.user_prefix, gpt_prefix=self.gpt_prefix)
        

    def default(self, line):
        input_line = f"{self.user_prefix}: {line}"
        if len(input_line) >= self.max_length - 256:
            self.max_length = min(2048, self.max_length + 256) 
        prompt = self.preamble + self.history_as_text() + input_line
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cuda_device)
        gen_tokens = model.generate(input_ids,
                                    pad_token_id=tokenizer.eos_token_id,
                                    #attention_mask=input_ids['attention_mask'],
                                    do_sample=True,
                                    temperature=self.temperature,
                                    max_length=self.max_length)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        response = self.extract_response(gen_text, prompt)
        print(response)
        self.history.append((input_line, response))

    def history_as_text(self):
        text = ''
        for (i, o) in self.history:
            text += f"{self.user_prefix}: {i}\n"
            text += f"{self.gpt_prefix}: {o}\n"
            text += '###\n'
        return text

    def extract_response(self, generated, prompt):
        generated = generated.removeprefix(prompt)

        if self.debug:
            print("full text response:\n" + generated)
            return

        try:
            start = generated.find(f'{self.gpt_prefix}:')  + len(self.gpt_prefix) + 1
            end = generated.find('\n', start)
            if end - start <= 2:
                return generated
            return generated[start:end]
        except:
            return generated
            

    def do_preamble(self, line):
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        preamble = '\n'.join(lines)
        self.history = []
        print("My preamble has been changed and history reset")

    def do_temperature(self, line):
        try:
            self.temperature = float(line)
        except e:
            print(e)

    def do_max_length(self, line):
        try:
            self.max_length = int(line)
        except e:
            print(e)

    def do_debug(self, input):
        self.debug = bool(input)
        
    def emptyline(self):
        pass


def entry():
    import argparse

    parser = argparse.ArgumentParser(description='Run the GPT-based chat bot')
    parser.add_argument('preamble', metavar='FILE',
                        help='path the preamble file')

    args = parser.parse_args()
    preamble = None
    if args.preamble:
        with open(args.preamble) as preamble_file:
            preamble = preamble_file.read()

    initialize_model()
    GPTShell(preamble).cmdloop()

if __name__ == '__main__':
    entry()
