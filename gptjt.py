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
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1", cache_dir="D:\gpt\.cache", low_memory=True)
    #model = AutoModelForCausalLM.from_pretrained(
    #        #"togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float16, low_cpu_mem_usage=True
    #        "./GPT-JT-6B-v1", low_cpu_mem_usage=True, low_memory=True, torch_dtype=torch.float32
    #        )
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1",
            cache_dir="D:\gpt\.cache",
            torch_dtype=dtype,
            low_cpu_mem_usage=True)

    model = model.to(cuda_device, dtype=dtype)
    #pipe = pipeline('text-generation', model, tokenizer=tokenizer)

    #prompt = "hello computer, how are you?"
    #input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cuda_device)

    #gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    #gen_text = tokenizer.batch_decode(gen_tokens)[0]
    #print(gen_text)

class GPTShell(cmd.Cmd):
    intro = 'Welcome!'
    debug = False

    temperature = 0.9
    max_new_tokens = 128
    user_prefix = 'user'
    gpt_prefix = 'robot'
    preamble = ''
    history = []
    last_generation = ''

    def __init__(self, preamble = None, username='user', botname='chatbot'):
        self.prompt=f'[{username}]:'
        self.user_prefix=username
        self.gpt_prefix=botname
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
        prompt = self.preamble + self.history_as_text() + input_line + f'\n{self.gpt_prefix}: '
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cuda_device)
        print(str(datetime.now()) + " before generation")
        gen_tokens = model.generate(input_ids,
                                    pad_token_id=tokenizer.eos_token_id,
                                    #attention_mask=input_ids['attention_mask'],
                                    do_sample=True,
                                    temperature=self.temperature,
                                    max_new_tokens=self.max_new_tokens,
                                    use_cache=True)
        print(str(datetime.now()) + " after generation")
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        self.last_generation = gen_text
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

        try:
            end = generated.find('\n')
            if end <= 2:
                return generated
            return generated[:end]
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

    def do_max_new_tokens(self, line):
        try:
            self.max_new_tokens = int(line)
        except e:
            print(e)

    def do_print_last(self, input):
        print(self.last_generation)
 
    def do_debug(self, input):
        self.debug = bool(input)
        
    def emptyline(self):
        pass


def entry():
    import argparse

    parser = argparse.ArgumentParser(description='Run the GPT-based chat bot')
    parser.add_argument('--username', default='user', metavar='NAME', help='the name of the user which the bot will address')
    parser.add_argument('--botname', default='chatbot', metavar='NAME', help='the bot\'s name')
    parser.add_argument('preamble', metavar='FILE',
                        help='path the preamble file')

    args = parser.parse_args()
    preamble = None
    if args.preamble:
        with open(args.preamble) as preamble_file:
            preamble = preamble_file.read()

    initialize_model()
    GPTShell(preamble, args.username, args.botname).cmdloop()

if __name__ == '__main__':
    entry()
