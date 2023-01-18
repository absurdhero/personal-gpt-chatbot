# ML-powered Chat Bot

Uses community-driven GPT models and the
[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
library to power a chatbot.

## Getting Started

### System Requirements

By default, it uses the very large GPT-JT-6B-v1 model which is 12GB in size.

It requires 32GB of RAM and a minimum of 16GB of VRAM on the GPU (but 
only tested with 24GB).

Smaller causal language models may also work but have not been tested.

With a smaller model and some small code changes, this chatbot could run on 
lower powered machines with less memory and no GPU.

### Installation

The project depends on the following open-source packages:

Python 3 - Available in popular package managers or directly from python.org.

PyTorch - Installation instructions at pytorch.org.
Select the CUDA 11.7 installation option.


### Running for the First Time

#### Downloading the Model

The first time you run it, it will download the 12GB model and some other files
to the `.cache` directory in your home directory.

If you don't like that location or want to store these large files on 
another disk, run this program with `--cache-dir=<your-path>` to set your 
own location.

#### Giving it a personality

When running the program, give it a preamble that shows it how to behave.

Preambles are located in the `preambles` directory, but you can load your own
from anywhere.

#### Running:

Change to the directory of this project and run it:

Windows (Command Prompt)
```commandline
python3.exe cli.py preambles\default.txt
```

Linux
```commandline
python3 cli.py preambles/default.txt
```

#### Interactive Commands

In addition to chatting, you can run some special commands to modify the
personality of the chatbot or see what it is doing.

Commands:
 - reset - clears the chat history so you can begin a new conversation
 - temperature - Controls how creative it is. Higher values make it act very 
   creative but a little crazy. Low values make it emulate what it has seen 
   before very closely. It plays it safe.
   - It defaults to 0.9 and must be a value 
     between 0 - 1.0. 0.9 is a good compromise. 0.5 will be a bit boring. 0.
     97 is likely to make up some fascinating things but it could go 
     disturbingly wrong.
 - max_new_tokens (default 128) - The most new words it will generate per 
   interaction.
 - print_last - a debugging tool that prints out the previous generation in 
   its entirety so you can see the bot's complete input and response.
 - debug - Set to 1 to print performance metrics. Set to 2 for more verbose 
   output when developing.

## Writing Preambles

GPT-based chatbots operate by reading a whole chat history so far and then
adding on to it. When you first start a session, it needs an example of how
it should be chatting. That's the preamble.

A good preamble may start with a brief description of the interaction between
the user and the bot. It then must show at least 3 examples of the chat in 
action.

We have developed a particular format that the bot speaks. It looks like
a chat log but certain characters must be used to clearly indicate where each
message and response starts and ends.

#### A Full Example
```
// These lines just explains what this preamble is about.
// The bot will not see them because they start with //
//
// The lines below defines the introductory message when
// a user first starts a session and the names of each party.
//
///INTRO: Welcome! What would you like to talk about?
///BOTNAME: chatbot
///USERNAME: user
This is a discussion between a {username} and a {botname}.
The {botname} is very nice and empathetic.

{username}: Hi. Who are you?
{botname}: I'm {botname}. Nice to meet you.
###
{username}: How is it going today?
{botname}: Not so bad, thank you! How about you?
###
{username}: I am ok, but I am a bit sad...
{botname}: Oh? Why is that?
###
{username}: I caught a cold and couldn't go out to a special dinner with my friends.
{botname}: I'm so sorry to hear that. I hope you feel better soon. Do you want to talk about it?
###
```

### The Format

`{username}` and `{botname}` are replaced by the names of the user and the bot
as described at the top of the file. They can also be customized by the user 
when running the program.

#### Anatomy of a single chat

```
user: What the user types
chatbot: What the bot responds with.
###
```

Every back-and-forth interaction is formatted just like this. The model picks
up on this pattern and repeats it when it generates a reply. The rest of the
program expects the model to output in this format so it's very important to 
follow it exactly.
