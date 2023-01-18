import cmd
import re

from datetime import datetime


class ChatContext:
    """ This class formats inputs in our chatbot format and extracts the results.

     It supports all the functions of the chatbot including history and parsing the preamble format.
     """

    history = []
    variable_re = re.compile(r'///(\w+): *(.*)')

    def __init__(self, preamble, username=None, botname=None, enable_history=None):
        self.intro_message = None
        self.username = username
        self.botname = botname
        self.enable_history = enable_history
        self.set_preamble(preamble)

    def set_preamble(self, preamble):
        processed = []
        for line in preamble.splitlines():
            stripped = line.strip()
            if stripped[:3] == '///':
                match = self.variable_re.match(stripped)
                key, value = match.group(1, 2)
                if key == 'INTRO':
                    self.intro_message = value
                if key == 'USERNAME' and not self.username:
                    self.username = value
                if key == 'BOTNAME' and not self.botname:
                    self.botname = value
                if key == 'HISTORY' and self.enable_history is None:
                    self.enable_history = value.lower() == 'true'
            if stripped[:2] == '//':
                continue
            processed.append(line)

        # If settings were not read from the preamble, set default values.
        if self.username is None:
            self.username = 'user'
        if self.botname is None:
            self.botname = 'chatbot'
        if self.enable_history is None:
            self.enable_history = True

        self.preamble = '\n'.join(processed).format(username=self.username, botname=self.botname)

    def make_prompt(self, line):
        """ create a complete prompt to send to the model """
        return f'{self.preamble}\n{self._history_as_text()}{self.username}: {line}\n{self.botname}:'

    def add_history(self, line, response):
        if self.enable_history:
            self.history.append((line, response.strip()))

    def extract_response(self, prompt, generated):
        """ Given the original prompt and the generated output, returns the first response. """
        generated = generated.removeprefix(prompt)

        end = generated.find('###')
        if end is None or end <= 2:
            return generated
        return generated[:end].strip()

    def _history_as_text(self):
        text = ''
        for (i, o) in self.history:
            text += f"{self.username}: {i}\n"
            text += f"{self.botname}: {o}\n"
            text += '###\n'
        return text


class GPTShell(cmd.Cmd):
    debug_level = 0

    temperature = 0.9
    max_new_tokens = 128
    last_generation = ''

    def __init__(self, model, preamble, username=None, botname=None, enable_history=None):
        self.chat_context = ChatContext(preamble, username, botname, enable_history)
        self.model = model
        self.intro = self.chat_context.intro_message
        self.prompt = f'[{self.chat_context.username}]: '

        super(GPTShell, self).__init__()

    def default(self, line):
        prompt = self.chat_context.make_prompt(line)
        start = datetime.now().timestamp()
        gen_text = self.model.generate_text(prompt,
                                            temperature=self.temperature,
                                            max_new_tokens=self.max_new_tokens)
        end = datetime.now().timestamp()
        if self.debug_level > 0:
            print(f'generation took {end - start:.2f} seconds')
        self.last_generation = gen_text
        response = self.chat_context.extract_response(prompt, gen_text)
        print(response)
        self.chat_context.add_history(line, response)

    def do_preamble(self, _line):
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        self.chat_context.set_preamble('\n'.join(lines))
        self.chat_context.history = []
        print("My preamble has been changed and history reset")

    def do_reset(self, _line):
        self.chat_context.history = []
        print("My chat history has been reset")

    def do_temperature(self, line):
        if line == '':
            print(f'current temperature: {self.temperature}')
            return

        try:
            self.temperature = float(line)
        except ValueError as e:
            print(e)

    def do_max_new_tokens(self, line):
        try:
            self.max_new_tokens = int(line)
        except ValueError as e:
            print(e)

    def do_print_last(self, _line):
        print(self.last_generation)

    def do_print_last_repr(self, _line):
        print(repr(self.last_generation))

    def do_debug(self, input):
        if len(input) == 0:
            print("set debug level:\n 0 - off\n 1 - show stats\n 2 - verbose debug info")

        try:
            self.debug_level = int(input)
        except ValueError:
            print("the argument to this command must be an integer")

    def emptyline(self):
        pass


def entry():
    import argparse

    parser = argparse.ArgumentParser(description='Run the GPT-based chat bot')
    parser.add_argument('preamble', metavar='FILE',
                        help='path the preamble file')
    parser.add_argument('--cache-dir', default='HOME/.cache', metavar='PATH', help='model storage location')

    parser.add_argument('--username', metavar='NAME',
                        help='the name of the user which the bot will address')
    parser.add_argument('--botname', metavar='NAME', help='the bot\'s name')
    parser.add_argument('--no-history',
                        default=None,
                        help='set if you do not want the model to be given the chat history as context',
                        action='store_false')

    args = parser.parse_args()

    with open(args.preamble) as preamble_file:
        preamble = preamble_file.read()

    import model
    model = model.Model(args.cache_dir)
    GPTShell(model, preamble, args.username, args.botname, enable_history=args.no_history).cmdloop()


if __name__ == '__main__':
    entry()
