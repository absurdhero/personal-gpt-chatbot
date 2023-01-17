import cmd

from datetime import datetime


class ChatContext:
    """ This class formats inputs in our chatbot format and extracts the results.

     It supports all the functions of the chatbot including history and parsing the preamble format.
     """

    history = []

    def __init__(self, preamble=None, username='user', botname='chatbot'):
        self.username = username
        self.botname = botname

        if not preamble:
            preamble = """This is a discussion between a {username} and a {botname}. 
The {botname} is very nice and empathetic.

{username}: Hello nice to meet you.
{botname}: Nice to meet you too.
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
        """
        self.preamble = preamble.format(username=self.username, botname=self.botname)

    def make_prompt(self, line):
        """ create a complete prompt to send to the model """
        input_line = f"{self.username}: {line}"
        return self.preamble + self._history_as_text() + input_line + f'\n{self.botname}: '

    def add_history(self, line, response):
        self.history.append((line, response))

    def extract_response(self, prompt, generated):
        """ Given the original prompt and the generated output, returns the first response. """
        generated = generated.removeprefix(prompt)

        end = generated.find('###')
        if end is None or end <= 2:
            return generated
        return generated[:end]

    def _history_as_text(self):
        text = ''
        for (i, o) in self.history:
            text += f"{self.username}: {i}\n"
            text += f"{self.botname}: {o}\n"
            text += '###\n'
        return text


class GPTShell(cmd.Cmd):
    intro = 'Welcome!'
    debug_level = 0

    temperature = 0.9
    max_new_tokens = 128
    last_generation = ''

    def __init__(self, model, preamble=None, username='user', botname='chatbot', enable_history=True):
        self.chat_context = ChatContext(preamble, username, botname)
        self.model = model
        self.prompt = f'[{username}]:'
        self.enable_history = enable_history

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
        if self.enable_history:
            self.chat_context.add_history(line, response)

    def do_preamble(self, _line):
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        self.chat_context.preamble = '\n'.join(lines)
        self.chat_context.history = []
        print("My preamble has been changed and history reset")

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

    parser.add_argument('--username', default='user', metavar='NAME',
                        help='the name of the user which the bot will address')
    parser.add_argument('--botname', default='chatbot', metavar='NAME', help='the bot\'s name')
    parser.add_argument('--no-history',
                        help='set if you do not want the model to be given the chat history as context',
                        action='store_true')

    args = parser.parse_args()
    preamble = None
    if args.preamble:
        with open(args.preamble) as preamble_file:
            preamble = preamble_file.read()

    import model
    model = model.Model(args.cache_dir)
    GPTShell(model, preamble, args.username, args.botname, enable_history=not args.no_history).cmdloop()


if __name__ == '__main__':
    entry()
