from unittest import TestCase
from cli import ChatContext


class TestChatContext(TestCase):
    def test_parse_preamble(self):
        context = ChatContext("//A comment\n"
                              "///INTRO:Hello\n"
                              "Preamble body bot:{botname} user:{username}", 'user', 'chatbot')
        self.assertEqual('Preamble body bot:chatbot user:user', context.preamble)
        self.assertEqual('Hello', context.intro_message)

    def test_set_name_in_preamble(self):
        context = ChatContext("///USERNAME:foo\n"
                              "///BOTNAME:baz\n"
                              "///INTRO:Hello\n"
                              "Preamble body bot:{botname} user:{username}")
        self.assertEqual('Preamble body bot:baz user:foo', context.preamble)
        self.assertEqual('Hello', context.intro_message)

    def test_set_name_with_override(self):
        context = ChatContext("///USERNAME: foo\n"
                              "///BOTNAME: baz\n"
                              "///INTRO:Hello\n"
                              "Preamble body bot:{botname} user:{username}", botname='override')
        self.assertEqual('Preamble body bot:override user:foo', context.preamble)
        self.assertEqual('Hello', context.intro_message)

    def test_no_history(self):
        context = ChatContext("///USERNAME: foo\n"
                              "///BOTNAME: baz\n"
                              "///HISTORY: False\n"
                              "Preamble body bot:{botname} user:{username}\n", botname='override')
        self.assertEqual('Preamble body bot:override user:foo', context.preamble)
        self.assertEqual(False, context.enable_history)

    def test_set_history(self):
        context = ChatContext("///USERNAME: foo\n"
                              "///BOTNAME: baz\n"
                              "///HISTORY: True\n"
                              "Preamble body bot:{botname} user:{username}\n", botname='override')
        self.assertEqual('Preamble body bot:override user:foo', context.preamble)
        self.assertEqual(True, context.enable_history)