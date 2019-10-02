from nltk.data import load
from nltk.tokenize import _treebank_word_tokenizer


class NewTokenizer:
    def __init__(self, language='english'):
        self.tokenizer = load('tokenizers/punkt/{0}.pickle'.format(language))
        self.treebank_word_tokenizer = _treebank_word_tokenizer

    def sent_tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def word_tokenize(self, text, preserve_line=False):
        sentences = [text] if preserve_line else self.sent_tokenize(text)
        return [
            token for sent in sentences for token in self.treebank_word_tokenizer.tokenize(sent)
        ]
