import jieba
import re
from typing import List

from overrides import overrides

from allennlp.common import Params, Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter


@WordSplitter.register('jieba')
class JieBaWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.  We use a somewhat odd name here to avoid coming too close to the more
    commonly used ``SpacyWordSplitter``.

    Note that we use ``sentence.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """
    def __init__(self):
        self.para_pattern = re.compile('<Paragraph>')
    
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        sentence = self.para_pattern.sub(' ',sentence)
        return [Token(t) for t in jieba.cut(sentence, cut_all=False)]

    @classmethod
    def from_params(cls, params: Params) -> 'JieBaWordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

