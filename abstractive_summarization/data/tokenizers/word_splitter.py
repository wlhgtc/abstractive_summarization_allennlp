import jieba
import thulac
import pyltp
import os
import re
from typing import List

from overrides import overrides

from allennlp.common import Params, Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter

@WordSplitter.register('jieba')
class JieBaWordSplitter(WordSplitter):
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(t) for t in jieba.cut(sentence, cut_all=False)]

    @classmethod
    def from_params(cls, params: Params) -> 'JieBaWordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

@WordSplitter.register('thulac')
class ThuLacWordSplitter(WordSplitter):
    def __init__(self,pos_tags: bool = False) -> None:
        self._pos_tags = pos_tags
        self.thu = thulac.thulac(seg_only=self._pos_tags)
    
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(text=t[0]) if not self._pos_tags else Token(text=t[0],pos=t[1]) for t in self.thu.cut(sentence)]

    @classmethod
    def from_params(cls, params: Params) -> 'ThuLacWordSplitter':
        pos_tags = params.pop_bool('pos_tags', False)
        params.assert_empty(cls.__name__)
        return cls()
        
@WordSplitter.register('ltp')
class LtpWordSplitter(WordSplitter):
    def __init__(self,
             LTP_DATA_DIR: str = None,
             pos_tags: bool = False,
             parse: bool = False,
             ner: bool = False) -> None:
        self.seg = pyltp.Segmentor().load(os.path.join(LTP_DATA_DIR, 'cws.model'))
        if pos_tags:
            self.pos = pyltp.Postagger().load(os.path.join(LTP_DATA_DIR, 'pos.model'))
        if parse:
            self.parser = pyltp.Parser().load(os.path.join(LTP_DATA_DIR, 'parser.model'))
        if ner:
            self.ner = pyltp.NamedEntityRecognizer().load(os.path.join(LTP_DATA_DIR, 'ner.model'))
    
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        words = self.seg.segment(sentence)
        self.seg.release()
        tokens = [{'text':word} for word in words]
        if self.pos:
            postags = self.pos.postag(words)
            self.pos.release()
            tokens = [token.update({'pos_':postag}) for token,postag in zip(tokens,postags)]
        if self.parser:
            arcs = self.parser.parse(words, postags)
            self.parser.release()
            tokens = [token.update({'dep_':arc}) for token,arc in zip(tokens,arcs)]
        if self.ner:
            netags = self.ner.recognize(words, postags)
            self.ner.release()
            tokens = [token.update({'ent_type_':netag}) for token,netag in zip(tokens,netags)]
        
        return [Token(text=token['text'],pos_=token.get('pos_',None),dep_=token.get('dep_',None),ent_type_=token.get('ent_type_',None)) for token in tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'LtpWordSplitter':
        LTP_DATA_DIR = params.pop_bool('LTP_DATA_DIR', None)
        pos_tags = params.pop_bool('pos_tags', False)
        parse = params.pop_bool('parse', False)
        ner = params.pop_bool('ner', False)
        params.assert_empty(cls.__name__)
        return cls()
