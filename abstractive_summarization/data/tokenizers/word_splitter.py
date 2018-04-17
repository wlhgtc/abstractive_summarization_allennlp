import jieba
import thulac
#import pyltp
import os
import re
import json
import xmltodict
import requests
from typing import List

from overrides import overrides

from allennlp.common import Params, Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter

##zh-nlp-service
#docker run -ti -p 7000:12345 lixiepeng/lxp:ltp /ltp_server --last-stage all

# 字段名	含义
# s	输入字符串，在xml选项x为n的时候，代表输入句子；为y时代表输入xml
# x	用以指明是否使用xml
# t	用以指明分析目标，t可以为分词（ws）,词性标注（pos），命名实体识别（ner），依存句法分析（dp），语义角色标注（srl）或者全部任务（all）

class Ltp:
    
    def __init__(self, host='127.0.0.1', port='7000'):
        self.host = host
        self.port = port

    def annotate(self, doc,x='xml',t='all'):
        self.server_url = 'http://'+self.host+':'+self.port+'/ltp'
        data = {
            's': doc,
            'x': 'n',
            't': 'all'}
        try:
            res = requests.post(self.server_url,
                                data=data, 
                                headers={'Connection': 'close'})
            return json.dumps(xmltodict.parse(res.content))
        except Exception as e:
            print(e)
            
#docker run -p 9000:9000 lixiepeng/lxp:corenlp 
#java -mx6g -cp * edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000
class StanfordCoreNLP:
    '''
    Wrapper for Starford Corenlp Restful API
    annotators:"truecase,tokenize,ssplit,pos,lemma,ner,regexner,parse,depparse,openie,coref,kbp,sentiment"
    nlp = StanfordCoreNLP()
    output = nlp.annotate(text, properties={ 'annotators':,outputFormat': 'json',})
    '''

    def __init__(self, host='127.0.0.1', port='9000'):
        self.host = host
        self.port = port

    def annotate(self, data, properties=None, lang='en'):
        self.server_url = 'http://'+self.host+':'+self.port
        properties['outputFormat'] = 'json'
        try:
            res = requests.post(self.server_url,
                                params={'properties': str(properties),
                                        'pipelineLanguage':lang},
                                data=data, 
                                headers={'Connection': 'close'})
            return res.json()
        except Exception as e:
            print(e)
            
# docker run -ti --name thulac -p 8000:8000 lixiepeng/lxp:thulac hug -f thulac-service.py
########################################


@WordSplitter.register('jieba')
class JieBaWordSplitter(WordSplitter):
    @overrides
    def split_words(self, doc: str) -> List[Token]:
        return [Token(t) for t in jieba.cut(doc, cut_all=False)]

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
    def split_words(self, doc: str) -> List[Token]:
        return [Token(text=t[0]) if not self._pos_tags else Token(text=t[0],pos=t[1]) for t in  self.thu.fast_cut(doc)]

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
    def split_words(self, doc: str) -> List[Token]:
        words = self.seg.segment(doc)
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

@WordSplitter.register('ltp-server')
class LtpServerWordSplitter(WordSplitter):
    def __init__(self):
        self.ltp = Ltp()
        
    @overrides
    def split_words(self, doc: str) -> List[Token]:
        return [Token(t) for t in self.ltp.annotate(doc=doc,t='ws')]

    @classmethod
    def from_params(cls, params: Params) -> 'LtpServerWordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

@WordSplitter.register('thulac-server')
class ThulacServerWordSplitter(WordSplitter):
    @overrides
    def split_words(self, doc: str) -> List[Token]:
        #tokens = requests.get("http://127.0.0.1:8000/thulac?text=%s"%doc).json()
        tokens = requests.post("http://127.0.0.1:8000/thulac",data={'text':doc}).json()
        return [Token(t) for t in tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'ThulacServerWordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

@WordSplitter.register('corenlp-server')
class CorenlpServerWordSplitter(WordSplitter):
    def __init__(self):
        self.snlp = StanfordCoreNLP()
        
    @overrides
    def split_words(self, doc: str) -> List[Token]:
        return [Token(t) for t in self.snlp.annotate(doc,properties={'annotators':'tokenize'},lang='zh')]

    @classmethod
    def from_params(cls, params: Params) -> 'CorenlpServerWordSplitter':
        params.assert_empty(cls.__name__)
        return cls()