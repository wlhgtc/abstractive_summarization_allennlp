from typing import List, Dict, Iterable
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = "@@START@@"
END_SYMBOL = "@@END@@"

# import re
# end_tokens = ['.','?','!']
# URL_PATTERN=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
# HASHTAG_PATTERN = re.compile(r'#\w*')
# MENTION_PATTERN = re.compile(r'@\w*')
# RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)')
# try:
    #UCS-4
    # EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
# except re.error:
    #UCS-2
    # EMOJIS_PATTERN = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
# SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)

# def preproccess_tweet(tweet):
    # text = tweet['text']
    # text = RESERVED_WORDS_PATTERN.sub('',text)
    # text = MENTION_PATTERN.sub('',text)
    # text = URL_PATTERN.sub('',text)
    # text = EMOJIS_PATTERN.sub('',text)
    # text = SMILEYS_PATTERN.sub('',text)
    # text = re.sub('\.\.\.','.',text)
    # text = text.strip()
    # text = text if text[-1] in end_tokens else text+'.'
    # tweet['text'] = text

@DatasetReader.register("news_event_tweet")
class NETDatasetReader(DatasetReader):
    def __init__(self,
             source_tokenizer: Tokenizer = None,
             target_tokenizer: Tokenizer = None,
             source_token_indexers: Dict[str, TokenIndexer] = None,
             target_token_indexers: Dict[str, TokenIndexer] = None,
             source_add_start_token: bool = True,
             lazy: bool = False,
             make_vocab: bool = False,
             max_encoding_steps: int = 1000) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._make_vocab = make_vocab
        self._max_encoding_steps = max_encoding_steps

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            for line_num, line in enumerate(tqdm.tqdm(data_file.readlines())):
                line = line.strip("\n")
                if not line:
                    continue
                try:
                    event_json = json.loads(line)
                except:
                    continue
                event = event_json['event']
                #if len(event['title']) and len(event['title']) == 0:
                 #   continue
                tweets = event_json['tweets']
                #[preproccess_tweet(tweet) for tweet in tweets]
                #tweets = sorted(tweets,key=lambda tweet:tweet['boe_cosine'],reverse=True)
                tweet_texts = ' '.join([tweet['text'] for tweet in tweets])
                description = event.get('description',None)
                #title = event.get('title',None)
                #category = event.get('category',None)
                yield self.text_to_instance(tweet_texts,description)#,title,category)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._make_vocab is False:
            tokenized_source = tokenized_source[:self._max_encoding_steps]
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})

    @classmethod
    def from_params(cls, params: Params) -> 'NETDatasetReader':
        source_tokenizer_type = params.pop('source_tokenizer', None)
        source_tokenizer = None if source_tokenizer_type is None else Tokenizer.from_params(source_tokenizer_type)
        target_tokenizer_type = params.pop('target_tokenizer', None)
        target_tokenizer = None if target_tokenizer_type is None else Tokenizer.from_params(target_tokenizer_type)
        source_indexers_type = params.pop('source_token_indexers', None)
        source_add_start_token = params.pop_bool('source_add_start_token', True)
        if source_indexers_type is None:
            source_token_indexers = None
        else:
            source_token_indexers = TokenIndexer.dict_from_params(source_indexers_type)
        target_indexers_type = params.pop('target_token_indexers', None)
        if target_indexers_type is None:
            target_token_indexers = None
        else:
            target_token_indexers = TokenIndexer.dict_from_params(target_indexers_type)
        lazy = params.pop('lazy', False)
        make_vocab = params.pop_bool('make_vocab', False)
        max_encoding_steps = params.pop('max_encoding_steps', 1000)
        params.assert_empty(cls.__name__)
        return NETDatasetReader(source_tokenizer, target_tokenizer,
                                    source_token_indexers, target_token_indexers,
                                    source_add_start_token, lazy, make_vocab, 
                                    max_encoding_steps)
