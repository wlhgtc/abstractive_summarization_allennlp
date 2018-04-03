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

import re
end_tokens = ['.','?','!']
URL_PATTERN=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
HASHTAG_PATTERN = re.compile(r'#\w*')
MENTION_PATTERN = re.compile(r'@\w*')
RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)')
try:
    # UCS-4
    EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
except re.error:
    # UCS-2
    EMOJIS_PATTERN = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)

def preproccess_tweet(tweet):
    text = tweet['text']
    text = RESERVED_WORDS_PATTERN.sub('',text)
    text = MENTION_PATTERN.sub('',text)
    text = URL_PATTERN.sub('',text)
    text = EMOJIS_PATTERN.sub('',text)
    text = SMILEYS_PATTERN.sub('',text)
    text = re.sub('\.\.\.','.',text)
    text = text.strip()
    text = text if text[-1] in end_tokens else text+'.'
    tweet['text'] = text

@DatasetReader.register("news_event_tweet")
class NETDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

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
                if len(event['title']) and len(event['title']) == 0:
                    continue
                tweets = event_json['events']
                [preproccess_tweet(tweet) for tweet in tweets]
                tweet_texts = ' '.join([tweet['text'] for tweet in tweets])
                description = event.get('description',None)
                title = event.get('title',None)
                category = event.get('category',None)
                yield self.text_to_instance(tweet_texts,description,title,category)

    @overrides
    def text_to_instance(self, tweet_texts: str,description: str = None,title: str = None, category: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_tweet_texts = self._tokenizer.tokenize(tweet_texts)
        tweet_texts_field = TextField(tokenized_tweet_texts, self._token_indexers)
        fields = {'tweet_texts': tweet_texts_field}
        if description:
            tokenized_description = self._tokenizer.tokenize(description)
            description_field = TextField(tokenized_description, self._token_indexers)
            fields.update({'description':description_field})
        if title:
            tokenized_title = self._tokenizer.tokenize(title)
            title_field = TextField(tokenized_title, self._token_indexers)
            fields.update({'title':tokenized_title})
        if category is not None:
            fieldsupdate({'label':LabelField(category)})
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'NETDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)
