'''
nlpcc3 single document summarization dataset_reader inherit from seq2seq dataset_reader
'''        
from typing import Dict
import logging
import json
import re
import codecs

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = "@@START@@"
END_SYMBOL = "@@END@@"

@DatasetReader.register("nlpcc3")
class NLPCC3DatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing source_tokens(article) and 
    target_tokens(summarization), and creates a dataset suitable
    for text summarization using these examples.

    Expected format for each input line: {"article", "summarization"}

    The JSON-line could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens(article): ``TextField``
        target_tokens(summarization): ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
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
    def _read(self, file_path):
        with codecs.open(file_path, "r", encoding='utf-8') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue
                 
                try:
                    _json = json.loads(line)
                except:
                    continue
                source_string = _json['article']
                source_string = re.sub(u'<Paragraph>','',source_string)
                target_string = _json['summarization']
                target_string = re.sub(u'<Paragraph>','',target_string)
                
                if source_string == '':
                    continue
                if target_string == '':
                    yield self.text_to_instance(source_string)
                else:
                    yield self.text_to_instance(source_string, target_string)

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
    def from_params(cls, params: Params) -> 'NLPCC3DatasetReader':
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
        return NLPCC3DatasetReader(source_tokenizer, target_tokenizer,
                                    source_token_indexers, target_token_indexers,
                                    source_add_start_token, lazy, make_vocab, 
                                    max_encoding_steps)
