"""
text summarization is loosely defined as follows: given a document, extract it's important information 
and generate human readable abstaction,like one sentence or several sentence.

These submodules contain readers for things that are predominantly text summarization datasets.
"""

from abstractive_summarization.data.dataset_readers.text_summarization.nlpcc3 import NLPCC3DatasetReader