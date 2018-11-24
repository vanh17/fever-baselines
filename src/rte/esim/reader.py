from typing import Dict, Iterator
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.data_set import DataSet as FEVERDataSet
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fever")
class FEVERReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 db: FeverDocDB,
                 sentence_level = False,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ner_facts = False,
                 filtering: str = None) -> None:
        self._sentence_level = sentence_level
        self._ner_facts = ner_facts
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()

        self._token_indexers = token_indexers or {'elmo': ELMoTokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema(),filtering=filtering)
        self.reader = JSONLineReader()


    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:

        ds = FEVERDataSet(file_path,reader=self.reader, formatter=self.formatter)
        ds.read()

        for instance in tqdm.tqdm(ds.data):
            if instance is None:
                continue

            if not self._sentence_level:
                pages = set(ev[0] for ev in instance["evidence"])
                premise = " ".join([self.db.get_doc_text(p) for p in pages])
            else:
                lines = set([self.get_doc_line(d[0],d[1]) for d in instance['evidence']])
                premise = " ".join(lines)
                if self._ner_facts:
                    premise = premise + " ".join(instance['fact'])

            if len(premise.strip()) == 0:
                premise = ""

            hypothesis = instance["claim"]
            label = instance["label_text"]
            ner_missing = instance["ner_missing"]
            yield self.text_to_instance(premise, hypothesis, label, ner_missing)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None,
                         ner_missing = False) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._wiki_tokenizer.tokenize(premise) if premise is not None else None
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers) if premise is not None else None
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        fields['metadata'] = MetadataField({'ner_missing': ner_missing})
        return Instance(fields)

    @staticmethod
    def custom_dict_from_params(params: Params) -> 'Dict[str, TokenIndexer]':  # type: ignore
        """
        We typically use ``TokenIndexers`` in a dictionary, with each ``TokenIndexer`` getting a
        name.  The specification for this in a ``Params`` object is typically ``{"name" ->
        {indexer_params}}``.  This method reads that whole set of parameters and returns a
        dictionary suitable for use in a ``TextField``.

        Because default values for token indexers are typically handled in the calling class to
        this and are based on checking for ``None``, if there were no parameters specifying any
        token indexers in the given ``params``, we return ``None`` instead of an empty dictionary.
        """
        token_indexers = {}
        for name, indexer_params in params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)
        if token_indexers == {}:
            token_indexers = None
        return token_indexers
