import sys
import csv
from typing import Iterator, List, Text
import os
import logging
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from common.dataset.data_set import DataSet as FEVERDataSet
import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class FakeScienceData(object):
    """Processor for the fake science data set."""
    def __init__(self,
                 db: FeverDocDB,
                 filtering: str = None
                 ):
        self.db = db
        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema(), filtering=filtering)
        self.reader = JSONLineReader()

    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]

    def _read(self, file_path: str) -> Iterator[List[Text]]:

        ds = FEVERDataSet(file_path, reader=self.reader, formatter=self.formatter)
        ds.read()

        for instance in tqdm.tqdm(ds.data):
            if instance is None:
                continue

            if not self._sentence_level:
                pages = set(ev[0] for ev in instance["evidence"])
                premise = " ".join([self.db.get_doc_text(p) for p in pages])
            else:
                if instance["domain"] == "source":
                    lines = set([self.get_doc_line(d[0], d[1]) for d in instance['evidence']])
                else:
                    lines = set(ev[0] for ev in instance["evidence"])

                premise = " ".join(lines)
                if self._ner_facts:
                    premise = premise + " ".join(instance['fact'])

            if len(premise.strip()) == 0:
                premise = ""

            hypothesis = instance["claim"]
            label = instance["label_text"]

            yield [premise, hypothesis, label]

    def get_examples(self, data_path):
        """See base class."""
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(self._read(data_path))

    def get_labels(self):
        """See base class."""
        return FEVERLabelSchema()

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
