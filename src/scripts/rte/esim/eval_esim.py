import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
#from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.esim.reader import FEVERReader
from tqdm import tqdm
import argparse
import logging
import sys
import json
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# PYTHONPATH=src python src/scripts/rte/esim/eval_esim.py data/fever/fever.db data/models/esim.tar.gz data/fever/dev.ns.pages.p1.jsonl

def eval_model(db: FeverDocDB, args) -> Model:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    reader = FEVERReader(db,
                                 sentence_level=ds_params.pop("sentence_level",False),
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=FEVERReader.custom_dict_from_params(ds_params.pop('token_indexers', {})),
                                 ner_facts=args.ner_facts
                         )

    logger.info("Reading training data from %s", args.in_file)
    data = reader.read(args.in_file)

    actual = []
    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")

    for item in tqdm(data):
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
        else:
            prediction = model.forward_on_instance(item)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]

        if "label" in item.fields:
            actual.append(item.fields["label"].label)
            if args.ner_missing is not None:
                if args.ner_missing == 'oracle' and item.fields["label"].label == "NOT ENOUGH INFO" and cls != "NOT ENOUGH INFO":
                    if item.fields["metadata"].metadata["ner_missing"]:
                        cls = "NOT ENOUGH INFO"

                if args.ner_missing == 'oracle' and item.fields["label"].label == "SUPPORTS" and cls != "SUPPORTS":
                    if item.fields["metadata"].metadata["ner_missing"]:
                        cls = "SUPPORTS"

                if args.ner_missing == 'oracle' and item.fields["label"].label == "REFUTES" and cls != "REFUTES":
                    if item.fields["metadata"].metadata["ner_missing"]:
                        cls = "REFUTES"

                if args.ner_missing == 'naive' and cls == 'SUPPORTS':
                    if item.fields["metadata"].metadata["ner_missing"]:
                        highest = np.argmax(prediction["label_probs"])
                        lowest = np.argmin(prediction["label_probs"])
                        copy = []
                        for pred in prediction["label_probs"]:
                            copy.append(pred)

                        copy[highest] = prediction["label_probs"][lowest]

                        original_logits =  prediction["label_logits"][highest]
                        chosen_logits = prediction["label_logits"][np.argmax(copy)]
                        difference_logits = original_logits - chosen_logits

                        if difference_logits < 3.0:
                            cls = model.vocab._index_to_token["labels"][np.argmax(copy)]

        predicted.append(cls)

        if args.log is not None:
            if "label" in item.fields:
                f.write(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")
            else:
                f.write(json.dumps({"predicted":cls})+"\n")

    if args.log is not None:
        f.close()

    if len(actual) > 0:
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted))
        print(confusion_matrix(actual, predicted))

    return model


if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('archive_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--log', required=False, default=None,  type=str, help='/path/to/saved/db.db')

    # ner based features
    parser.add_argument('--ner_facts', required=False, default=False,  type=bool, help='include ner based facts or not')
    parser.add_argument('--ner_missing', required=False, default=None, type=str, help='oracle / naive handling of named entity missing tags')

    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()
    db = FeverDocDB(args.db)
    eval_model(db,args)