import argparse
import json
import logging
import http.client
import urllib.parse
import datetime
import time
from tqdm import tqdm
from flair.models import SequenceTagger
from flair.data import Sentence
from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERLabelSchema, FEVERGoldFormatter
from nltk.corpus import stopwords
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
#from allennlp.predictors.predictor import Predictor


# Replace the subscriptionKey string value with your valid subscription key.
subscriptionKey = '1f5276a0d951474cb672826b30098ed2'

host = 'api.cognitive.microsoft.com'
path = '/bing/v7.0/entities'


def get_suggestions(query):
    headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
    conn = http.client.HTTPSConnection(host)

    mkt = 'en-US'
    params = '?mkt=' + mkt + '&q=' + urllib.parse.quote(query)

    conn.request("GET", path + params, None, headers)
    response = conn.getresponse()
    return response.read()

class FEVERReader:
    """
    Read full text for evidence sentences from fever db
    """

    def __init__(self,
                 db: FeverDocDB) -> None:
        self.db = db
        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema())
        self.reader = JSONLineReader()


    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]

    def get_evidence_text(self, evidence):
        lines = set([self.get_doc_line(d[0], d[1]) for d in evidence])
        premise = " ".join(lines)
        return premise


def contains_word(s, w):
    return f' {w} ' in f' {s} '

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--split',type=str)
    args = parser.parse_args()

    db = FeverDocDB(args.db)
    split = args.split
    fever = FEVERReader(db)

    #delete the below model from the cache
    #predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz")
    tagger = SequenceTagger.load('ner')

    throttle_start = datetime.datetime.now()

    with open("data/fever/{0}.ns.pages.p1.jsonl".format(split),"r") as f:
        with open("data/fever/{0}.ns.ner.pages.p1.jsonl".format(split),"w+") as f2:
            for line in tqdm(f.readlines()):
                js = json.loads(line)

                fever_line = fever.formatter.format_line(js)
                evidence = fever.get_evidence_text(fever_line["evidence"])

                #tags = predictor.predict(
                 #   sentence=evidence
                #)

                evidence_ner = Sentence(evidence)
                claim_ner = Sentence(fever_line["claim"])
                # predict NER tags
                tagger.predict(evidence_ner)
                tagger.predict(claim_ner)
                evidence_tags = evidence_ner.to_dict(tag_type='ner')
                claim_tags = claim_ner.to_dict(tag_type='ner')

                missing_entity = False
                for c_entity in claim_tags["entities"]:
                    if not contains_word(evidence, c_entity["text"]):
                        missing_entity = True
                        break

                claim_tags_list = list([c["text"] for c in claim_tags["entities"]])
                js["ner_claim"] = claim_tags_list
                js["ner_evidence"] = list([c["text"] for c in evidence_tags["entities"]])
                js["ner_missing"] = missing_entity
                js["ner_related"] = []
                js["fact"] = []

                try:
                    throttle_now = datetime.datetime.now() - throttle_start
                    if throttle_now.total_seconds() < 0.25:
                        time.sleep(0.25)

                    bing_res = get_suggestions(' '.join(claim_tags_list))
                    throttle_start = datetime.datetime.now()
                    bing = json.loads(bing_res)
                    if 'entities' in bing:
                        for b_values in bing['entities']['value']:
                            for c in claim_tags_list:
                                if contains_word(b_values['description'], c):
                                    js["ner_related"].append([b_values['name'], c])

                            for w in fever_line["claim"].split():
                                if w not in claim_tags_list and w not in stopwords.words('english'):
                                    if contains_word(b_values['description'], w):
                                        js["fact"].append([b_values['name'], w])
                    else:
                        js["ner_related"] = [["None"]]

                except Exception as e:
                    print(e)

                f2.write(json.dumps(js)+"\n")
