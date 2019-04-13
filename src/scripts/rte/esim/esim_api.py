import argparse
import flask
from flask import request
from scripts.rte.esim.eval_esim import EvalEsim
import json
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

app = flask.Flask(__name__)
app.config["DEBUG"] = True

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
esim = EvalEsim(args)


@app.route('/', methods=['GET'])
def home():
    return "send a post request to /predict"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # global result
        result = esim.eval_instance(data["evidence"], data["claim"])
    except Exception as e:
        # global result
        result = "error occurred: " + str(e)
    return json.dumps(result)


app.run(host="localhost", port=8000)

# example usage:
# PYTHONPATH=src python src/scripts/rte/esim/esim_api.py data/fever/fever.db data/models/esim_fake_science_v2/esim.tar.gz data/fever/dev.ns.pages.p1.jsonl