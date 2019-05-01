import argparse
import flask
from flask import request
from scripts.rte.bert.bert_predict import BertPredict
import json
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

app = flask.Flask(__name__)
app.config["DEBUG"] = True

parser = argparse.ArgumentParser()

## Bert Required parameters
parser.add_argument("--bert_model", type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--bert_model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--cache_dir",
                    default="",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length",
                    default=512,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--testset',
                    type=bool,
                    default=False,
                    help="get predictions on test dataset")


args = parser.parse_args()
bert = BertPredict(args)


@app.route('/', methods=['GET'])
def home():
    return "send a post request to /predict"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # global result
        result = bert.predict_one(data["evidence"], data["claim"])
    except Exception as e:
        # global result
        result = "error occurred: " + str(e)
    return json.dumps(result)


app.run(host="localhost", port=8000)
