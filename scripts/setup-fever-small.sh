#!/bin/bash

mkdir -p data
mkdir -p data/fever-data

cp fever-data-small/train.jsonl data/fever-data/train.jsonl
cp fever-data-small/dev.jsonl data/fever-data/dev.jsonl
cp fever-data-small/test.jsonl data/fever-data/test.jsonl

