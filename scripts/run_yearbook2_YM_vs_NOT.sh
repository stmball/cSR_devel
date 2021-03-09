#!/bin/bash

FEAT="--exclude_features selection similar_articles structured_abstract "
ARGS=" --mode YM|N --RF_mode none --n_runs 1 --n_repetitions 10 --tt_split --verbosity 1 --data data/full/yearbook2_ALL "

python bin/ML.py --fold_by source=2017 --trec_output_name no_AF ${FEAT} ${ARGS}
