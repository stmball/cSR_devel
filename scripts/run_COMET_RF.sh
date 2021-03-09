#!bin/bash

python -m bin.Train --data data/full/COMET/COMET_all.json \
       --sparse \
       --pipeline data/param/pipeline/waterloo_orig.yaml \
       --classifier data/param/classifier/LogReg_default.yaml \
       --RF_mode 'Y|MN' \
       --train_on split!=update3 \
       --test_on  split=update3 \
       --output '../runs/COMET/RF_update3_cleaned_html.json' \
       --output_format 'DataStream' \
       --verbosity 10
