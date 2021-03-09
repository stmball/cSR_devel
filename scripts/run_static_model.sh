#!bin/bash



python -m bin.temp_apply_static_model --data data/full/clef2018_ALL \
       --sparse \
       --pipeline data/param/pipeline/waterloo_pipeline.yaml \
       --classifier data/param/classifier/LogReg_default.yaml \
       --static_models ../models/CLEF2017/clef2017_sparse_SGD_train_split_noUS\[1\] \
       --RF_mode 'Y|MN' \
       --train_on split=seed  split!=test \
       --test_on  split!=seed split=test \
       --output_format 'TREC' \
       --verbosity 10
