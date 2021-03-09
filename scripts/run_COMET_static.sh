#!bin/bash

#python -m bin.Train --data data/full/COMET/COMET_4_only_cv.json \
#       --sparse \
#       --pipeline data/param/pipeline/sparse_text_only.yaml \
#       --classifier data/param/classifier/SGD.yaml \
#       --RF_mode 'none' \
#       --cv_column cv_split \
#       --output '../runs/COMET/static_text_only_update4_only_2018_recheck_duplicates_cv.json' \
#       --output_format 'DataStream' \
#       --verbosity 10

python -m bin.Train --data data/full/COMET/COMET_all_4_recheck_duplicates.json \
       --sparse \
       --pipeline data/param/pipeline/sparse_text_only.yaml \
       --classifier data/param/classifier/SGD.yaml \
       --RF_mode 'none' \
       --train_on split=original \
       --test_on  split=update2 \
       --output '../runs/COMET/static_text_only_update2_2019_recheck_duplicates.json' \
       --output_format 'DataStream' \
       --verbosity 10

python -m bin.Train --data data/full/COMET/COMET_all_4_cv.json \
       --sparse \
       --pipeline data/param/pipeline/sparse_text_only.yaml \
       --classifier data/param/classifier/SGD.yaml \
       --RF_mode 'none' \
       --cv_column cv_split \
       --output '../runs/COMET/static_text_only_update4_2019_recheck_duplicates_cv.json' \
       --output_format 'DataStream' \
       --verbosity 10

python -m bin.Train --data data/full/COMET/COMET_all_4_recheck_duplicates.json \
       --sparse \
       --pipeline data/param/pipeline/sparse_text_only.yaml \
       --classifier data/param/classifier/SGD.yaml \
       --RF_mode 'none' \
       --train_on split=original \
       --test_on  split=update1 \
       --output '../runs/COMET/static_text_only_update1_2019_recheck_duplicates.json' \
       --output_format 'DataStream' \
       --verbosity 10

python -m bin.Train --data data/full/COMET/COMET_all_4_recheck_duplicates.json \
       --sparse \
       --pipeline data/param/pipeline/sparse_text_only.yaml \
       --classifier data/param/classifier/SGD.yaml \
       --RF_mode 'none' \
       --train_on split=original \
       --test_on  split=update3 \
       --output '../runs/COMET/static_text_only_update3_2019_recheck_duplicates.json' \
       --output_format 'DataStream' \
       --verbosity 10

python -m bin.Train --data data/full/COMET/COMET_all_4_recheck_duplicates.json \
       --sparse \
       --pipeline data/param/pipeline/sparse_text_only.yaml \
       --classifier data/param/classifier/SGD.yaml \
       --RF_mode 'none' \
       --train_on split!=update4 \
       --test_on  split=update4 \
       --output '../runs/COMET/static_text_only_update4_2019_recheck_duplicates.json' \
       --output_format 'DataStream' \
       --verbosity 10
