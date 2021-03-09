#!bin/bash

PIPELINE="data/param/pipeline/sparse_titles_only.yaml"
CLASSIFIER="data/param/classifier/SGD.yaml"
VERBOSITY="6"
OUTPUT_NAME="../runs/COMET/static_amended_comp_train_on_titles"
INPUT_NAME="data/full/COMET/COMET_all_v3.json"

# ~~~~~~~~~~~~ Original CV ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --cv_train_on split=original \
       --test_on split=original abstract=_ \
       --cv_column cv_split \
       --output "${OUTPUT_NAME}_original_cv.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 1 ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split=original \
       --test_on  split=update1 abstract=_ \
       --output "${OUTPUT_NAME}_update1.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 1 CV ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split=original \
       --cv_train_on split=update1 \
       --test_on  split=update1 abstract=_ \
       --cv_column cv_split \
       --output "${OUTPUT_NAME}_update1_cv.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 2 ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split!=update2 split!=update3 split!=update4 \
       --test_on  split=update2 abstract=_ \
       --output "${OUTPUT_NAME}_update2.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 2 CV ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split!=update2 split!=update3 split!=update4 \
       --cv_train_on split=update2 \
       --test_on  split=update2 abstract=_ \
       --cv_column cv_split \
       --output "${OUTPUT_NAME}_update2_cv.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 3 ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split!=update3 split!=update4 \
       --test_on  split=update3 abstract=_ \
       --output "${OUTPUT_NAME}_update3.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 3 CV ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split!=update3 split!=update4 \
       --cv_train_on split=update3 \
       --test_on  split=update3 abstract=_ \
       --cv_column cv_split \
       --output "${OUTPUT_NAME}_update3_cv.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 4 ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split!=update4 \
       --test_on  split=update4 abstract=_ \
       --output "${OUTPUT_NAME}_update4.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY

# ~~~~~~~~~~~~ Update 4 CV ~~~~~~~~~~~~
python -m bin.Train --data $INPUT_NAME \
       --sparse \
       --pipeline $PIPELINE \
       --classifier $CLASSIFIER \
       --RF_mode 'none' \
       --train_on split!=update4 \
       --cv_train_on split=update4 \
       --test_on  split=update4 abstract=_ \
       --cv_column cv_split \
       --output "${OUTPUT_NAME}_update4_cv.json" \
       --output_format 'DataStream' \
       --verbosity $VERBOSITY
