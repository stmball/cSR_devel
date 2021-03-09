#!bin/bash

# ===================================================================
# Example script applying a static model to the CLEF dataset
# -------------------------------------------------------------------
# Takes data in json format and trains a logistic regression model.
# Using the specified pipeline, features are extracted from columns
# 'abstract' and 'title'. The data includes the different CLEF topics
# in the 'source' column, which is used for cross validation (i.e.
# the system uses 20-fold cross-validation with the 2017 CLEF data)
# ===================================================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"

# ~~~~~~~~~~ Train from json temp file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    echo
    echo "Training on annotated data..."
    echo
    python -m csr.Train --data data/full/clef20_ALL \
	   --sparse \
	   --pipeline csr/examples/common/pipelines/sparse_abstract_title_keywords_trivial.yaml \
	   --classifier csr/examples/common/classifiers/SGD_50epochs.yaml \
	   --cv_column source `# Use column 'source' as cv folds` \
	   --metrics AP P@0.5 R@0.5 F@0.5 \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

if [ $? -eq 0 ]; then
    echo
    echo "Inspecting top 20 results..."
    echo
    python -m csr.Data \
	   --inspect                     `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
	   --sort confidence-            `# Sort by inverse score` \
	   --col confidence title source `# Only output confidence (score) and title` \
	   --select 50                   `# Limit to top 50 results`
fi

rm -r /tmp/"$TMPDIR"
