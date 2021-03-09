#!bin/bash

# ===============================================
# Simple example use-case to classify sentences
# ----------------------------------------------
# Takes two annotated files of sentences, having
# at least the columns annotator, label, and text
# Given the data, we train a model on rows where
# annotator=Mariska, and test the results on rows
# where annotator=Rene
# ===============================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"

# Change the following to extract different data items:
datatype=1 # 1: TC, 2: IT, 3: RS

# ~~~~~~~~~~ Train from json input ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Train \
           --data data/full/CL_extraction/annotated_"$datatype"_*.json `# Use two files as input` \
	   --sparse \
	   --pipeline data/param/pipeline/sparse_terminology_with_context.yaml \
	   --classifier data/param/classifier/SGD_400epochs.yaml \
	   --train_on annotator=Mariska \
	   --test_on  annotator=Rene \
	   --metrics AP \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

# ~~~~~~~~~~ Print top 20 from sorted json file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Data --full \
	   --inspect                   `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
           --sort confidence-          `# Sort by inverse score` \
	   --col label confidence text `# Only output label, score and text` \
	   --select 20                 `# Limit to top 20 results`
fi

# ~~~~~~~~~~ Print positions of Y in sorted json file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Data \
	   --inspect                   `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
           --sort confidence-          `# Sort by inverse score` \
	   --col label confidence text `# Only output label, score and text` |\
	grep '^[YMN]' | grep '^Y' -n
fi

rm -r /tmp/"$TMPDIR"
