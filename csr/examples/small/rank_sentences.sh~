#!bin/bash

# ===================================================================
# Simple example use-case to classify sentences
# -------------------------------------------------------------------
# Takes two annotated files of sentences, having at least the columns
# columns 'annotator', 'label', and 'text'. Given the data, we train
# a model on rows where annotator=Mariska, and test the results on
# rows where annotator=Rene
# ===================================================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"

# Change the following to extract different data items:
# 1: target condition
# 2: index test
# 3: reference standard
datatype=1

if [ $? -eq 0 ]; then
    echo
    echo "Training on annotated data..."
    echo
    python -m csr.Train \
           --data data/full/CL_extraction/annotated_"$datatype"_*.json `# Use two files as input` \
	   --sparse \
	   --pipeline csr/examples/pipelines/sparse_terminology_with_context.yaml \
	   --classifier csr/examples/common/classifiers/SGD_400epochs.yaml \
	   --train_on annotator=Mariska \
	   --test_on  annotator=Rene \
	   --metrics AP P@10 R@10 F@10 P@0.5 R@0.5 F@0.5 \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

if [ $? -eq 0 ]; then
    echo
    echo "Inspecting top 20 extracted sentences..."
    echo
    python -m csr.Data --full \
	   --inspect                   `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
           --sort confidence-          `# Sort by inverse score` \
	   --col label confidence text `# Only output label, score and text` \
	   --select 20                 `# Limit to top 20 results`
fi

if [ $? -eq 0 ]; then
    echo
    echo "Printing the positions of the positive labels in the sorted data..."
    echo
    python -m csr.Data \
	   --inspect                   `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
           --sort confidence-          `# Sort by inverse score` \
	   --col label confidence text `# Only output label, score and text` |\
	grep '^[YMN]' | grep '^Y' -n
fi

rm -r /tmp/"$TMPDIR"
