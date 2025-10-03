#!bin/bash

# ===================================================================
# Simple example use-case to rank parts of the COMET dataset
# -------------------------------------------------------------------
# Takes data from update 3 in EndNote XML format with M as negative
# labels and Y as positive. Applies the model on the M in update 4
# and exports the ordered file to sorted/out.xml
# This is intended as a demonstration and both the files and the
# model have been chosen such that the whole script can run in under
# a minute
# ===================================================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"

OUTFILE=sorted/out.xml

# ~~~~~~~~~~ Import endnote xmls to temp file ~~~~~~~~~~
# The example preprocessing script in
# csr/examples/common/preprocess_data
# takes positive and negative data examples in different
# files, combines these with the application (test) data
# and outputs the combined results into a common json
# temp file that can be used for training
if [ $? -eq 0 ]; then
    echo
    echo "Importing data to internal format..."
    echo
    python -m csr.examples.common.preprocess_data \
	   --format endnote_xml \
	   --train_N_paths \
	   data/raw/COMET/COMET_update3_M.xml \
	   --train_Y_paths \
	   data/raw/COMET/COMET_update3_Y.xml \
	   --test_paths \
	   data/raw/COMET/COMET_update4_M.xml \
	   --out_path \
	   /tmp/"$TMPDIR"/data.json
fi

if [ $? -eq 0 ]; then
    echo
    echo "Training on imported data..."
    echo
    python -m csr.Train --data /tmp/"$TMPDIR"/data.json \
	   --sparse \
	   --pipeline csr/examples/common/pipelines/sparse_abstract_title_keywords_trivial.yaml \
	   --classifier csr/examples/common/classifiers/SGD_50epochs.yaml \
	   --train_on split=train \
	   --test_on  split=test \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

if [ $? -eq 0 ]; then
    echo
    echo "Printing top 20 results from sorted data..."
    echo
    python -m csr.Data \
	   --inspect              `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
	   --sort confidence-     `# Sort by inverse score` \
	   --col confidence title `# Only output confidence (score) and title` \
	   --select 20            `# Limit to top 20 results`
fi

# Use the sorted temp (json) file to rearrange the order
# of the original file, write the results to $OUTFILE
if [ $? -eq 0 ]; then
    echo
    echo "Sorting EndNote XML file..."
    echo
    python -m csr.examples.common.sort_endnote_xml \
	   --data data/raw/COMET/COMET_update4_M.xml \
	   --order /tmp/"$TMPDIR"/out.json* \
	   --output "$OUTFILE" `# Write to new file`
fi

if [ $? -eq 0 ]; then
    echo
    echo "Printing top 20 results from sorted EndNote XML file..."
    echo
    python -m csr.Import \
	   --input "$OUTFILE" \
	   --type endnote_xml | \
    python -m csr.Data \
	   --inspect   `# Human readable output` \
	   --full      `# Do not truncate columns` \
	   --col title `# Only output title` \
	   --select 20 `# Limit to top 20 results`
fi

rm -r /tmp/"$TMPDIR"
