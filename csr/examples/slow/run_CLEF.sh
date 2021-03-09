#!bin/bash

# ===============================================
# Simple example use-case for the COMET dataset
# ----------------------------------------------
# Takes data from update 3 in EndNote XML format
# with M as negative labels and Y as positive.
# Applies the model on the M in update 4 and
# exports the ordered file to sorted/out.xml
# This is intended as a demonstration and both
# the files and the model have been chosen such
# that the whole script can run in under a minute
# ===============================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"

# ~~~~~~~~~~ Train from json temp file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Train --data data/full/clef20_ALL \
	   --sparse \
	   --pipeline data/param/pipeline/sparse_text_only_3gram.yaml \
	   --classifier data/param/classifier/SGD.yaml \
	   --cv_column source `# Use column 'source' as cv folds` \
	   --metrics AP \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

# ~~~~~~~~~~ Print top 20 from sorted json file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Data \
	   --inspect                     `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
	   --sort confidence-            `# Sort by inverse score` \
	   --col confidence title source `# Only output confidence (score) and title` \
	   --select 50                   `# Limit to top 50 results`
fi

rm -r /tmp/"$TMPDIR"
