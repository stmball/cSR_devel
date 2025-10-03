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

OUTFILE=sorted/quickUpdate6.xml

# ~~~~~~~~~~ Import endnote xmls to temp file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.examples.common.preprocess_data \
	   --format endnote_xml \
	   --train_N_paths \
	   data/raw/COMET/negative/COMET_originalSR_M.xml \
	   data/raw/COMET/negative/COMET_update1_M.xml \
	   data/raw/COMET/negative/COMET_update2_M.xml \
	   data/raw/COMET/negative/COMET_update3_M.xml \
	   data/raw/COMET/negative/COMET_update4_M.xml \
	   --train_Y_paths \
	   data/raw/COMET/positive/COMET_originalSR_Y.xml \
	   data/raw/COMET/positive/COMET_update1_Y.xml \
	   data/raw/COMET/positive/COMET_update2_Y.xml \
	   data/raw/COMET/positive/COMET_update3_Y.xml \
	   data/raw/COMET/positive/COMET_update4_Y.xml \
	   --test_paths \
	   data/raw/COMET/COMET_update6-deduplicated.xml \
	   --out_path \
	   /tmp/"$TMPDIR"/data.json
fi

# ~~~~~~~~~~ Train from json temp file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Train --data /tmp/"$TMPDIR"/data.json \
	   --sparse \
	   --pipeline data/param/pipeline/sparse_text_trivial.yaml \
	   --classifier data/param/classifier/SVM_RBF.yaml \
	   --train_on split=train \
	   --test_on  split=test \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

# ~~~~~~~~~~ Print top 20 from sorted json file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Data \
	   --inspect              `# Human readable output` \
	   --input /tmp/"$TMPDIR"/out.json* \
	   --sort confidence-     `# Sort by inverse score` \
	   --col confidence title `# Only output confidence (score) and title` \
	   --select 20            `# Limit to top 20 results`
fi

# ~~~~~~~~~~ Use sorted temp file to rearrange ~~~~~~~~~~
# ~~~~~~~~~~   the order of the original file  ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.examples.common.sort_endnote_xml \
	   --data data/raw/COMET/COMET_update6-deduplicated.xml \
	   --order /tmp/"$TMPDIR"/out.json* \
	   --output "$OUTFILE" `# Write to new file`
fi

# ~~~~~~~~~~ Print top 20 from sorted xml file ~~~~~~~~~~
if [ $? -eq 0 ]; then
    echo "Writing results to '$OUTFILE'..."
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
