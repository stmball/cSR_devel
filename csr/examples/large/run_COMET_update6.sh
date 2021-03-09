#!bin/bash

# ================================================================
# Script to rank candidates for COMET update 5
# ----------------------------------------------------------------
# Takes data from updates 1-4 in EndNote XML format with MN as
# negative labels and Y as positive. Applies the model on the
# candidates in update 5 and exports the ordered file to
# sorted/out.xml
# These are the settings that were used in the 2019 update
# Note that running this script will take a fair  amount of time.
# To check that everything works as intended, it may be a good
# idea to first run the smaller examples in the 'small' folder
# ================================================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"

OUTFILE=sorted/out.xml

if [ $? -eq 0 ]; then
    echo
    echo "Importing data from EndNote XML format..."
    echo
    python -m csr.examples.common.preprocess_data \
	   --format endnote_xml \
	   --train_N_paths \
	   data/raw/COMET/COMET_update1_N.xml \
	   data/raw/COMET/COMET_update1_M.xml \
	   data/raw/COMET/COMET_update2_N.xml \
	   data/raw/COMET/COMET_update2_M.xml \
	   data/raw/COMET/COMET_update3_N.xml \
	   data/raw/COMET/COMET_update3_M.xml \
	   data/raw/COMET/COMET_update4_N.xml \
	   data/raw/COMET/COMET_update4_M.xml \
	   data/raw/COMET/COMET_update5_N.xml \
	   data/raw/COMET/COMET_update5_M.xml \
	   --train_Y_paths \
	   data/raw/COMET/COMET_update1_Y.xml \
	   data/raw/COMET/COMET_update2_Y.xml \
	   data/raw/COMET/COMET_update3_Y.xml \
	   data/raw/COMET/COMET_update4_Y.xml \
	   data/raw/COMET/COMET_update5_Y.xml \
	   --test_paths \
	   data/raw/COMET/COMET_update6-deduplicated.xml \
	   --out_path \
	   /tmp/"$TMPDIR"/data.json
fi

if [ $? -eq 0 ]; then
    echo
    echo "Train on imported data..."
    echo
    python -m csr.Train --data /tmp/"$TMPDIR"/data.json \
	   --sparse \
	   --pipeline data/param/pipeline/sparse_text_only_5gram.yaml \
	   --classifier data/param/classifier/SGD.yaml \
	   --train_on split=train \
	   --test_on  split=test \
	   --output /tmp/"$TMPDIR"/out.json \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
    echo
    echo "Data written to temp file: /tmp/$TMPDIR/out.json"
    echo
fi

if [ $? -eq 0 ]; then
    echo
    echo "Inspecting top 20 references..."
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
    python -m csr.examples.sort_endnote_xml \
	   --data data/raw/COMET/COMET_update5_N.xml \
	   --order /tmp/"$TMPDIR"/out.json* \
	   --output "$OUTFILE" `# Write to new file`
fi

if [ $? -eq 0 ]; then
    echo
    echo "Printing top 20 results from the sorted EndNote XML file..."
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
