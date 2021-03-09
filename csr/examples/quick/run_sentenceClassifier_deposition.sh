#!bin/bash

# ===============================================
# Simple example use-case to classify sentences
# ----------------------------------------------
# data deposition sentence set
# data MUST to have fields text, label
# ===============================================

# Generate random directory name
TMPDIR=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 24 ; echo ''`
mkdir /tmp/"$TMPDIR"


# ~~~~~~~~~~ Train from json input ~~~~~~~~~~
if [ $? -eq 0 ]; then
    python -m csr.Train \
           --data ~/commun-vbox/Documents/corpus/DataDepositionNCBI/DepositionDataSets2019/deposit_trainingAll_testSVM66_YN.json ~/commun-vbox/Documents/corpus/DataDepositionNCBI/DepositionDataSets2019/deposit_testBayes_YN.json \
	   --sparse \
	   --pipeline data/param/pipeline/sparse_terminology_with_context.yaml \
	   --classifier data/param/classifier/SGD_400epochs.yaml \
	   --train_on split=train \
	   --test_on  split=test \
	   --metrics AP \
	   --output ~/commun-vbox/Documents/corpus/DataDepositionNCBI/DepositionDataSets2019/deposit_trainingAll_YN_out.json  \
	   --output_format DataStream `# JSON output` \
	   --verbosity 6
fi

# ~~~~~~~~~~ Print top 20 from sorted json file ~~~~~~~~~~
#if [ $? -eq 0 ]; then
#    python -m csr.Data --full \
#	   --inspect                   `# Human readable output` \
#	   --input /tmp/"$TMPDIR"/out.json* \
#           --sort confidence-          `# Sort by inverse score` \
#	   --col label confidence text `# Only output label, score and text` \
#fi

# ~~~~~~~~~~ Print positions of Y in sorted json file ~~~~~~~~~~
#if [ $? -eq 0 ]; then
#    python -m csr.Data \
#	   --inspect                   `# Human readable output` \
#	   --input /tmp/"$TMPDIR"/out.json* \
#           --sort confidence-          `# Sort by inverse score` \
#	   --col label confidence text `# Only output label, score and text` |\
#	grep '^[YMN]' | grep '^Y' -n
#fi

#rm -r /tmp/"$TMPDIR"
