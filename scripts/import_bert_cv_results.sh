#export BERT_BASE_DIR=pretrained/uncased_L-12_H-768_A-12
export BERT_BASE_DIR=pretrained/pubmed_pmc_470k
export CLEX_DIR=data/CLEX/

declare -a reviews=('CD007394'
		    'CD007427'
		    'CD008054'
		    'CD008081'
		    'CD008760'
		    'CD008782'
		    'CD008892'
		    'CD009372'
		    'CD009647'
		    'CD010339'
		    'CD010360'
		    'CD010653'
		    'CD010705'
		    'CD011420')
declare -a annotators=('Mariska'
		       'Rene')

for annotator in "${annotators[@]}"
do
    for d in 1 2 3
    do
	for review in "${reviews[@]}"
	do
	    python -m bin.Import \
		   --type tsv \
    --input ../BERT/bert/data/CLEX/annotated_"$d"_dev_"${annotator}"_"${review}"_out.tsv \
   --output ../BERT/bert/data/CLEX/annotated_"$d"_dev_"${annotator}"_"${review}"_out.json
	    
	    python -m bin.Import \
		   --type tsv \
    --input ../BERT/bert/data/CLEX/annotated_UMLS_"$d"_dev_"${annotator}"_"${review}"_out.tsv \
   --output ../BERT/bert/data/CLEX/annotated_UMLS_"$d"_dev_"${annotator}"_"${review}"_out.json
	    
	done
    done
done
