
sources_2018 = ["CD008122",
		"CD008587",
		"CD008759",
		"CD008892",
		"CD009175",
		"CD009263",
		"CD009694",
		"CD010213",
		"CD010296",
		"CD010502",
		"CD010657",
		"CD010680",
		"CD010864",
		"CD011053",
		"CD011126",
		"CD011420",
		"CD011431",
		"CD011515",
		"CD011602",
		"CD011686",
		"CD011912",
		"CD011926",
		"CD012009",
		"CD012010",
		"CD012083",
		"CD012165",
		"CD012179",
		"CD012216",
		"CD012281",
		"CD012599"]

for s in sources_2018:
	 with open("../runs/CLEF/20180508/waterloo_orig_B_%s[rep 1]" % s, 'r') as input_file:
                  with open("../runs/CLEF/submission2018/cnrs_RF_unigram_%s" % s, 'w') as output_file:
                           for line in input_file:
                                    tokens = line.strip().split()
                                    topic, interaction, pmid, rank, score, run_id = tokens
                                    threshold = "0"
                                    run_id = "cnrs.RF.unigram"
                                    tokens = [topic, threshold, pmid, rank, score, run_id]
                                    output_line = " ".join(tokens)
                                    output_file.write("%s\n" % output_line)
                           
	 with open("../runs/CLEF/20180508/waterloo_pipeline_B_%s[rep 1]" % s, 'r') as input_file:
                  with open("../runs/CLEF/submission2018/cnrs_RF_bigram_%s" % s, 'w') as output_file:
                           for line in input_file:
                                    tokens = line.strip().split()
                                    topic, interaction, pmid, rank, score, run_id = tokens
                                    threshold = "0"
                                    run_id = "cnrs.RF.bigram"
                                    tokens = [topic, threshold, pmid, rank, score, run_id]
                                    output_line = " ".join(tokens)
                                    output_file.write("%s\n" % output_line)
                           

	 with open("../runs/CLEF/20180508/meta_B_%s[rep 1]" % s, 'r') as input_file:
                  with open("../runs/CLEF/submission2018/cnrs_combined_%s" % s, 'w') as output_file:
                           for line in input_file:
                                    tokens = line.strip().split()
                                    topic, interaction, pmid, rank, score, run_id = tokens
                                    threshold = "0"
                                    run_id = "cnrs.combined"
                                    tokens = [topic, threshold, pmid, rank, score, run_id]
                                    output_line = " ".join(tokens)
                                    output_file.write("%s\n" % output_line)
                           
