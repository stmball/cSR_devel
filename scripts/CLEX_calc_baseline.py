
import os
import argparse
import numpy

from bin.Data import DataStream

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--data',             required = True,
                    help = 'Input Datastream file')
parser.add_argument('--output',           required = True, type = str,
                    help = 'Filepath to output')
parser.add_argument('--truth',            required = True, type = str,
                    help = 'Filepath to gold standard')
args = parser.parse_args()

data  = DataStream.parse(open(args.data).read())
truth = DataStream.parse(open(args.truth).read())
pos_set = set()
neg_set = set()
for line in truth:
    if line.label == 'Y':
        pos_set.add(line.text)
    else:
        neg_set.add(line.text)
data.add_column('confidence')
for line in data:
    if line.text in pos_set:
        line.confidence = 1
    else:
        line.confidence = 0

with open(args.output, 'w') as out:
    out.write(data.marshall())
