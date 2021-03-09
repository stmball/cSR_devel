
import os
import argparse
import numpy

from bin.Data import DataStream

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--data',             required = True,
                    help = 'Input Datastream file')
parser.add_argument('--col',              required = True, type = str,
                    help = 'Column to use for splitting')
parser.add_argument('--train_out',        required = True, type = str,
                    help = 'Filepath to output training set')
parser.add_argument('--dev_out',          required = True, type = str,
                    help = 'Filepath to output development set')
#parser.add_argument('--test_out',         required = True, type = str,
#                    help = 'Filepath to output test set')
args = parser.parse_args()

data = DataStream.parse(open(args.data).read())

labels = data[args.col]
labels = numpy.unique(labels)

#print("Input labels:\n%s" % '\n'.join(labels))

#ii = set(numpy.random.choice(range(len(labels)), len(labels)//2, replace=False))
#train_labels  = [l for i, l in enumerate(labels) if not i in ii]
#ii = set(numpy.random.choice(range(len(train_labels)), len(train_labels)//2, replace=False))
#dev_labels    = set([l for i, l in enumerate(train_labels) if i in ii])
#train_labels  = set([l for i, l in enumerate(train_labels) if not i in ii])

reviews = [
    'CD007394',
    'CD007427',
    'CD008054',
    'CD008081',
    'CD008760',
    'CD008782',
    'CD008892',
    'CD009372',
    'CD009647',
    'CD010339',
    'CD010360',
    'CD010653',
    'CD010705',
    'CD011420'
]
annotators = [
    'Mariska',
    'Rene'
]

for annotator in annotators:
    for review in reviews:
        train_labels = [r for r in reviews if not r == review]
        dev_labels   = [review]
        #print("Split train:\n%s" % '\n'.join(train_labels))
        #print("Split dev:\n%s" % '\n'.join(dev_labels))
        
        train_set = DataStream(*data.header)
        dev_set   = DataStream(*data.header)
        for row in data:
            if row.annotator == annotator:
                if row[args.col] in dev_labels:
                    dev_set.append([row[x] for x in data.header])
                elif row[args.col] in train_labels:
                    train_set.append([row[x] for x in data.header])
        if len(train_set) == 0 and len(dev_set) == 0: continue

        train_filename, train_ext = os.path.splitext(args.train_out)
        with open("%s_%s_%s%s" % (train_filename, annotator, review, train_ext), 'w') as out:
            out.write(train_set.marshall())
        dev_filename, dev_ext = os.path.splitext(args.dev_out)
        with open("%s_%s_%s%s" % (dev_filename, annotator, review, dev_ext), 'w') as out:
            out.write(dev_set.marshall())
