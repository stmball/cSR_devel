
import argparse
import sys, random, re
import collections

from csr.Data import DataStream
from csr import Import
from csr.ML.pipeline import Tokenizer
import nltk
from nltk.corpus import stopwords
import ftfy

# ~~~~~~ Arguments ~~~~~~

parser = argparse.ArgumentParser(
    description = 'Generic data processing suitable for most screening context. Takes training data with negative and positive examples separated into different files. Duplicates (if any) are removed from negative training data only.')
parser.add_argument('--train_N_paths',
                    nargs = '+',
                    required = True,
                    help = 'Negative examples'
)
parser.add_argument('--train_Y_paths',
                    nargs = '+',
                    required = True,
                    help = 'Positive examples'
)
parser.add_argument('--test_paths',
                    nargs = '+',
                    required = True,
                    help = 'Data to apply the trained model on'
)
parser.add_argument('--out_path',
                    required = True,
                    help = 'Path to save the data to'
)
parser.add_argument('--format',
                    choices = [
                        'bibreview',
                        'endnote_xml',
                        'embase_xml',
                        'RIS',
                        'pmids',
                        'tsv',
                        'csv'
                    ],
                    required = True,
                    help = 'Input format'
)
args = parser.parse_args(sys.argv[1:])

# ~~~~~~ Function Definitions ~~~~~~

noop_sent_tokenize = lambda x: [x]
noop               = lambda x:  x
tokenize = Tokenizer(sent_tokenize = noop_sent_tokenize,
                     word_tokenize = nltk.word_tokenize,
                     stem = noop,
                     stop_words = stopwords.words('english'))
def normalize(text):
    print("Pre-normalization: '%s'" % text)
    n_text = text
    n_text = ftfy.fix_text(n_text)
    n_text = re.sub('[^A-z ]', '', n_text)
    n_text = " ".join(tokenize(n_text.lower()))
    print("Post-normalization: '%s'" % n_text)
    return n_text

def make_descriptor(row):
    return (normalize(row.journal), row.year, normalize(row.title))
def make_descriptors(data):
    return [make_descriptor(row) for row in data]

def assert_no_duplicates(data):
    descs = make_descriptors(data)
    desc_to_ref = {}
    for desc, row in zip(descs, data):
        if not desc in desc_to_ref: desc_to_ref[desc] = []
        desc_to_ref[desc].append(row)
    duplicates = [(desc, count) for desc, count in collections.Counter(descs).items() if count > 1]
    assert duplicates == [], "%i/%i duplicates in set: \n%s" % (len(duplicates),
                                                                len(descs),
                        '\n'.join(["Count %i: " % count + '\n'.join([str((ref.journal, ref.year, ref.title)) for ref in desc_to_ref[desc]]) for desc, count in duplicates]))

def assert_is_subset(data_sub, data_sup):
    descs_sup = set(make_descriptors(data_sup))
    descs_sub = set(make_descriptors(data_sub))
    sub_descs_not_in_sup = descs_sub - descs_sup
    is_subset = len(sub_descs_not_in_sup) == 0
    if not is_subset:
        print("%i references from other sources:\n%s" % (
            len(sub_descs_not_in_sup),
            '\n'.join(map(str, sub_descs_not_in_sup))
        ))

def check_in_set_and_add(items, item):
    prev_len = len(items)
    items.add(item)
    return len(items) == prev_len

def seen_already(items):
    return lambda row: check_in_set_and_add(items, make_descriptor(row))

# ~~~~~~ Data handling ~~~~~~

if args.format == 'bibreview':
    import_handler = Import.import_bibreview_xml
elif args.format == 'endnote_xml':
    import_handler = Import.import_endnote_xml
elif args.format == 'embase_xml':
    import_handler = Import.import_embase_xml
elif args.format == 'RIS':
    import_handler = Import.import_RIS
elif args.format == 'pmids':
    import_handler = Import.import_pmids
elif args.format == 'tsv':
    import_handler = Import.import_tsv
elif args.format == 'csv':
    import_handler = Import.import_csv
# else should not happen

# Set of document descriptors for duplicate tracking
prev_descs = set([])

# Load Y first so that duplicates are removed from the
# negative examples
# We don't know how many columns are in the data before we
# load so we must construct this from the first data we load
train_Y = None
for path in args.train_Y_paths:
    records = import_handler(open(path, 'rb').read())
    records.delete_if(seen_already(prev_descs))
    if not train_Y: train_Y = records
    else:           train_Y.merge(records)
train_Y.label = ['Y']*len(train_Y)

train_N = DataStream(*train_Y.header)
for path in args.train_N_paths:
    records = import_handler(open(path, 'rb').read())
    records.delete_if(seen_already(prev_descs))
    train_N.merge(records)
train_N.label = ['N']*len(train_N)

train = train_Y
train.merge(train_N)
assert_no_duplicates(train)

# Reset duplicate tracking information
prev_descs = set([]) # <= comment out to remove duplicates across train/test set

test = DataStream(*train.header)
for path in args.test_paths:
    records = import_handler(open(path, 'rb').read())
    records.delete_if(seen_already(prev_descs))
    test.merge(records)

train.add_column('split', 'train')
test.add_column('split', 'test')

# Check no overlap between train and test
# This check should normally succeed when evaluating performance,
# but does not matter when we are only applying the model on new data
#assert(len(set(make_descriptors(train)).intersection(set(make_descriptors(test)))) == 0)

data = train
data.merge(test)
print("Preprocessed %i data rows" % len(data))
with open(args.out_path, 'w') as out:
    out.write(data.marshall())

print("Data written to path: '%s'" % args.out_path)
