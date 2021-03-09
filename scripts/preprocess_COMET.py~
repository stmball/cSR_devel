
import sys, random, re
import collections

from bin.Data import DataStream
from bin import Import
from bin.ML.pipeline import Tokenizer
import nltk
from nltk.corpus import stopwords
import ftfy

N0 = DataStream.parse(open('data/full/COMET/COMET_orig_N.json').read())
M0 = DataStream.parse(open('data/full/COMET/COMET_orig_M.json').read())
Y0 = DataStream.parse(open('data/full/COMET/COMET_orig_Y.json').read())

N1 = DataStream.parse(open('data/full/COMET/COMET_update1_N.json').read())
M1 = DataStream.parse(open('data/full/COMET/COMET_update1_M.json').read())
Y1 = DataStream.parse(open('data/full/COMET/COMET_update1_Y.json').read())

N2 = DataStream.parse(open('data/full/COMET/COMET_update2_N.json').read())
M2 = DataStream.parse(open('data/full/COMET/COMET_update2_M.json').read())
Y2 = DataStream.parse(open('data/full/COMET/COMET_update2_Y.json').read())

N3 = DataStream.parse(open('data/full/COMET/COMET_update3_N.json').read())
M3 = DataStream.parse(open('data/full/COMET/COMET_update3_M.json').read())
Y3 = DataStream.parse(open('data/full/COMET/COMET_update3_Y.json').read())

N4 = DataStream.parse(open('data/full/COMET/COMET_update4_N.json').read())
M4 = DataStream.parse(open('data/full/COMET/COMET_update4_M.json').read())
Y4 = DataStream.parse(open('data/full/COMET/COMET_update4_Y.json').read())

N5 = DataStream.parse(open('data/full/COMET/COMET_update5_N.json').read())
#M5 = DataStream.parse(open('data/full/COMET/COMET_update5_M.json').read())
#Y5 = DataStream.parse(open('data/full/COMET/COMET_update5_Y.json').read())

# ~~~~~~ Definitions ~~~~~~

noop_sent_tokenize = lambda x: [x]
noop               = lambda x:  x
tokenize = Tokenizer(noop_sent_tokenize, nltk.word_tokenize, noop, stopwords.words('english'))
def normalize(text):
#    print("Pre-normalization: '%s'" % text)
    n_text = text
    n_text = ftfy.fix_text(n_text)
    n_text = re.sub('[^A-z ]', '', n_text)
    n_text = " ".join(tokenize(n_text.lower()))
#    print("Post-normalization: '%s'" % n_text)
    return n_text

def make_descriptor(row):
    return (normalize(row.journal), row.year, normalize(row.title))
#    return (row.journal, row.year, row.title)
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
#    sub_items_in_sup = [desc in descs_sup for desc in descs_sub]
#    is_subset = all(sub_items_in_sup)
#    missing_descs = [desc for desc in descs_sub if not desc in descs_sup]
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

# ~~~~~~ Definitions ~~~~~~

all_data = [N0,
            M0,
            Y0,
            N1,
            M1,
            Y1,
            N2,
            M2,
            Y2,
            N3,
            M3,
            Y3,
            N4,
            M4,
            Y4,
            N5,
#            M5,
#            Y5,
]

# ~~~~~~ Remove refs without abstracts ~~~~~~

#for X in all_data:
#    print("Current size: %i" % len(X))
#    X.delete_if(lambda row: row.abstract == "")
#    print("Current size: %i" % len(X))

# ~~~~~~ Remove duplicates ~~~~~~

for X in all_data[:-2]:
    prev_descs = set([])
    X.delete_if(seen_already(prev_descs))
    assert_no_duplicates(X)

# ~~~~~~ Translate IDs to N set IDs ~~~~~~
# This is only necessary when translating the rankings back
# to the original EndNote files

#desc_to_ID0 = dict(zip(make_descriptors(N0), N0.ID))
#M0.ID = [desc_to_ID0[desc] for desc in make_descriptors(M0)]
#Y0.ID = [desc_to_ID0[desc] for desc in make_descriptors(Y0)]

#desc_to_ID1 = dict(zip(make_descriptors(N1), N1.ID))
#M1.ID = [desc_to_ID1[desc] for desc in make_descriptors(M1)]
#Y1.ID = [desc_to_ID1[desc] for desc in make_descriptors(Y1)]

#desc_to_ID2 = dict(zip(make_descriptors(N2), N2.ID))
#M2.ID = [desc_to_ID2[desc] for desc in make_descriptors(M2)]
#Y2.ID = [desc_to_ID2[desc] for desc in make_descriptors(Y2)]

#desc_to_ID3 = dict(zip(make_descriptors(N3), N3.ID))
#M3.ID = [desc_to_ID3[desc] for desc in make_descriptors(M3)]
#Y3.ID = [desc_to_ID3[desc] for desc in make_descriptors(Y3)]

desc_to_ID4 = dict(zip(make_descriptors(N4), N4.ID))
M4.ID = [desc_to_ID4[desc] for desc in make_descriptors(M4)]
Y4.ID = [desc_to_ID4[desc] for desc in make_descriptors(Y4)]

desc_to_ID5 = dict(zip(make_descriptors(N5), N5.ID))
#M5.ID = [desc_to_ID5[desc] for desc in make_descriptors(M5)]
#Y5.ID = [desc_to_ID5[desc] for desc in make_descriptors(Y5)]

# ~~~~~~ Remove expected supersets (Y <= YM <= YMN) ~~~~~~

#assert_is_subset(M0, N0)
#assert_is_subset(Y0, N0)

prev_descs = set([])
Y0.delete_if(seen_already(prev_descs))
M0.delete_if(seen_already(prev_descs))
N0.delete_if(seen_already(prev_descs))
assert(len(set(make_descriptors(Y0)).intersection(set(make_descriptors(M0)))) == 0)
assert(len(set(make_descriptors(M0)).intersection(set(make_descriptors(N0)))) == 0)

#assert_is_subset(M1, N1)
#assert_is_subset(Y1, N1)

prev_descs = set([])
Y1.delete_if(seen_already(prev_descs))
M1.delete_if(seen_already(prev_descs))
N1.delete_if(seen_already(prev_descs))
assert(len(set(make_descriptors(Y1)).intersection(set(make_descriptors(M1)))) == 0)
assert(len(set(make_descriptors(M1)).intersection(set(make_descriptors(N1)))) == 0)

#assert_is_subset(M2, N2)
#assert_is_subset(Y2, N2)

prev_descs = set([])
Y2.delete_if(seen_already(prev_descs))
M2.delete_if(seen_already(prev_descs))
N2.delete_if(seen_already(prev_descs))
assert(len(set(make_descriptors(Y2)).intersection(set(make_descriptors(M2)))) == 0)
assert(len(set(make_descriptors(M2)).intersection(set(make_descriptors(N2)))) == 0)

#assert_is_subset(M3, N3)
#assert_is_subset(Y3, N3)

prev_descs = set([])
Y3.delete_if(seen_already(prev_descs))
M3.delete_if(seen_already(prev_descs))
N3.delete_if(seen_already(prev_descs))
assert(len(set(make_descriptors(Y3)).intersection(set(make_descriptors(M3)))) == 0)
assert(len(set(make_descriptors(M3)).intersection(set(make_descriptors(N3)))) == 0)

#assert_is_subset(M4, N4)
#assert_is_subset(Y4, N4)

prev_descs = set([])
Y4.delete_if(seen_already(prev_descs))
M4.delete_if(seen_already(prev_descs))
N4.delete_if(seen_already(prev_descs))
assert(len(set(make_descriptors(Y4)).intersection(set(make_descriptors(M4)))) == 0)
assert(len(set(make_descriptors(M4)).intersection(set(make_descriptors(N4)))) == 0)

#prev_descs = set([])
#Y5.delete_if(seen_already(prev_descs))
#M5.delete_if(seen_already(prev_descs))
#N5.delete_if(seen_already(prev_descs))
#assert(len(set(make_descriptors(Y5)).intersection(set(make_descriptors(M5)))) == 0)
#assert(len(set(make_descriptors(M5)).intersection(set(make_descriptors(N5)))) == 0)

# ~~~~~~ Add metadata ~~~~~~

for X in [N0, N1, N2, N3, N4, N5]:
    X.label = ['N'] * len(X)
for X in [M0, M1, M2, M3, M4]:
    X.label = ['M'] * len(X)
for X in [Y0, Y1, Y2, Y3, Y4]:
    X.label = ['Y'] * len(X)
for X in [N0, M0, Y0]:
    X.add_column('split', 'original')
for X in [N1, M1, Y1]:
    X.add_column('split', 'update1')
for X in [N2, M2, Y2]:
    X.add_column('split', 'update2')
for X in [N3, M3, Y3]:
    X.add_column('split', 'update3')
for X in [N4, M4, Y4]:
    X.add_column('split', 'update4')
for X in [N5]:
    X.add_column('split', 'update5')

# ~~~~~~ Merge per iteration ~~~~~~

data0 = N0
data0.merge(M0)
data0.merge(Y0)
assert_no_duplicates(data0)

data1 = N1
data1.merge(M1)
data1.merge(Y1)
assert_no_duplicates(data1)

data2 = N2
data2.merge(M2)
data2.merge(Y2)
assert_no_duplicates(data2)

data3 = N3
data3.merge(M3)
data3.merge(Y3)
assert_no_duplicates(data3)

data4 = N4
data4.merge(M4)
data4.merge(Y4)
assert_no_duplicates(data4)

data5 = N5
#data5.merge(M5)
#data5.merge(Y5)
#assert_no_duplicates(data5)

# ~~~~~~ Remove unexpected supersets ~~~~~~

prev_descs = set([])

'''
# (YMN3 <= YMN2 <= YMN1 <= YMN0)
data0.delete_if(seen_already(prev_descs))
data1.delete_if(seen_already(prev_descs))
data2.delete_if(seen_already(prev_descs))
data3.delete_if(seen_already(prev_descs))
data4.delete_if(seen_already(prev_descs))
""" '''
# (YMN3 >= YMN2 >= YMN1 >= YMN0)
prev_descs = set(make_descriptors(data5)) # no duplicate checking on last update
#data5.delete_if(seen_already(prev_descs))
data4.delete_if(seen_already(prev_descs))
data3.delete_if(seen_already(prev_descs))
data2.delete_if(seen_already(prev_descs))
data1.delete_if(seen_already(prev_descs))
data0.delete_if(seen_already(prev_descs))
#"""

assert(len(set(make_descriptors(data0)).intersection(set(make_descriptors(data1)))) == 0)
assert(len(set(make_descriptors(data1)).intersection(set(make_descriptors(data2)))) == 0)
assert(len(set(make_descriptors(data2)).intersection(set(make_descriptors(data3)))) == 0)
assert(len(set(make_descriptors(data3)).intersection(set(make_descriptors(data4)))) == 0)
#assert(len(set(make_descriptors(data4)).intersection(set(make_descriptors(data5)))) == 0)

# ~~~~~~ Merge final ~~~~~~

data = data0
data.merge(data1)
data.merge(data2)
data.merge(data3)
data.merge(data4)
data.merge(data5)
#assert_no_duplicates(data) # 5 Duplicates in update5

# ~~~~~~ Construct CV splits ~~~~~~

data.add_column('cv_split', '')
for d in data:
    d.cv_split = "%s_%i" % (d.split, random.randint(0, 9))

#print("Total data: %i" % len(data))

print(data.marshall())
