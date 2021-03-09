
from common.Data import DataStream

import sys

import lxml.html
from lxml.cssselect import CSSSelector

import glob

records = {}

filenames = glob.glob('data/raw/cochrane_crowd/*.xml')
for filename in filenames:
    print "Reading file '%s'" % filename

    tree = lxml.html.fromstring(open(filename).read())
    
    i = 0
    record_tags = CSSSelector('record')(tree)
    for record in record_tags:
        i += 1
        sys.stdout.write("\r%s\r" % ' '*40)
        sys.stdout.write("Reading %i records (%i%%)" % (i, int(round(100*i/len(record_tags)))))
        
        ids = CSSSelector('embaseid')(record)
        if len(ids) == 0:
            id = None
        elif len(ids) == 1:
            id = ids[0].text_content()
        else:
            raise ValueError("Record had %i EmbaseIDs" % len(ids))
#        labels = CSSSelector('decision:not(assessment)')(record)
        labels = record.xpath('decision')
        if len(labels) == 0:
            label = None
        elif len(labels) == 1:
            label = labels[0].text_content()
        else:
            raise ValueError("Record had %i Decisions" % len(labels))
        if not label in records:
            records[label] = []
        records[label].append(id)
        
    print
    print "Done"
filenames = glob.glob('data/raw/cochrane_crowd/*.csv')
for filename in filenames:
    # This can be done simpler with a one-liner, but would assume
    # that ids are unique (wrong) and that each id has only one
    # label (possibly wrong)
    i = 0
    for line in open(filename):
        i += 1
        if i == 1: continue
        tokens = line.strip().split(',')
#        print tokens
        if len(tokens) != 2: continue
        id, label = tokens
        if not label in records:
            records[label] = []
        records[label].append(id)

data = DataStream('PMID', 'source', 'label')

for label in records.keys():
    
    ids = records[label]
    n = len(ids)
    # Assume PMID 8 digits, Embase accession number 10 digits
    PMIDs   = [id for id in ids if id and len(id) == 8]
    EMIDs   = [id for id in ids if id and len(id) == 10]
    n_blank = len([id for id in ids if not id])
    print "ID available for label '%s':" % label
    print "  PubMed:              %i (%i unique)" % (len(PMIDs), len(filter(None, set(PMIDs))))
    print "  Embase:              %i (%i unique)" % (len(EMIDs), len(filter(None, set(EMIDs))))
    print "  Missing:             %i" % n_blank
    print "  Unrecognized format: %i" % (len(ids) - n_blank - len(PMIDs) - len(EMIDs))
    print "  Total:               %i (%i unique)" % (len(ids),
                                                     len(set(filter(None, ids))))
    
    print "Overlap:"
    for other_label in records.keys():
        if label == other_label:
            print "    %20s: -" % other_label
        else:
            print "    %20s: %i unique" % (other_label, len(set(ids).intersection(records[other_label])))
    
    if label in ['Reject', 'DTA', 'RCT or CCT']:
        for PMID in PMIDs:
            data.append([PMID, 'cochrane_crowd_20171026', label])

#with open('data/pmid/cochrane_crowd_20171026', 'w') as out:
#    out.write(data.marshall())
