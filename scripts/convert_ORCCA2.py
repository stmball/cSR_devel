
import sys, random

from Data import DataStream
import Import

#records = DataStream('PMID', 'label', 'source')

U_test = Import.import_endnote_xml(file('data/raw/ORRCA2/2015-2016.xml').read())
U_train = Import.import_endnote_xml(file('data/raw/ORRCA2/Search results from the beginning - 2014.xml').read())
#Y_train = Import.import_endnote_xml(file('data/raw/ORRCA2/References included ORRCA.xml').read())
Y_train = Import.import_endnote_xml(file('data/raw/ORRCA2/B7-included articles.xml').read())

U_test.add_column('split', 'test')
U_train.add_column('split', 'train')
Y_train.add_column('split', 'train')

Y = {}
for row in Y_train:
    Y[row.title] = row

n_Y = 0
for row in U_train:
    if random.random() < 0.5: row.split = 'test'
    if row.title in Y:
        row.label = 'Y'
        del Y[row.title]
        n_Y += 1
    else:
        row.label = 'N'

for title in Y.keys():
    sys.stderr.write("Missing in U: '%s'\n" % title)
    row = Y[title]
    row.label = 'Y'
    U_train.append([row[t] for t in U_train.header])
sys.stderr.write("  #Y in U:     '%i'\n" % n_Y)
sys.stderr.write("  #Y not in U: '%i'\n" % len(Y))

#U_train.merge(U_test)

U_train.add_column('source', 'ORCCA2')

print U_train.marshall()
