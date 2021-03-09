
import sys
from Data import DataStream

#records = DataStream('PMID', 'label', 'source')
records = DataStream.parse(open('data/full/clef20_test_ALL_temp').read())
records_out = DataStream(*records.header)

data = {}
for row in records:
    data[row.PMID] = row

M_file = file('data/raw/qrel_abs_test.txt')
Y_file = file('data/raw/qrel_content_test.txt')
for Y_line, M_line in zip(Y_file, M_file):
    Y_d = Y_line.strip().split()
    M_d = M_line.strip().split()

    assert Y_d[0] == M_d[0]
    assert Y_d[2] == M_d[2]
    source = Y_d[0] # topic
    pmid = Y_d[2]
    if Y_d[3] == '1': label = 'Y'
    elif M_d[3] == '1': label = 'M'
    else: label = 'N'
    if not pmid in data:
        sys.stderr.write('Missing pmid (%s) in topics: %s\n' % (label, pmid))
        continue
    row_0 = data[pmid]
    
    records_out.append([row_0[h] for h in records.header])
    row = records_out[-1]
    row.label = label
    row.source = source
#    processed.add((source, pmid))
#    records.append([pmid, label, source])

#sys.stderr.write("%i missing, %i found" % (n_missing, n_found))
#assert len(processed) == 0

print records_out.marshall()
