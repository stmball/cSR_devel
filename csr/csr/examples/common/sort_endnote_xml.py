
import xml.etree.ElementTree as ET
import argparse

from csr.Data import DataStream

def get_id(record):
    ID = record.find('rec-number')
    return ID.text

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--data',             required = True,
                    help = 'Input EndNote file')
parser.add_argument('--order',            required = True,
                    help = 'Order file in DataStream format')
parser.add_argument('--output',           type = str,
                    help = 'Filepath to output to')
args = parser.parse_args()

open(args.order)
order = DataStream.parse(open(args.order).read())

tree = ET.parse(args.data)
xml = tree.getroot()

id_to_score = dict(zip(order.ID, order.confidence))
def tag_to_score(tag):
    return id_to_score[get_id(tag)]

ID_out = []
for records in xml:
    content = [record for record in records if get_id(record) in id_to_score]
    records[:] = sorted(content, key = tag_to_score, reverse = True)
    i = 0
    for record in records:
        i += 1
        for title in record.find('titles'):
            span = title.find('style')
            span.text = "%5i %s" % (i, span.text)
    i = 0
    for record in records:
        i += 1
        if i > 10: break
        ID_out.append(get_id(record))

tree.write(args.output)
