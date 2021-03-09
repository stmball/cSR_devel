# coding: utf-8

import lxml.html
from lxml.cssselect import CSSSelector

import sys
import argparse

from csr.Data import DataStream

def export_BERT(data, out):
    '''
    Write data in BERT tsv format
    '''
    def normalize(s):
        s = str(s)
        s = s.replace('\n', ' ')
        return s
    assert all([not '\t' in x for x in data.header])
    out.write('%s\n' % '\t'.join(data.header))
    for row in data:
        d = [normalize(row[x]) for x in data.header]
        if d[1] == 'Y': print(d)
        assert all([not '\t' in x for x in d if x])
        out.write('%s\n' % '\t'.join(d))

def export_csv(data, out):
    '''
    Converts a comma separated file into a DataStream.
    '''
    import csv
    writer = csv.DictWriter(out, fieldnames = data.header)

    writer.writeheader()
    for row in data:
        writer.writerow(dict([(f, row[f]) for f in data.header]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts data from Datastream format to external formats. Input and output default to stdin and stdout.')
    parser.add_argument('--input',  nargs = '?', type = argparse.FileType('rb'),
                        default = sys.stdin)
    parser.add_argument('--output', nargs = '?', type = argparse.FileType('w'),
                        default = sys.stdout)
    parser.add_argument('--type', type = str)
    args = parser.parse_args()

    input_string = isinstance(args.input, str) and args.input or args.input.read()
    data = DataStream.parse(input_string)
    if args.type == 'BERT':
        exporter = export_BERT
    elif args.type == 'csv':
        exporter = export_csv
    else:
        sys.stderr.write("Unrecognized format option: '%s'\n" % args.type)
        sys.exit(-1)
    exporter(data, args.output)
