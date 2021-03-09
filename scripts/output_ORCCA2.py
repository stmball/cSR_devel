# coding: utf-8

import lxml.html
from lxml import etree
from lxml.cssselect import CSSSelector

import sys
import argparse

from Data import DataStream

def order_endnote_xml(source, order):
    '''
    '''
    base = lxml.html.fromstring(source)
#    base = etree.XML(source, etree.XMLParser(remove_blank_text = True))
    rank = {}
    i = 0
    for line in order.split('\n'):
        if not line: continue
        i += 1
#        sys.stderr.write('%s\n' % line)
        d = line.split()
        ID = d[2]
#        sys.stderr.write('Title: %s\n' % title)
        rank[ID] = i
    def sort_order(reference):
        ID = CSSSelector('rec-number')(reference)
        if not ID: return float('inf') # Not much we can do here...
        ID = ID[0].text_content()
        if ID in rank:
            sys.stderr.write('Found ID: %s\n' % ID)
            return rank[ID]
        else:
            sys.stderr.write('Missing ID: %s\n' % ID)
            return float('inf')
    container = CSSSelector('record')(base)
#    container = base.xpath('//*[./*]')
    container[:] = sorted(container, key=sort_order)
    i = 0
    for tag in container:
        i += 1
        CSSSelector('rec-number')(tag)[0].text = str(i)
    # LXML appears to be broken and assigning to any kind of
    # hook retrieved by CSSSelector, xpath or whatever has no
    # effect. Workaround by sorting the list and
    # reconstructing the xml file manually
    
    return container

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts data from external sources into Datastream format. Currently only supports BibReview XML. Input and output default to stdin and stdout.')
    parser.add_argument('--input',  nargs = '?', type = argparse.FileType('r'),
                        default = sys.stdin)
    parser.add_argument('--output', nargs = '?', type = argparse.FileType('w'),
                        default = sys.stdout)
    parser.add_argument('--order', nargs = '?', type = argparse.FileType('r'))
    parser.add_argument('--type', type = str)
    args = parser.parse_args()
    
    input_string = isinstance(args.input, str) and args.input or args.input.read()
    input_file   = isinstance(args.input, str) and open(args.input) or args.input
    order_string = isinstance(args.order, str) and args.order or args.order.read()
    if args.type == 'endnote_xml':
        ordered_xml = order_endnote_xml(input_string, order_string)
    else:
        sys.stderr.write("Unrecognized format option: '%s'\n" % args.type)
        sys.exit(-1)
    # Assignment doesn't work in LXML
    for elem in ordered_xml:
        args.output.write(etree.tostring(elem))
