# coding: utf-8

import lxml.html
from lxml.cssselect import CSSSelector

import sys
import argparse

try:
    from StringIO import StringIO
except ImportError:
    # for Python 3.x
    from io import StringIO

from csr.Data import DataStream

def import_pmids(source):
    '''
    Converts a list of pmids into a DataStream.
    '''
    records = DataStream('PMID')
    for pmid in source.split('\n'):
        records.append([pmid])
    return records

def import_tsv(source):
    '''
    Converts a tab separated file into a DataStream.
    '''
    line_no = 0
    for line in source.split('\n'):
        if line.strip() == "": continue
        tokens = line.split('\t')
        print(tokens)
        if line_no == 0:
            records = DataStream(*tokens)
        else:
            records.append(tokens)
        line_no += 1
    return records

def import_csv(source):
    '''
    Converts a comma separated file into a DataStream.
    '''
    import csv
    line_no = 0
    for row in csv.DictReader(StringIO(source), delimiter=',', quotechar='"'):
        if line_no == 0:
            fields = list(filter(None, row.keys()))
            records = DataStream(*fields)
        records.append([row[f] for f in fields])
        line_no += 1
    return records

def import_RIS(source):
    '''
    Converts RIS into a DataStream.
    Takes input as an iterator in order to allow any type of source. Therefore source loading is expected to be handles by the caller.
    '''
    from copy import deepcopy
    records = DataStream('PMID',
                         'AN',
                         'label',
                         'title',
                         'abstract_BACKGROUND',
                         'abstract_METHODS',
                         'abstract_RESULTS',
                         'abstract_CONCLUSIONS',
                         'abstract',
                         'publication_types',
                         'keywords',
                         'journal',
                         'author',
                         'year'
        )
    
    buffer_template = {
        'PMID': [],
        'AN': [],
        'label': 'U',
        'title': [],
        'abstract_BACKGROUND':  "MISSING_FEATURE",
        'abstract_METHODS':     "MISSING_FEATURE",
        'abstract_RESULTS':     "MISSING_FEATURE",
        'abstract_CONCLUSIONS': "MISSING_FEATURE",
        'abstract': [],
        'publication_types': [],
        'keywords': [],
        'journal': [],
        'author': [],
        'year': []
        }
    delim = {
        'PMID': "",
        'AN': "",
        'label': '',
        'title': ' ',
        'abstract_BACKGROUND':  "",
        'abstract_METHODS':     "",
        'abstract_RESULTS':     "",
        'abstract_CONCLUSIONS': "",
        'abstract': ' ',
        'publication_types': ', ',
        'keywords': ', ',
        'journal': ' ',
        'author': ',',
        'year': ""
        }
    buffer = deepcopy(buffer_template)
    translation = {
        'TY': 'publication_types',
        'AB': 'abstract',
        'KW': 'keywords',
        'TI': 'title',
        'T2': 'journal',
        'ID': 'PMID', # Hack
        'AN': 'AN',
        'AU': 'author',
        'PY': 'year'
        }
    state = None
    acc = 0
    for line in source.split('\n'):
#        sys.stderr.write(line + '\n')
        if line.strip() == "":
            if acc > 0:
                records.append([delim[t].join(buffer[t]) for t in records.header])
                buffer = deepcopy(buffer_template)
                acc = 0
            continue
        
        acc += 1
        k_v = line.split(' - ')
        if len(k_v) == 2:
            state = k_v[0].strip()
            line = k_v[1]
        value = line.strip()
        
        for s in translation.keys():
            if state == s:
                buffer[translation[s]].append(value)
        
        
    return records

def import_embase_xml(source):
    '''
    Converts Embase XML into a DataStream.
    Takes xml input as a string in order to allow any type of source. Therefore source loading is expected to be handles by the caller.
    '''
    base = lxml.html.fromstring(source)
    records = DataStream('PMID',
                         'label',
                         'title',
                         'abstract_BACKGROUND',
                         'abstract_METHODS',
                         'abstract_RESULTS',
                         'abstract_CONCLUSIONS',
                         'abstract',
                         'publication_types',
                         'keywords',
                         'journal',
                         'author',
                         'year'
        )
    
    for reference in CSSSelector('record')(base):

        d = {
            }
        translation = {
            'Title': 'title',
            'PMID': 'PMID',
            'Journal': 'journal',
            'Publication Type': 'publication_type',
            'Abstract': 'abstract',
            'Author': 'author',
            'Year of Publication': 'year'
            }
        def flatten(elems): return ''.join([elem.text_content() for elem in elems])
        def to_csv(elems): return ','.join([elem.text_content() for elem in elems])
        transformation = {
            'PMID': lambda elems: ''.join([elem.text_content() for elem in elems]).split()[0],
            'Title': flatten,
            'Journal': flatten,
            'Publication Type': flatten,
            'Abstract': flatten,
            'Author': to_csv,
            'Year of Publication': flatten
            }
        
        for elem in CSSSelector('f')(reference):
#            sys.stderr.write('%s\n' % reference)
            
            field = elem.get('l')
            target_field = translation.get(field, 'NULL')
            values = CSSSelector('d')(elem)
            value = transformation.get(field, flatten)(values)
            d[target_field] = value
#            sys.stderr.write('%s (%s) = %s\n' % (target_field, field, value))
        
        records.append([d.get(t, 'MISSING_FEATURE') for t in records.header])
    return records

def import_endnote_xml(source):
    '''
    Converts EndNote XML into DataStream.
    Takes xml input as a string in order to allow any type of source. Therefore source loading is expected to be handles by the caller.
    Since EndNote does not support structured abstracts, keywords, or publication_types, these fields will be empty.
    '''
    base = lxml.html.fromstring(source)
    records = DataStream('PMID',
                         'ID',
                         'label',
                         'title',
                         'abstract_BACKGROUND',
                         'abstract_METHODS',
                         'abstract_RESULTS',
                         'abstract_CONCLUSIONS',
                         'abstract',
                         'author',
                         'year',
                         'publication_types',
                         'keywords',
                         'journal'
        )
    
    for reference in CSSSelector('record')(base):
        # We have no IDs at all here
        pmid = "MISSING_FEATURE"
        
        label = "U"
        
        ID = CSSSelector('rec-number')(reference)
        ID = ''.join([t.text_content() for t in ID])
        title = CSSSelector('title')(reference)
        title = ''.join([t.text_content() for t in title])
        abstract = CSSSelector('abstract')(reference)
        abstract = ''.join([t.text_content() for t in abstract])
        authors = CSSSelector('authors')(reference)
        authors = ','.join([t.text_content() for t in authors])
        year = CSSSelector('year')(reference)
        year = ''.join([t.text_content() for t in year])
        if year == '': year = -1
        else: year = int(year)
        abstract_BACKGROUND  = "MISSING_FEATURE"
        abstract_METHODS     = "MISSING_FEATURE"
        abstract_RESULTS     = "MISSING_FEATURE"
        abstract_CONCLUSIONS = "MISSING_FEATURE"
#        publication_types    = "MISSING_FEATURE"
        publication_types = [t.get('name') for t in CSSSelector('ref_type')(reference)]
        keywords = [e.text_content() for e in CSSSelector('keyword')(reference)]
        keywords = '; '.join(keywords)
        # Is the journal always the secondary title?
        journal = CSSSelector('secondary-title')(reference)
        journal = ''.join([t.text_content() for t in journal])
        records.append([pmid,
                        ID,
                        label,
                        title,
                        abstract_BACKGROUND,
                        abstract_METHODS,
                        abstract_RESULTS,
                        abstract_CONCLUSIONS,
                        abstract,
                        authors,
                        year,
                        publication_types,
                        keywords,
                        journal])
    return records

def import_bibreview_xml(source):
    '''
    Converts BibReview XML into a DataStream.
    Takes xml input as a string in order to allow any type of source. Therefore source loading is expected to be handles by the caller.
    Since BibReview does not support structured abstracts, keywords, or publication_types, these fields will be empty. Users are advised to reretrieve the results from PubMed if these are required.
    '''
    base = lxml.html.fromstring(source)
    records = DataStream('PMID',
                         'label',
                         'title',
                         'abstract_BACKGROUND',
                         'abstract_METHODS',
                         'abstract_RESULTS',
                         'abstract_CONCLUSIONS',
                         'abstract',
                         'publication_types',
                         'keywords',
                         'journal'
        )
    class_labels = ['Y', 'N', 'M', 'OT']
    
    for reference in CSSSelector('reference')(base):
        pmid = reference.get('pmid')
        tags = reference.get('tags').split(',')
        label = ','.join([tag for tag in tags if tag in class_labels])
        title = reference.get('title')
        abstract = CSSSelector('abstract')(reference)[0].text # Assume only one for now
        abstract_BACKGROUND  = "MISSING_FEATURE"
        abstract_METHODS     = "MISSING_FEATURE"
        abstract_RESULTS     = "MISSING_FEATURE"
        abstract_CONCLUSIONS = "MISSING_FEATURE"
        publication_types    = "MISSING_FEATURE"
        keywords = "MISSING_FEATURE"
        journal = reference.get('journal')
        records.append([pmid,
                        label,
                        title,
                        abstract_BACKGROUND,
                        abstract_METHODS,
                        abstract_RESULTS,
                        abstract_CONCLUSIONS,
                        abstract,
                        publication_types,
                        keywords,
                        journal])
    return records        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts data from external sources into Datastream format. All input data is assumed to be in UTF-8 or ASCII format. Input and output default to stdin and stdout if --input and --output are unspecified.')
    parser.add_argument('--input',  nargs = '?', type = argparse.FileType('r'),
                        default = sys.stdin)
    parser.add_argument('--output', nargs = '?', type = argparse.FileType('w'),
                        default = sys.stdout)
    parser.add_argument('--type', type = str, choices = [
        'bibreview',
        'endnote_xml',
        'embase_xml',
        'RIS',
        'pmids',
        'tsv',
        'csv'
    ])
    args = parser.parse_args()

    input_string = isinstance(args.input, str) and args.input or args.input.read()
#    input_file   = isinstance(args.input, str) and open(args.input) or args.input
    if args.type == 'bibreview':
        records = import_bibreview_xml(input_string)
    elif args.type == 'endnote_xml':
        records = import_endnote_xml(input_string)
    elif args.type == 'embase_xml':
        records = import_embase_xml(input_string)
    elif args.type == 'RIS':
        records = import_RIS(input_string)
    elif args.type == 'pmids':
        records = import_pmids(input_string)
    elif args.type == 'tsv':
        records = import_tsv(input_string)
    elif args.type == 'csv':
        records = import_csv(input_string)
    else:
        sys.stderr.write("Unrecognized format option: '%s'\n" % args.type)
        sys,exit(-1)
    args.output.write(records.marshall())
