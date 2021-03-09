
import sys
import csv

in_file = csv.reader(sys.stdin,
                     delimiter = ',',
                     quoting = csv.QUOTE_ALL,
                     quotechar = '"',
                     skipinitialspace = True)
out_file = csv.writer(sys.stdout, delimiter = '\t', quoting = csv.QUOTE_NONE, escapechar = '')
for row in in_file:
    row = [s.strip(' "') for s in row]
    row = [s.replace('/\w/', ' ') for s in row]
    
    out_file.writerow(row)
