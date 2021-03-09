# coding: utf-8

from Data import DataStream
import random

d = DataStream.parse(open("data/full/clef50_ALL_labeled_ALL").read())

dropped_fields = ['publication_types', 'mesh_terms', 'keywords', 'journal', 'references']

for row in d:
    if row.split == 'test' and random.random() < 0.5:
        for field in dropped_fields:
            row[field] = 'MISSING_FEATURE'

print d.marshall()
