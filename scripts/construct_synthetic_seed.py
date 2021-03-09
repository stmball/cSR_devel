
from bin.Data import DataStream

import glob
from enum import Enum

seed_set = DataStream('source', 'title', 'abstract')

# 

class Section(Enum):
    TOPIC: 1
    TITLE: 2
    QUERY: 3
    PMIDS: 4
#    DESC:  5

#for filename in glob.glob('../../clef_tar_2018/tar/training/topics_train/*'):
for filename in glob.glob('../../clef_tar_2018/tar/testing/topics/*'):
    with open(filname) as infile:
        contents = {
            Section.TOPIC: [],
            Section.TITLE: [],
            Section.QUERY: [],
            Section.PMIDS: []
            }
        section = None
        for line in infile:
            if   line.startswith('Topic:'):
                content = line.split(': ')[1]
                section = Section.TOPIC
            elif line.startswith('Title:'):
                content = line.split(': ')[1]
                section = Section.TITLE
            elif line.startswith('Query:'):
                content = ''
                section = Section.QUERY
            elif line.startswith('Pids:'):
                content = ''
                section = Section.PMIDS
            else:
                content = line.strip()
            contents[section].append(content)
        extract_data_for_section = lambda sec: ' '.join(contents[sec])
        seed_set.append(map(extract_data_for_section, [Section.TOPIC, Section.TITLE, Section.Query]))

print(seed_set.marshall())
