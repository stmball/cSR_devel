
# ~~~~~~                    Import                  ~~~~~~

import os, sys
import shutil
import subprocess
import tempfile
import string
import random
import re
import glob

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_svmlight_file

from csr.Log import start_statusbar

# ~~~~~~                    Setup                   ~~~~~~

STATUS = start_statusbar(sys.stderr)

# ~~~~~~                  Definitions               ~~~~~~

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

class TREC_BMI_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, tf_idf = True):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        installation_root = os.path.join(dir_path, '..', '..', '..')
        exec_dir = os.path.join(installation_root, 'lib', 'TREC_BMI', 'featurekit')
        self.TREC_BMI_dir = os.path.realpath(exec_dir)
        
    def fit(self, x, y = None):
        # Unfitted transformer
        return self
    
    # For debugging
    def transform_preloaded(self, data, run_id):
        
        dirname = 'TREC_%s' % run_id
        STATUS[5]('Extracting pretrained TREC BMI features (run ID: %s)' % run_id)
        
        STATUS[5]('Using external executables in: %s' % self.TREC_BMI_dir)
        # This section will crash and burn if another thread changes the cwd...
        
        owd = os.getcwd()
        try:
            os.chdir('temp_trec')
            
            ret_val = load_svmlight_file('labeled.svm.%s.fil' % dirname)
            X_t, y = ret_val[0], ret_val[1]
            n, m = X_t.shape
            STATUS[5]('Extracted data of size [%i x %i] from input data of length %i' % (
                    n, m, len(data)
                    ))
            # Sanity: Did the processing change the order of the labels?
            assert all([(l == 'Y' and 1 or 0) == y for l, y in zip(data.label, y)])
            assert len(data) == n
            return X_t
        finally:
            os.chdir(owd)
            STATUS[10]('Changing to original cwd: %s' % os.getcwd())                
    
    def transform(self, data):
        
        return self.transform_preloaded(data, 'OC6YJ2FB')
    
#        return self.transform_preloaded(data, '59CGBS0W')
#        return self.transform_preloaded(data, '744VLEYD')
#        return self.transform_preloaded(data, 'M56HTJZF') # Prefit title + abstract
#        return self.transform_preloaded(data, 'HNS8A8GR') # Prefit abstract vs seed title
#        return self.transform_preloaded(data, 'ZT3ZRNV4') # Prefit abstract vs seed title (all topics?)
#        return self.transform_preloaded(data, 'BWYR750G') # Prefit abstract vs seed title
        
        chars = string.ascii_uppercase + string.digits
        run_id = ''.join(random.SystemRandom().choice(chars) for _ in range(8))
        dirname = 'TREC_%s' % run_id
        STATUS[5]('Extracting TREC BMI features (run ID: %s)' % run_id)
        
        STATUS[5]('Using external executables in: %s' % self.TREC_BMI_dir)
        # This section will crash and burn if another thread changes the cwd...
        
        with tempfile.TemporaryDirectory() as temp_dir:
            STATUS[5]('Changing to temp directory: %s' % temp_dir)
            
            # Copy stopwords file to temp dir so the scripts can access it
            src_stopwords = os.path.join(self.TREC_BMI_dir, 'english.stop.stem')
            dst_stopwords = os.path.join(temp_dir, 'english.stop.stem')
            shutil.copyfile(src_stopwords, dst_stopwords)
            
            owd = os.getcwd()
            try:
#                os.chdir(temp_dir)
                os.chdir('temp_trec')
                
                # Book keeping for reconstructing the features
                filename_to_label = {}
                row_index_to_filename = []
                
                # Filename format is accession number padded with zeroes
                # up to the largest digit length
                filename_fmt = 'doc_%%0%dd' % len(str(len(data)))
                os.mkdir(dirname)
                row_i = 0
                for row in data:
                    filename = filename_fmt % row_i
                    filename = os.path.join(dirname, filename)
                    filename_to_label[filename] = row.label
                    row_index_to_filename.append(filename)
#                    STATUS[5]('Creating file: %s' % os.path.realpath(filename))
                    with open(filename, 'w') as out:
                        if row.split == 'seed':
                            out.write('%s\n' % row.title)
                        else:
                            out.write('%s\n' % row.title)
                            out.write('%s\n' % row.abstract)
                    row_i += 1
                
                f_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(dirname, '*'))
                             if os.path.isfile(f))
                f_n = len(os.listdir(dirname))
                STATUS[5]('Created %i temporary files with total size %s' % (f_n, sizeof_fmt(f_size)))
                
#                print(os.listdir(dirname))
                temp_env = os.environ.copy()
                temp_env["PATH"] = ':'.join([temp_env["PATH"], self.TREC_BMI_dir])
#                exec_path = os.path.join('featurekit', 'dofeatures')
                exec_path = 'dofeatures'
                proc = subprocess.Popen([exec_path, dirname], env = temp_env)
                STATUS[5]('Processing data with external script...')
                proc.communicate() # Wait for completion
#                STATUS[5]('Done.')
                # Check if failed?
                
                # The BMI feature pipeline outputs labels = filenames
                # whereas the classifiers expect actual labels
                STATUS[5]('Relabeling extracted data...')
                pattern = re.compile('(%s/[-\w]*)\W(.*)' % dirname)
                extracted_data = {}
                with open('svm.%s.fil' % dirname) as svm_file_in:
                    for line in svm_file_in:
                        match = pattern.match(line)
                        assert match
                        filename, cargo = match.groups(1)
#                            STATUS[10]("Parsing '%s'" % (line))
#                        STATUS[10]('Relabeling %s => %s' % (filename,
#                                                            filename_to_label[filename]))
                        # Usually the label must be numerical in the svmlight format?
                        label = (filename_to_label[filename] == 'Y' and 1 or 0)
                        out_string = '%i %s\n' % (label, cargo)
                        extracted_data[filename] = out_string
                # The extraction script outputs rows in lexical order of filenames
                # Make no assumptions, this might well differ on different systems
                with open('labeled.svm.%s.fil' % dirname, 'w') as svm_file_out:
                    for filename in row_index_to_filename:
                        if filename in extracted_data:
                            out_string = extracted_data[filename]
                            svm_file_out.write(out_string)
                        else:
                            label = (filename_to_label[filename] == 'Y' and 1 or 0)
                            out_string = '%i\n' % (label)
                            svm_file_out.write(out_string)
                
#                STATUS[5]('Done.')
                
                ret_val = load_svmlight_file('labeled.svm.%s.fil' % dirname)
                X_t, y = ret_val[0], ret_val[1]
#                for y_1, y_2 in zip(data.label, y):
#                    print('%s - %i' % (y_1, y_2))
                n, m = X_t.shape
                STATUS[5]('Extracted data of size [%i x %i] from input data of length %i' % (
                        n, m, len(data)
                        ))
                # Sanity: Did the processing change the order of the labels?
                assert all([(l == 'Y' and 1 or 0) == y for l, y in zip(data.label, y)])
                assert len(data) == n
                return X_t
                
                        
            finally:
                os.chdir(owd)
                STATUS[10]('Changing to original cwd: %s' % os.getcwd())
