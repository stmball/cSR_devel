
cSR Screening Automation Framework
==================================

The cSR framework is intend

Preliminaries
-------------

This quickstart assumes the user is running some variant of UNIX, and has a working installation of `python3` (at least 3.5.2 is expected).
For the code examples, the cSR package is assumed to be checked out to a directory `~/cSR`

Installing Dependencies
-----------------------

Before starting, make sure the current working directory is the cSR root directory.
Also test that you have a recent version of python3:

```console
norman@colette:~/cSR$ which python3
/usr/bin/python3
norman@colette:~/cSR$ python3 --version
Python 3.5.2
```

### Using pip

First test that pip is linked to the right python executable:

```console
norman@colette:~/cSR$ which pip3
/usr/bin/pip3
norman@colette:~/cSR$ pip3 --version
pip 19.3.1 (python 3.5.2)
```

To create a virtual environment called `SR_env`:

```console
norman@colette:~/cSR$ mkvirtualenv -p `which python3` SR_env
norman@colette:~/cSR$ workon SR_env
(SR_env) norman@colette:~/cSR$ pip3 install -r requirements.txt
````

You should now be able to run cSR.
To reload the environment after logging out, run:

```console
norman@colette:~/cSR$ workon SR_env
````

### Using conda (unsupported)

First test that conda is linked to the right python executable:

```console
norman@colette:~/cSR$ which conda
/usr/bin/conda
norman@colette:~/cSR$ conda --version
```

To create a virtual environment called SR_env:

```console
norman@colette:~/cSR$ conda create -n SR_env python==3.5.2
norman@colette:~/cSR$ activate SR_env
(SR_env) norman@colette:~/cSR$ conda install -c conda-forge --file requirements.txt
````

You should now be able to run cSR.
To reload the environment after logging out, run:

```console
norman@colette:~/cSR$ activate SR_env
````

### Testing the installation

If all dependencies have been installed correctly, the following command should run without error:

```console
norman@colette:~/cSR$ python -m csr.Data --help
````

### Tensorflow GPU dependencies (optional)

The cSR framework will use GPU acceleration if this is supported by the local installation of tensorflow (tensorflow-gpu). Installing these dependencies is beyond the scope of this readme.
For more information see the [tensorflow instructions](https://www.tensorflow.org/install/pip)

conda is recommended to install tensorflow-gpu

Modules/tools
-------------

All modules and tools reside in the csr module (folder). All classes and functions can be  called from python programs, and several can be used from the command line.
The documentation of the tools are available directly in the tools, accessible with the `--help` flag. The documentation of the framework for use as a python library is available in the doc folder (`doc/index.html`)

The following modules can be used as standalone tools from the command line:

- `csr.Data`: Class to inspect/store/load/handle data. Running the module from the command line allows inspecting and modifying existing data files
- `csr.Import`: Methods to convert external formats to datastream format
- `csr.Export`: Methods to convert datastream files to external formats
- `csr.Medline`: Methods to query PubMed
- `csr.Train`: Training, applying and evaluating machine learning models on existing datasets
- `csr.ML.Evaluation`: Methods to calculate ranking performance on existing data
- `csr.Vocabulary`: Methods to pre-construct vocabulary files from datasets to decrease resource usage during training

As an example, to inspect the contents of `data/full/COMET/COMET_update1_M.json`:

```console
norman@colette:~/cSR$ python -m csr.Data --inspect --input data/full/COMET/COMET_update1_M.json
````

All command line usage require calling the tools as python modules. That is, the following does not work:

```console
norman@colette:~/cSR$ python csr/Data.py --inspect --input data/full/COMET/COMET_update1_M.json
````

Code examples
-------------

### Example use case scripts

1.

```console
norman@colette:~/cSR$ bash csr/examples/small/rank_endnote_xml.sh
```

This example uses the pipeline `csr/examples/common/pipelines/sparse_trivial.yaml` and the classifier `csr/examples/common/classifiers/SGD_50epochs.yaml`to import data from an external data format (`EndNote XML`), train and apply a ranker on the data, and sort the data files. The pipeline takes input features from titles, abstracts and keywords.

2.

```console
norman@colette:~/cSR$ bash csr/examples/small/rank_sentences.sh
```


```console
norman@colette:~/cSR$ bash csr/examples/quick/run_COMET.sh
```

### Data Handling

1. Inspect the entire file contents of `file1.json`.

```bash
norman@colette:~/cSR$ python -m csr.Data --inspect --input file1.json
```

2. Inspect the file contents of `file1.json`, limiting the results to columns `label` and `title`, and to the first 20 rows.

```bash
norman@colette:~/cSR$ python -m csr.Data --inspect --input file1.json --col label title --select 20
```

3. Merge files `file1.json`, `file2.json` and `file3.json`, writing the results to `file_out.json`. Data format (columns) must be compatible.

```bash
norman@colette:~/cSR$ python -m csr.Data --input file1.json file2.json file3.json --output file_out.json
```

OR

```bash
norman@colette:~/cSR$ python -m csr.Data --input file*.json --output file_out.json
```

4. Inspect all rows in `file1.json` where the column `split` is equal to `train`, and `label` is equal to `Y`.

```bash
norman@colette:~/cSR$ python -m csr.Data --inspect --input file1.json --get split=train label=Y
```

5. Open `file1.json`, set the column `date_constructed` to `Jan 1, 1970` for all rows (creating the column if it does not exist), and save the results to `file2.json`.

```bash
norman@colette:~/cSR$ python -m csr.Data --inspect --input file1.json --set date_constructed="Jan 1, 1970" --output file2.json
```

6. Replace the label values `yes` with `Y`, and `no` with `N`, saving the the results to `file2.json`

```bash
norman@colette:~/cSR$ python -m csr.Data --input file1.json --get label=yes --set label=Y --output file1_tempY.json
norman@colette:~/cSR$ python -m csr.Data --input file1.json --get label=no --set label=N --output file1_tempN.json
norman@colette:~/cSR$ python -m csr.Data --input file1_temp{Y,N}.json --output file2.jso
norman@colette:~/cSR$ rm file1_temp{Y,N}.json
```

