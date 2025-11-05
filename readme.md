# COMET CSR Tool 

This is a systematic review tool for the COMET project adapted from https://github.com/aneveol/cSR_devel - it includes Dockerisation of the workflow for easier installation and running.

## Prerequisites

To run this code on any machine, you only require ![Docker](https://www.docker.com/) installed.

## Running the tool

The data and models for ordering literature is not included in this repository, for *reasons*. Instead, below are instructions for training a model and using it to predict your own data.

### Organising your Data

The Docker container and Python package requires your data to be stored under the `user_data` folder. This folder has several subfolders:

* `files_for_analysis` contains any files you want to order. Currently, please only put one file at a time in this folder.
* `negative_labels` contains any files that *are not* related to your topic. In older systems, these had the suffix `_M.xml`  
* `positive_labels` contains files that *are* related to your topic. In older versions, these had the suffix `_Y.xml`.
* `output_files` is a folder that will contain the output order file when the program is finished running.

Your folder structure should look as follows:

```
csr/
data/
user_data/
    files_for_analysis/
        input_file.xml
    negative_labels/
        COMET_originalSR_M.xml
    positive_labels/
        COMET_originalSR_Y.xml
    outputs/
Dockerfile
README.md
requirements.txt
```

### Running the Code

To run the code, you will need to build the Docker container and run the analysis pipeline using the following terminal command:

```bash
docker build -t csr-new . && docker run -v "$(pwd)/user_data:/MLapp/user_data" csr-new
```

Your terminal should fill with lines starting with either `Pre-normalization` or `Post-normalization`; pausing for a while (sometimes hours!); and finally finishing. In your output folder should be an EndNote XML file that is ordered correctly 


## Frequently Asked Questions

Q: I am getting `docker: permission denied` error.
A: Run the command with elevated permissions (via `sudo` on Linux)

Q: I am getting a `file not found` error.
A: You will need to rebuild the docker container after moving your files to the `user_data` folder. I know - this isn't the right way to do it!
