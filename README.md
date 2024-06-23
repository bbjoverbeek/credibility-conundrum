# claimant-credibility

Most of the results can be found in the output of the Jupyter Notebooks. 
If you wish to run the experiment yourself, follow these steps:

## Environment setup

Using python 3.10.12, create a virtual environment and install the dependencies. 

```sh
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Download the data
// TODO fetch claimant dataset

Get the original vaccination corpus (used for testing example validity and extracting embeddings in context)
```sh
git clone git@github.com:cltl/vaccination-corpus.git
mv vaccination-corpus/data/annotations-pickle/ data/vaccination_corpus_pickle/
```

While this is downloaded, also move over the conll parsing dependency:
```sh
mv vaccination-corpus/code/conll_data.py ./
```

Then (optionally) remove the rest of the repository
```sh
rm -rf vaccination-corpus/
```

// TODO get publisher and claimant category annotations.

## Pre-process the dataset
Assuming your claimant data is in the ./data/claimant_data folder, remove invalid examples using the following command:
```sh
python3 pre-process.py
```
This will create a folder `claimant_data_processed/` that has the same structure as `claimant_data/`without examples that have invalid context (there should be 6).

## Running (descriptive) statistics
You can now run notebooks 1-3 in the virtual environment:  
[1_data_exploration.ipynb](./1_data_exploration.ipynb)  
[2_publisher_annotations.ipynb](./2_publisher_annotations.ipynb)  
[3_claimant_annotations.ipynb](./3_claimant_annotations.ipynb)  

## Running the regression experiments
To run the regression experiments, the word embeddings need to be extracted from the claimants. 
Run the script to do this, and make sure you are running this on a machine capable enough to run [DeBERTaV3 Large](https://huggingface.co/microsoft/deberta-v3-large).

```sh
python3 extract_embeddings.py
```

## Running the regression experiments
You can now run the regression experiments in the notebook:  
[4_regression.ipynb](./4_regression.ipynb)  
