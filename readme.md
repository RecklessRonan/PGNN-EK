# PGNN-EK

We provide our data in anonymous [Google Drive](https://drive.google.com/drive/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn?usp=sharing).

## Setup

This implemetation is based on PyTorch. To run the code, you need the following dependencies:

- PyTorch==1.5.1

- PyTorch Geometric==1.7.0

- Transformers==4.8.2

- javalang==0.11.0

- anytree==2.8.0

- pandas==0.25.3

## Repository structure

```python
|-- code
    |-- configs # configurations for code summarization (cs) and code clone detection (ccd)
    |   |-- config_ccd.yml
    |   |-- config_cs.yml
    |-- features # store the processed features for 4 datasets
    |   |-- BCB
    |   |-- BCB-F
    |   |-- CSN
    |   |-- TLC
    |-- models # model design
    |   |-- bleu.py # calculate bleu in cs
    |   |-- codebert_seq2seq.py # the seq2seq model for cs
    |   |-- pgnn.py # partitioning-based graph neural network
    |   |-- run_ccd.py # run ccd
    |   |-- run_cs.py # run cs
    |-- preprocess # preprocessing
        |-- api_match.py # match API
        |-- bcbf_construct.py # construct BCB-F
        |-- ccd_enhanced_with_api.py # enhance ccd dataset with API description
        |-- ccd_features_generate.py # generate processed features for ccd
        |-- cs_enhanced_with_api.py # enhance cs dataset with API description
        |-- cs_features_generate.py # generate processed features for cs
        |-- get_javaapi.py # get java API from documentation
        |-- sast_construct.py # construct s-ast
|-- data
    |-- BCB
    |-- BCB-F
    |-- CSN
    |-- TLC
    |-- java-api # store java API documents and extracted method-description pairs, you can download from Google Drive.
```

## Run pipeline

We use the code summarization task as example. The code clone detection task follows the similar pipeline.

1.Enhance raw dataset with API description. You need to specify the dataset by setting args 'dataset'. This procedure will cost dozens of minutes. After that, you will see new enhanced data in the corresponding directory, for example, "data/CSN". You can download the raw dataset and enhanced dataset from [Google Drive](https://drive.google.com/drive/u/2/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn).

```python
cd code/preprocess
python3 cs_enhanced_with_API.py --dataset=CSN
```

2.Construct S-AST and generate input features for the model. You need to specify the dataset by setting args 'dataset'. This procedure will cost 1-2 hours. After that, you will see new features data in the corresponding directory, for example, "code/features/CSN". You can download the processed features from Google Drive. For the limitation size(15G) of [Google drive](https://drive.google.com/drive/u/2/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn), we can only provide the features of CSN and TLC.

```python
python3 cs_features_generate.py --dataset=CSN
```

3.Make final prediction. You need to specify dataset by setting args 'dataset'.

```python
cd ../models
python3 run_cs.py --dataset=CSN
```

## Attribution

Parts of this code are based on the following repositories:

- [CodeBERT](https://github.com/microsoft/CodeBERT)

- [code2seq](https://github.com/m3yrin/code2seq)

- [FA-AST](https://github.com/jacobwwh/graphmatch_clone)
