# PGNN-EK

This is the official implementation of ACL 2022 Paper ***A Neural Network Architecture for Program Understanding Inspired by Human Behaviors***

We provide our data in [Google Drive](https://drive.google.com/drive/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn?usp=sharing).

## Setup

This implemetation is based on Python 3.7. To run the code, you need the following dependencies:

- torch=1.7.0+cu110
- torch-cluster==1.5.9
- torch-scatter==2.0.5
- torch-spline-conv==1.2.0
- torch-sparse==0.6.8
- torch-geometric==2.0.4
- transformers==4.18.0
- javalang==0.13.0
- anytree==2.8.0
- pandas==1.3.5
- nltk==3.5

Or you can simply run:

```python
pip install -r requirements.txt
```

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
    |   |-- run_cs.py # run cs
    |-- preprocess # preprocessing
        |-- api_match.py # match API
        |-- bcbf_construct.py # construct BCB-F
        |-- ccd_enhanced_with_api.py # enhance ccd dataset with API description
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

We use the code summarization task as example. The code clone detection task follows the similar pipeline. We conduct all experiments on two Tesla V100 GPUs.

1.Enhance raw dataset with API description. You need to specify the dataset by setting args 'dataset'. This procedure will cost dozens of minutes. After that, you will see new enhanced data in the corresponding directory, for example, "data/CSN/". You can download the raw dataset and enhanced dataset from [Google Drive](https://drive.google.com/drive/u/2/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn).

```python
cd code/preprocess
python3 cs_enhanced_with_API.py --dataset=CSN
```

2.Construct S-AST and generate input features for the model. You need to specify the dataset by setting args 'dataset'. This procedure will cost 1-2 hours. After that, you will see new features data in the corresponding directory, for example, "code/features/CSN/". You can download the processed features from [Google Drive](https://drive.google.com/drive/u/2/folders/10r5kkg6QNOhHkaoMVBXjprepd_LzuwDn). For the limitation size(15G) of Google Drive, we can only provide the features of CSN and TLC.

```python
python3 cs_features_generate.py --dataset=CSN
```

3.Make the final prediction. You need to specify the dataset by setting args 'dataset'. This procedure will cost 1-2 days. Notice, you can experiment with different hyper-parameters by altering configs in "config_cs.yml" or "config_ccd.yml", such as 'divide_node_num', namely $\lambda$ that specifies the minimum number of nodes in the subgraph.

```python
cd ../models
python3 run_cs.py --dataset=CSN
```

## BCB-F Construction

We download the [BigCloneBench](https://github.com/clonebench/BigCloneBench) 2015 full database (PostgreSQL) from [link](https://1drv.ms/u/s!AhXbM6MKt_yLkLF5_iiuoWhmQUScqg?e=yAEHI5).

You can construct the BCB-F dataset after configuring PostgreSQL:

```python
!pip install psycopg2
cd code/preprocess
python3 bcbf_construct.py
```

Here is an example of BCB-F data:
```python
20643742	23677117	0
```
From left to right: code1 index, code2 index, label.

## Attribution

Parts of this code are based on the following repositories:

- [CodeBERT](https://github.com/microsoft/CodeBERT)

- [code2seq](https://github.com/m3yrin/code2seq)

- [FA-AST](https://github.com/jacobwwh/graphmatch_clone)

# Citation

if you find this code working for you, please cite:

```python
@inproceedings{zhu2022neural,
  title={A Neural Network Architecture for Program Understanding Inspired by Human Behaviors},
  author={Zhu, Renyu and Yuan, Lei and Li, Xiang and Gao, Ming and Cai, Wenyuan},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5142--5153},
  year={2022}
}
```
