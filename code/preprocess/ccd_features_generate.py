from transformers import RobertaTokenizer
import torch
from torch_geometric.data import Data
import pandas as pd
import argparse
from sast_construct import get_subgraph_node_num, get_pyg_data_from_ast, parse_program
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate features for model')
parser.add_argument('--dataset', type=str, help='BCB or BCB-F')
parser.add_argument('--divide_node_num', type=int, default=30, help='\lambda')
parser.add_argument('--max_source_length', type=int,
                    default=400, help='the max length of code')
parser.add_argument('--max_node_num', type=int,
                    default=450, help='used to limit the max subgraph number')
args = parser.parse_args()
if args.dataset == 'BCB':
    dataset_url = '../../data/BCB/'
    features_url = '../features/BCB/'
elif args.dataset == 'BCB-F':
    dataset_url = '../../data/BCB-F/'
    features_url = '../features/BCB-F/'
else:
    print('Wrong dataset name')
divide_node_num = args.divide_node_num
max_source_length = args.max_source_length
max_subgraph_num = int(args.max_node_num/divide_node_num)


# read dataset
data_url = dataset_url + 'data_enhanced.jsonl'
train_url = dataset_url + 'train.txt'
valid_url = dataset_url + 'test.txt'
test_url = dataset_url + 'valid.txt'
data = pd.read_json(path_or_buf=data_url, lines=True)


def read_ccd_pairs(url):
    data = []
    with open(url) as f:
        for line in f:
            line = line.strip()
            id1, id2, label = line.split('\t')
            label = 0 if label == '0' else 1
            data.append((int(id1), int(id2), label))
    return data


train_pairs = read_ccd_pairs(train_url)
valid_pairs = read_ccd_pairs(valid_url)
test_pairs = read_ccd_pairs(test_url)


# obtain the s-ast data
x_list = []
edge_index_list = []
edge_attr_list = []
subgraph_node_num_list = []
real_graph_num_list = []

for i in tqdm(range(len(data))):
    ast = parse_program(data['func'][i])
    x, edge_index, edge_attr, root_children_node_num = get_pyg_data_from_ast(
        ast)
    subgraph_node_num, real_graph_num = get_subgraph_node_num(
        root_children_node_num, divide_node_num, max_subgraph_num)
    x_list.append(x)
    edge_index_list.append(edge_index)
    edge_attr_list.append(edge_attr)
    subgraph_node_num_list.append(subgraph_node_num)
    real_graph_num_list.append(real_graph_num)
data['ast_token'] = data['ast'].str.split()
data['des_token'] = data['des'].str.split()
data['ast_length'] = data['ast_token'].str.len()
data['des_length'] = data['des_token'].str.len()


# define a pair class pf pytorch geometric
class PairData(Data):
    def __init__(self, edge_index_s, edge_attr_s, x_s, source_ids_s, subgraph_node_num_s, real_graph_num_s,
                 edge_index_t, edge_attr_t, x_t, source_ids_t, subgraph_node_num_t, real_graph_num_t, label):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.source_ids_s = source_ids_s
        self.subgraph_node_num_s = subgraph_node_num_s
        self.real_graph_num_s = real_graph_num_s

        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.source_ids_t = source_ids_t
        self.subgraph_node_num_t = subgraph_node_num_t
        self.real_graph_num_t = real_graph_num_t

        self.label = label

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


def convert_examples_to_features(examples, tokenizer, data):
    features = []
    for example in tqdm(examples):
        id1 = example[0]
        id2 = example[1]
        label = example[2]

        x1 = data['x'][id1]
        edge_index1 = data['edge_index'][id1]
        edge_attr1 = data['edge_attr'][id1]
        subgraph_node_num1 = data['subgraph_node_num'][id1]
        real_graph_num1 = data['real_graph_num'][id1]
        ast_des1 = tokenizer.tokenize(data['ast_des'][id1])[
            : max_source_length-2]
        ast_des1 = [tokenizer.cls_token] + ast_des1 + [tokenizer.sep_token]
        source_ids1 = tokenizer.convert_tokens_to_ids(ast_des1)
        padding_length = max_source_length - len(source_ids1)
        source_ids1 = source_ids1 + [tokenizer.pad_token_id] * padding_length

        x2 = data['x'][id2]
        edge_index2 = data['edge_index'][id2]
        edge_attr2 = data['edge_attr'][id2]
        subgraph_node_num2 = data['subgraph_node_num'][id2]
        real_graph_num2 = data['real_graph_num'][id2]
        ast_des2 = tokenizer.tokenize(data['ast_des'][id2])[
            : max_source_length-2]
        ast_des2 = [tokenizer.cls_token] + ast_des2 + [tokenizer.sep_token]
        source_ids2 = tokenizer.convert_tokens_to_ids(ast_des2)
        padding_length = max_source_length - len(source_ids2)
        source_ids2 = source_ids2 + [tokenizer.pad_token_id] * padding_length

        if data['ast_des_length'][id1] < 600 and data['ast_des_length'][id2] < 600:
            features.append(
                PairData(
                    x_s=torch.tensor(x1, dtype=torch.long),
                    edge_index_s=torch.tensor(edge_index1, dtype=torch.long),
                    edge_attr_s=torch.tensor(edge_attr1, dtype=torch.long),
                    source_ids_s=torch.tensor(source_ids1, dtype=torch.long),
                    subgraph_node_num_s=torch.tensor(
                        subgraph_node_num1, dtype=torch.long),
                    real_graph_num_s=torch.tensor(
                        real_graph_num1, dtype=torch.long),

                    x_t=torch.tensor(x2, dtype=torch.long),
                    edge_index_t=torch.tensor(edge_index2, dtype=torch.long),
                    edge_attr_t=torch.tensor(edge_attr2, dtype=torch.long),
                    source_ids_t=torch.tensor(source_ids2, dtype=torch.long),
                    subgraph_node_num_t=torch.tensor(
                        subgraph_node_num2, dtype=torch.long),
                    real_graph_num_t=torch.tensor(
                        real_graph_num2, dtype=torch.long),

                    label=torch.tensor(label, dtype=torch.long)
                )
            )
    return features


checkpoint = 'microsoft/codebert-base'
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
train_features = convert_examples_to_features(train_pairs, tokenizer, data)
valid_features = convert_examples_to_features(valid_pairs, tokenizer, data)
test_features = convert_examples_to_features(test_pairs, tokenizer, data)


torch.save(train_features, features_url+'train_features.pt')
torch.save(valid_features, features_url+'valid_features.pt')
torch.save(test_features, features_url+'test_features.pt')
