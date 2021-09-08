from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, DataCollatorWithPadding
import torch
from torch_geometric.data import Data
import pandas as pd
import argparse
from sast_construct import get_subgraph_node_num, get_pyg_data_from_ast, parse_program
from tqdm import tqdm
import yaml
import json


parser = argparse.ArgumentParser(description='Generate features for model')
parser.add_argument('--dataset', type=str, help='TLC or CSN')

args = parser.parse_args()
if args.dataset == 'TLC':
    dataset_url = '../../data/TLC/'
    features_url = '../features/TLC/'
elif args.dataset == 'CSN':
    dataset_url = '../../data/CSN/'
    features_url = '../features/CSN/'
else:
    print('Wrong dataset name')

# Configuration
config_file = '../configs/config_cs.yml'
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
max_source_length = config['preprocess']['max_source_length']
max_target_length = config['preprocess']['max_target_length']
divide_node_num = config['preprocess']['divide_node_num']
max_node_num = config['preprocess']['max_node_num']
max_subgraph_num = int(args.max_node_num/divide_node_num)


# define tokenizer
checkpoint = 'microsoft/codebert-base'
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
ast_tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
roberta = RobertaModel.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
config = RobertaConfig.from_pretrained(checkpoint)
javalang_special_tokens = ['CompilationUnit', 'Import', 'Documented', 'Declaration', 'TypeDeclaration', 'PackageDeclaration',
                           'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationDeclaration', 'Type',
                           'BasicType', 'ReferenceType', 'TypeArgument', 'TypeParameter', 'Annotation', 'ElementValuePair',
                           'ElementArrayValue', 'Member', 'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration',
                           'ConstantDeclaration', 'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration',
                           'VariableDeclarator', 'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement',
                           'WhileStatement', 'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement', 'ContinueStatement',
                           'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement', 'SwitchStatement',
                           'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause', 'CatchClauseParameter',
                           'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression', 'Assignment', 'TernaryExpression',
                           'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression', 'Primary', 'Literal', 'This',
                           'MemberReference', 'Invocation', 'ExplicitConstructorInvocation', 'SuperConstructorInvocation',
                           'MethodInvocation', 'SuperMethodInvocation', 'SuperMemberReference', 'ArraySelector', 'ClassReference',
                           'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator', 'InnerClassCreator', 'EnumBody',
                           'EnumConstantDeclaration', 'AnnotationMethod', 'Modifier']
special_tokens_dict = {'additional_special_tokens': javalang_special_tokens}
num_added_toks = ast_tokenizer.add_special_tokens(special_tokens_dict)


# read dataset

class Example(object):
    def __init__(self, idx, source, ast_des, target):
        self.idx = idx
        self.source = source
        self.ast_des = ast_des
        self.target = target


def read_examples(filename):
    examples = []
    with open(filename, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code = js['code']
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            ast_des = js['ast_des']
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    ast_des=ast_des,
                    target=nl,
                )
            )
    return examples


train_examples = read_examples(dataset_url +
                               'train_enhanced.jsonl')
valid_examples = read_examples(dataset_url +
                               'valid_enhanced.jsonl')
test_examples = read_examples(dataset_url +
                              'test_enhanced.jsonl')


def convert_examples_to_features(examples, ast_tokenizer, tokenizer, stage=None):
    features = []
    for example in tqdm(examples):
        # pyg
        ast = parse_program(example.source)
        x, edge_index, edge_attr, root_children_node_num = get_pyg_data_from_ast(
            ast, ast_tokenizer)
        subgraph_node_num, real_graph_num = get_subgraph_node_num(
            root_children_node_num, divide_node_num, max_subgraph_num)

        # source
        source_tokens = tokenizer.tokenize(example.ast_des)[
            : max_source_length-2]
        source_tokens = [tokenizer.cls_token] + \
            source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_ids))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == 'test':
            target_tokens = tokenizer.tokenize('None')
        else:
            target_tokens = tokenizer.tokenize(example.target)[
                : max_target_length-2]
        target_tokens = [tokenizer.cls_token] + \
            target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            Data(
                x=torch.tensor(x, dtype=torch.long),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.long),
                source_ids=torch.tensor(source_ids, dtype=torch.long),
                source_mask=torch.tensor(source_mask, dtype=torch.long),
                target_ids=torch.tensor(target_ids, dtype=torch.long),
                target_mask=torch.tensor(target_mask, dtype=torch.long),
                subgraph_node_num=torch.tensor(
                    subgraph_node_num, dtype=torch.long),
                real_graph_num=torch.tensor(real_graph_num, dtype=torch.long)
            )
        )
    return features


train_features = convert_examples_to_features(
    train_examples, ast_tokenizer, tokenizer, stage='train')
valid_features = convert_examples_to_features(
    valid_examples, ast_tokenizer, tokenizer, stage='valid')
test_features = convert_examples_to_features(
    test_examples, ast_tokenizer, tokenizer, stage='test')
torch.save(train_features, features_url + 'train_features.pt')
torch.save(valid_features, features_url + 'valid_features.pt')
torch.save(test_features, features_url + 'test_features.pt')
