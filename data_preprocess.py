import pandas as pd
import javalang
from javalang.ast import Node
from tqdm import tqdm
from anytree import AnyNode
from torch_geometric.data import Data
import torch

# read the json data
# data_url = '/data/code/represent-code-in-human/data/code-summarization-new/'
# the small version of dataset for experimental test
data_url = '/data/code/represent-code-in-human/data/code-summarization-small/'

train_data_url = data_url + 'train.csv'
valid_data_url = data_url + 'valid.csv'
test_data_url = data_url + 'test.csv'

train_data = pd.read_csv(train_data_url, encoding='utf-8')
valid_data = pd.read_csv(valid_data_url, encoding='utf-8')
test_data = pd.read_csv(test_data_url, encoding='utf-8')


# generate AST trees for codes
def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


# get asts and labels for all data
# generate text(summarization) vocab for all data
def get_asts_and_labels():
    train_asts = []
    valid_asts = []
    test_asts = []
    train_labels = []
    valid_labels = []
    test_labels = []

    all_text_tokens = []
    for doc in train_data['docstring_tokens']:
        # print('doc[1:-1]', doc[1:-1])
        for token in doc[1:-1].split(', '):
            # print('token', token[1:-1])
            all_text_tokens.append(token[1:-1])

    for doc in valid_data['docstring_tokens']:
        for token in doc[1:-1].split(', '):
            all_text_tokens.append(token[1:-1])

    for doc in test_data['docstring_tokens']:
        for token in doc[1:-1].split(', '):
            all_text_tokens.append(token[1:-1])

    print('text tokens num ', len(all_text_tokens))
    all_text_tokens = list(set(all_text_tokens))
    text_vocab_len = len(all_text_tokens)
    text_token_ids = range(text_vocab_len)
    text_vocab_dict = dict(zip(all_text_tokens, text_token_ids))
    print('text vocab_len ', text_vocab_len)

    for i in range(len(train_data)):
        labels = []
        train_asts.append(parse_program(train_data['code'][i]))
        for token in train_data['docstring_tokens'][i][1:-1].split(', '):
            # print('docstring_tokens', train_data['docstring_tokens'][i])
            # print('token', token)
            # print('token[1:-1]', token[1:-1])
            labels.append(text_vocab_dict[token[1:-1]])
        train_labels.append(labels)

    # print('train_labels', train_labels)
    for i in range(len(valid_data)):
        labels = []
        valid_asts.append(parse_program(valid_data['code'][i]))
        for token in valid_data['docstring_tokens'][i][1:-1].split(', '):
            labels.append(text_vocab_dict[token[1:-1]])
        valid_labels.append(labels)

    for i in range(len(test_data)):
        labels = []
        test_asts.append(parse_program(test_data['code'][i]))
        for token in test_data['docstring_tokens'][i][1:-1].split(', '):
            labels.append(text_vocab_dict[token[1:-1]])
        test_labels.append(labels)

    return train_asts, valid_asts, test_asts, train_labels, valid_labels, test_labels, text_vocab_len, text_vocab_dict


# generate token vocab in AST for all data
def get_code_token_vocab(train_asts, valid_asts, test_asts):
    all_code_tokens = []
    print('get all tokens of train data...')
    for ast in train_asts:
        sequence = []
        get_sequence(ast, sequence)
        for s in sequence:
            all_code_tokens.append(s)
    print('get all tokens of valid data...')
    for ast in valid_asts:
        sequence = []
        get_sequence(ast, sequence)
        for s in sequence:
            all_code_tokens.append(s)
    print('get all tokens of test data...')
    for ast in test_asts:
        sequence = []
        get_sequence(ast, sequence)
        for s in sequence:
            all_code_tokens.append(s)
    print('code tokens num ', len(all_code_tokens))
    all_code_tokens = list(set(all_code_tokens))
    code_vocab_len = len(all_code_tokens)
    code_token_ids = range(code_vocab_len)
    code_vocab_dict = dict(zip(all_code_tokens, code_token_ids))
    print('code vocab_len ', code_vocab_len)
    return code_vocab_len, code_vocab_dict


# traverse the AST tree to get all the nodes and edges
def get_node_and_edge(node, node_index_list, vocab_dict, src, tgt):
    token = node.token
    node_index_list.append([vocab_dict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        get_node_and_edge(child, node_index_list, vocab_dict, src, tgt)


#  Generate tree for AST Node
def create_tree(root, node, node_list, parent=None):
    id = len(node_list)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        new_node = AnyNode(id=id, token=token, data=node, parent=parent)
    node_list.append(node)
    for child in children:
        if id == 0:
            create_tree(root, child, node_list, parent=root)
        else:
            create_tree(root, child, node_list, parent=new_node)


# transfer ast generated by javalang to torch_geometric.data
def trans_ast_to_tgdata(asts, vocab_dict, labels):
    tgdatas = []
    for ast, label in zip(asts, labels):
        node_list = []
        new_tree = AnyNode(id=0, token=None, data=None)
        create_tree(new_tree, ast, node_list)
        x = []
        edge_src = []
        edge_tgt = []
        edge_attr = []
        get_node_and_edge(new_tree, x, vocab_dict, edge_src, edge_tgt)
        edge_index = [edge_src, edge_tgt]
        y = torch.tensor(label, dtype=torch.long)
        # print('y', y.shape[0])
        # set max length to 32, use zero tensor for testing
        max_length = 32
        if y.shape[0] >= max_length:
            y = y[0: max_length]
        else:
            zeros = torch.zeros(max_length - y.shape[0], dtype=torch.long)
            # print('zeros', zeros)
            y = torch.cat((y, zeros), dim=0)
        # print('y', y.shape)
        tgdatas.append(Data(x=torch.tensor(x, dtype=torch.long),
                       edge_index=torch.tensor(edge_index, dtype=torch.long),
                       edge_attr=torch.tensor(edge_attr, dtype=torch.long),
                       y=y))
    return tgdatas


# create data for ggnn input on code summarization task
def create_ggnn_cs_data(train_asts, valid_asts, test_asts, train_labels, valid_labels, test_labels, vocab_dict):
    # vocab_len, vocab_dict = get_token_vocab(train_asts, valid_asts, test_asts)
    train_tgdatas = trans_ast_to_tgdata(train_asts, vocab_dict, train_labels)
    valid_tgdatas = trans_ast_to_tgdata(valid_asts, vocab_dict, valid_labels)
    test_tgdatas = trans_ast_to_tgdata(test_asts, vocab_dict, test_labels)
    return train_tgdatas, valid_tgdatas, test_tgdatas
