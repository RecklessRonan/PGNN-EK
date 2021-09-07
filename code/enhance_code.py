# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import javalang
from javalang.ast import Node
from tqdm import tqdm

# %% [markdown]
# Code Clone Detection

# %%
# raw_code_url = '/data/dataset/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl'
# raw_code = pd.read_json(path_or_buf=raw_code_url, lines=True)


# %%
# raw_code


# %%
# raw_code['func'][1]


# %%
java_api_url = '/data/code/represent-code-in-human/data/java_api.csv'
java_api = pd.read_csv(java_api_url, header=0, encoding='utf-8')
java_api['index_name'] = java_api['index_name'].apply(str)
# java_api


# %%
# use javalang to generate ASTs and depth-first traverse to generate ast nodes corpus
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
                yield from expand(item)
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence, api_sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    if token == 'MethodInvocation':
        api = [get_token(child) for child in children if not get_child(child)]
        # api_sequence.append(' '.join(api))
        if len(api) > 1:
            api_sequence.append(api[-1])
    for child in children:
        get_sequence(child, sequence, api_sequence)


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    return parser.parse_member_declaration()


# %%
# test_func = '''
#     public int larger(int a, int b) {
# 		a = Math.abs(a);
# 		b = Math.abs(b);
# 		if(a > b) {
# 			return a;
# 		}else {
# 			return b;
# 		}
# 	}
# '''
# sequence = []
# api_sequence = []
# get_sequence(parse_program(test_func), sequence, api_sequence)


# %%
# ' '.join(sequence)


# %%
# list(set(api_sequence))


# %%
def api_match(api_sequence, java_api):
    description_sequence = []
    for api in api_sequence:
        loc = java_api.loc[java_api['index_name'].str.contains(api, case=True)]
        if not loc.empty:
            description = loc['method_description'].iloc[0]
            if description != 'None':
                description_sequence.append(description)
    return description_sequence


# %%

# description_sequence = []
# for i in tqdm(range(len(raw_code))):
#     sequence = []
#     api_sequence = []
#     get_sequence(parse_program(raw_code['func'][i]), sequence, api_sequence)
#     api_sequence = list(set(api_sequence))
#     # print('api_sequence', api_sequence)
#     description = '\n'.join(api_match(api_sequence, java_api))
#     # print('description', description)
#     description_sequence.append(description)


# %%
# description_sequence[0]


# %%
# raw_code['description'] = description_sequence


# %%
# raw_code


# %%
# ast_sequence = []
# for i in tqdm(range(len(raw_code))):
#     sequence = []
#     api_sequence = []
#     get_sequence(parse_program(raw_code['func'][i]), sequence, api_sequence)
#     ast_sequence.append(' '.join(sequence))


# %%
# raw_code['ast'] = ast_sequence


# %%
# raw_code.to_json(path_or_buf='/data/dataset/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset/data_enhanced.jsonl',
#                  orient='records', lines=True)


# %%
# raw_code


# %%
# sequence1 = []
# api_sequence1 = []
# get_sequence(parse_program(test_func), sequence1, api_sequence1)
# description = '\n'.join(api_match(api_sequence1, java_api))
# ast_sequence.append(' '.join(sequence))


# %%
# description


# %%
# ' '.join(sequence1)

# %% [markdown]
# Code Summarization

# %%
TRAIN_DIR = '/data/code/represent-code-in-human/data/code-summarization-new/train.jsonl'
VALID_DIR = '/data/code/represent-code-in-human/data/code-summarization-new/valid.jsonl'
TEST_DIR = '/data/code/represent-code-in-human/data/code-summarization-new/test.jsonl'


# %%
# read dataset
train_data = pd.read_json(path_or_buf=TRAIN_DIR, lines=True)
valid_data = pd.read_json(path_or_buf=VALID_DIR, lines=True)
test_data = pd.read_json(path_or_buf=TEST_DIR, lines=True)


# %%
train_data = train_data.sample(random_state=555, frac=1)
valid_data = valid_data.sample(random_state=555, frac=1)
test_data = test_data.sample(random_state=555, frac=1)


# %%
def get_ast_and_description(data):
    description_sequence = []
    ast_sequence = []
    ast_sum = 0
    description_sum = 0
    data_size = len(data)
    for i in tqdm(range(data_size)):
        sequence = []
        api_sequence = []
        get_sequence(parse_program(
            data['code'].iloc[i]), sequence, api_sequence)
        ast = ' '.join(sequence)
        ast_sequence.append(ast)
        ast_sum += len(ast.split(' '))

        api_sequence = list(set(api_sequence))
        description = ' '.join(api_match(api_sequence, java_api))
        description_sequence.append(description)
        description_sum += len(description.split(' '))
    print('ast average length', ast_sum/data_size)
    print('description average length', description_sum/data_size)
    return description_sequence, ast_sequence


# %%
train_description, train_ast = get_ast_and_description(train_data)
valid_description, valid_ast = get_ast_and_description(valid_data)
test_description, test_ast = get_ast_and_description(test_data)


# %%
train_data['des'] = train_description
train_data['ast'] = train_ast
train_data['ast_des'] = train_data['ast'] + ' ' + train_data['des']

valid_data['des'] = valid_description
valid_data['ast'] = valid_ast
valid_data['ast_des'] = valid_data['ast'] + ' ' + valid_data['des']

test_data['des'] = test_description
test_data['ast'] = test_ast
test_data['ast_des'] = test_data['ast'] + ' ' + test_data['des']


# %%
# train_data.to_json(path_or_buf='/data/code/represent-code-in-human/data/code-summarization-enhanced-middle/train.jsonl',
#                      orient='records', lines=True)
# valid_data.to_json(path_or_buf='/data/code/represent-code-in-human/data/code-summarization-enhanced-middle/valid.jsonl',
#                      orient='records', lines=True)
# test_data.to_json(path_or_buf='/data/code/represent-code-in-human/data/code-summarization-enhanced-middle/test.jsonl',
#                      orient='records', lines=True)


# %%
train_data.to_json(path_or_buf='/data/code/represent-code-in-human/data/code-summarization-enhanced-full/train.jsonl',
                   orient='records', lines=True)
valid_data.to_json(path_or_buf='/data/code/represent-code-in-human/data/code-summarization-enhanced-full/valid.jsonl',
                   orient='records', lines=True)
test_data.to_json(path_or_buf='/data/code/represent-code-in-human/data/code-summarization-enhanced-full/test.jsonl',
                  orient='records', lines=True)
