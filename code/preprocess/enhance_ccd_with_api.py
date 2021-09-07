from threading import main_thread
import javalang
from javalang.ast import Node
import pandas as pd
from multiprocessing import Process, cpu_count, Manager, Pool


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
    # find api
    if token == 'MethodInvocation':
        api = [get_token(child) for child in children if not get_child(child)]
        # api_sequence.append(' '.join(api))
        if len(api) > 1:
            api_sequence.append(api[-1])
    for child in children:
        get_sequence(child, sequence, api_sequence)


def api_match(api_sequence, java_api):
    description_sequence = []
    for api in api_sequence:
        loc = java_api.loc[java_api['index_name'].str.contains(api, case=True)]
        if not loc.empty:
            description = loc['method_description'].iloc[0]
            if description != 'None':
                description_sequence.append(description)
    return description_sequence


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    return parser.parse_member_declaration()


dataset_url = '../../data/BCB-F/'
data_url = dataset_url + 'data.jsonl'
data = pd.read_json(path_or_buf=data_url, lines=True)
java_api_url = '../../data/java-api/java_api.csv'
java_api = pd.read_csv(java_api_url, header=0, encoding='utf-8')
java_api['index_name'] = java_api['index_name'].apply(str)


# multi-process
def multi_get_ast_and_des(l, i):
    # print('run2')
    sequence = []
    api_sequence = []
    # print('sequence', sequence)
    get_sequence(parse_program(data['func'].iloc[i]), sequence, api_sequence)
    # print('sequence', sequence)
    ast = ' '.join(sequence)
    api_sequence = list(set(api_sequence))
    des = ' '.join(api_match(api_sequence, java_api))
    d = {'ast': ast, 'des': des, 'i': i}
    # print('d', d)
    l.append(d)


manager = Manager()
data_size = len(data)
# print('data_size', data_size)
l = manager.list()
p = Pool(processes=30)
for i in range(data_size):
    # print('run1')
    p.apply_async(multi_get_ast_and_des, (l, i))
p.close()
p.join()

ast = []
des = []
i = []
for d in l[:]:
    # print('i', i)
    ast.append(d['ast'].encode('utf-8', 'ignore').decode("utf-8"))
    des.append(d['des'].encode('utf-8', 'ignore').decode("utf-8"))
    i.append(d['i'])
d = {'ast': ast, 'des': des, 'i': i}
df = pd.DataFrame.from_dict(d)
df = df.sort_values(by=['i']).reset_index(drop=True)
data['ast'] = df['ast'].to_list()
data['des'] = df['des'].to_list()
data['ast_des'] = data['ast'] + ' ' + data['des']
data.to_json(path_or_buf=dataset_url + 'data_enhanced.jsonl',
             orient='records', lines=True)
