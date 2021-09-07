import pandas as pd
from multiprocessing import Manager, Pool
import argparse
from api_match import get_sequence, parse_program, api_match
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process different dataset')
parser.add_argument('--dataset', type=str, help='CSN or TLC')
args = parser.parse_args()
if args.dataset == 'CSN':
    dataset_url = '../../data/CSN/'
elif args.dataset == 'TLC':
    dataset_url = '../../data/TLC/'
else:
    print('Wrong dataset name')


# read data and java api
train_url = dataset_url + 'train.jsonl'
valid_url = dataset_url + 'valid.jsonl'
test_url = dataset_url + 'test.jsonl'

train_data = pd.read_json(path_or_buf=train_url, lines=True)
valid_data = pd.read_json(path_or_buf=valid_url, lines=True)
test_data = pd.read_json(path_or_buf=test_url, lines=True)

java_api_url = '../../data/java-api/java_api.csv'
java_api = pd.read_csv(java_api_url, header=0, encoding='utf-8')
java_api['index_name'] = java_api['index_name'].apply(str)


# Delete the codes that cannot be parsed by javalang and shuffle the code
def get_syntax_error_ids(data):
    syntax_error_ids = []
    for i in tqdm(range(len(data['code']))):
        try:
            tree = parse_program(data['code'][i])
        except:
            syntax_error_ids.append(i)
    return syntax_error_ids


train_syntax_error_ids = get_syntax_error_ids(train_data)
valid_syntax_error_ids = get_syntax_error_ids(valid_data)
test_syntax_error_ids = get_syntax_error_ids(test_data)
train_data_new = train_data.drop(train_syntax_error_ids).sample(
    frac=1).reset_index(drop=True)
valid_data_new = valid_data.drop(valid_syntax_error_ids).sample(
    frac=1).reset_index(drop=True)
test_data_new = test_data.drop(test_syntax_error_ids).sample(
    frac=1).reset_index(drop=True)


# multi-process train data
data_new = train_data_new


def multi_get_ast_and_des(l, i):
    sequence = []
    api_sequence = []
    get_sequence(parse_program(
        data_new['code'].iloc[i]), sequence, api_sequence)
    ast = ' '.join(sequence)
    api_sequence = list(set(api_sequence))
    des = ' '.join(api_match(api_sequence, java_api))
    d = {'ast': ast, 'des': des, 'i': i}
    l.append(d)


manager = Manager()
data_size = len(data_new)
# print('data_size', data_size)
l = manager.list()
p = Pool(processes=30)
for i in range(data_size):
    p.apply_async(multi_get_ast_and_des, (l, i))
p.close()
p.join()

ast = []
des = []
i = []
for d in l[:]:
    ast.append(d['ast'].encode('utf-8', 'ignore').decode("utf-8"))
    des.append(d['des'].encode('utf-8', 'ignore').decode("utf-8"))
    i.append(d['i'])
d = {'ast': ast, 'des': des, 'i': i}
train_df = pd.DataFrame.from_dict(d)


# multi-process valid_data
data_new = valid_data_new


def multi_get_ast_and_des(l, i):
    sequence = []
    api_sequence = []
    get_sequence(parse_program(
        data_new['code'].iloc[i]), sequence, api_sequence)
    ast = ' '.join(sequence)
    api_sequence = list(set(api_sequence))
    des = ' '.join(api_match(api_sequence, java_api))
    d = {'ast': ast, 'des': des, 'i': i}
    l.append(d)


manager = Manager()
data_size = len(data_new)
# print('data_size', data_size)
l = manager.list()
p = Pool(processes=30)
for i in range(data_size):
    p.apply_async(multi_get_ast_and_des, (l, i))
p.close()
p.join()

ast = []
des = []
i = []
for d in l[:]:
    ast.append(d['ast'].encode('utf-8', 'ignore').decode("utf-8"))
    des.append(d['des'].encode('utf-8', 'ignore').decode("utf-8"))
    i.append(d['i'])
d = {'ast': ast, 'des': des, 'i': i}
valid_df = pd.DataFrame.from_dict(d)


# multi-process test data
data_new = test_data_new


def multi_get_ast_and_des(l, i):
    sequence = []
    api_sequence = []
    get_sequence(parse_program(
        data_new['code'].iloc[i]), sequence, api_sequence)
    ast = ' '.join(sequence)
    api_sequence = list(set(api_sequence))
    des = ' '.join(api_match(api_sequence, java_api))
    d = {'ast': ast, 'des': des, 'i': i}
    l.append(d)


manager = Manager()
data_size = len(data_new)
# print('data_size', data_size)
l = manager.list()
p = Pool(processes=30)
for i in range(data_size):
    p.apply_async(multi_get_ast_and_des, (l, i))
p.close()
p.join()

ast = []
des = []
i = []
for d in l[:]:
    ast.append(d['ast'].encode('utf-8', 'ignore').decode("utf-8"))
    des.append(d['des'].encode('utf-8', 'ignore').decode("utf-8"))
    i.append(d['i'])
d = {'ast': ast, 'des': des, 'i': i}
test_df = pd.DataFrame.from_dict(d)


train_df = train_df.sort_values(by=['i']).reset_index(drop=True)
train_data_new['ast'] = train_df['ast'].to_list()
train_data_new['des'] = train_df['des'].to_list()
train_data_new['ast_des'] = train_data_new['ast'] + ' ' + train_data_new['des']

valid_df = valid_df.sort_values(by=['i']).reset_index(drop=True)
valid_data_new['ast'] = valid_df['ast'].to_list()
valid_data_new['des'] = valid_df['des'].to_list()
valid_data_new['ast_des'] = valid_data_new['ast'] + ' ' + valid_data_new['des']

test_df = test_df.sort_values(by=['i']).reset_index(drop=True)
test_data_new['ast'] = test_df['ast'].to_list()
test_data_new['des'] = test_df['des'].to_list()
test_data_new['ast_des'] = test_data_new['ast'] + ' ' + test_data_new['des']


train_data_new.to_json(path_or_buf=dataset_url + 'train_enhanced.jsonl',
                       orient='records', lines=True)
valid_data_new.to_json(path_or_buf=dataset_url + 'valid_enhanced.jsonl',
                       orient='records', lines=True)
test_data_new.to_json(path_or_buf=dataset_url + 'test_enhanced.jsonl',
                      orient='records', lines=True)
