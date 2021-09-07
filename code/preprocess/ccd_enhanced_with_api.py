import pandas as pd
from multiprocessing import Manager, Pool
import argparse
from api_match import get_sequence, parse_program, api_match


parser = argparse.ArgumentParser(description='Process different dataset')
parser.add_argument('--dataset', type=str, help='BCB or BCB-F')
args = parser.parse_args()
if args.dataset == 'BCB':
    dataset_url = '../../data/BCB/'
elif args.dataset == 'BCB-F':
    dataset_url = '../../data/BCB-F/'
else:
    print('Wrong dataset name')
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
