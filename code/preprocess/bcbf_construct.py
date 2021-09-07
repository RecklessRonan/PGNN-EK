import psycopg2
import pandas as pd

# you must set your postgres database configuration
database = ''
user = ''
password = ''
host = ''
port = ''

conn = psycopg2.connect(database=database, user=user,
                        password=password, host=host, port=port)
cur = conn.cursor()

# remove the codes that cannot be parsed by javalang
syntax_error_ids = [3273863, 3867253, 4191191, 4191192, 4519111, 4519112, 5264639, 9532844, 9830264, 11714040,
                    13525465, 14507873, 16226644, 16532114, 17286897, 18347503, 22561089, 23117206]


# read positive samples from database
select_pos_full = 'select function_id_one, function_id_two, functionality_id, syntactic_type, similarity_line, similarity_token from clones'
cur.execute(select_pos_full)
pos_full = cur.fetchall()
pos_full_set = list(set(pos_full))
pos_full_set_right = [
    pair
    for pair in pos_full_set
    if pair[0] not in syntax_error_ids and pair[1] not in syntax_error_ids
]


# read negative samples from database
select_neg_full = 'select function_id_one, function_id_two, functionality_id, syntactic_type, similarity_line, similarity_token from false_positives'
cur.execute(select_neg_full)
neg_full = cur.fetchall()
neg_full_set = list(set(neg_full))
neg_full_set_right = [
    pair
    for pair in neg_full_set
    if pair[0] not in syntax_error_ids and pair[1] not in syntax_error_ids
]


# convert to label
id1 = []
id2 = []
label = []
func_id = []
for pair in pos_full_set_right:
    id1.append(pair[0])
    id2.append(pair[1])
    func_id.append(pair[2])
    if pair[3] == 3:
        similarity = min(pair[4], pair[5])
        if similarity < 0.5:
            label.append(5)
        elif similarity > 0.7:
            label.append(3)
        else:
            label.append(4)
    else:
        label.append(pair[3])

for pair in neg_full_set_right:
    id1.append(pair[0])
    id2.append(pair[1])
    label.append(0)
    func_id.append(pair[2])

bcb2015_pair = {'id1': id1, 'id2': id2, 'label': label, 'func_id': func_id}
bcb2015_pair_df = pd.DataFrame(data=bcb2015_pair)
print('-' * 50)
print('BCB 2015 example:')
print(bcb2015_pair_df.head())
print('Label distribution')
print(bcb2015_pair_df['label'].value_counts())


neg_pair = bcb2015_pair_df[bcb2015_pair_df['label'].isin([0])]
pos_pair = bcb2015_pair_df[~bcb2015_pair_df['label'].isin([0])]


# divide train/valid/test by functionality
train_df = bcb2015_pair_df[~bcb2015_pair_df['func_id'].isin(
    [7, 6, 30, 17, 26, 25, 45, 22, 34, 35, 14, 10, 13, 43, 44, 29, 18, 12, 40, 24])]
valid_df = bcb2015_pair_df[bcb2015_pair_df['func_id'].isin(
    [7, 6, 30, 17, 26, 25, 45, 22, 34, 35, 14])]
test_df = bcb2015_pair_df[bcb2015_pair_df['func_id'].isin(
    [10, 13, 43, 44, 29, 18, 12, 40, 24])]


train_neg_label = train_df[(train_df['label'] == 0)]
train_pos_label = train_df[~(train_df['label'] == 0)].sample(
    len(train_neg_label), random_state=555)
train_label = pd.concat([train_pos_label, train_neg_label], axis=0)
train_label = train_label.sample(frac=1, random_state=555).drop(
    ['func_id'], axis=1).reset_index(drop=True)


valid_neg_label = valid_df[(valid_df['label'] == 0)]
valid_pos_label = valid_df[~(valid_df['label'] == 0)].sample(
    len(valid_neg_label), random_state=555)
valid_label = pd.concat([valid_pos_label, valid_neg_label], axis=0)
valid_label = valid_label.sample(frac=1, random_state=555).drop(
    ['func_id'], axis=1).reset_index(drop=True)

test_neg_label = test_df[(test_df['label'] == 0)]
test_pos_label = test_df[~(test_df['label'] == 0)].sample(
    len(test_neg_label), random_state=555)
test_label = pd.concat([test_pos_label, test_neg_label], axis=0)
test_label = test_label.sample(frac=1, random_state=555).drop(
    ['func_id'], axis=1).reset_index(drop=True)


train_str = ''
for i in range(len(train_label)):
    train_str += str(train_label['id1'][i])
    train_str += '\t'
    train_str += str(train_label['id2'][i])
    train_str += '\t'
    train_str += '0' if train_label['label'][i] == 0 else '1'
    train_str += '\n'

with open('../../data/BCB-F/train.txt', 'w', encoding='utf-8') as f:
    f.write(train_str)


valid_str = ''
for i in range(len(valid_label)):
    valid_str += str(valid_label['id1'][i])
    valid_str += '\t'
    valid_str += str(valid_label['id2'][i])
    valid_str += '\t'
    valid_str += '0' if valid_label['label'][i] == 0 else '1'
    valid_str += '\n'

with open('../../data/BCB-F/valid.txt', 'w', encoding='utf-8') as f:
    f.write(valid_str)

test_str = ''
for i in range(len(test_label)):
    test_str += str(test_label['id1'][i])
    test_str += '\t'
    test_str += str(test_label['id2'][i])
    test_str += '\t'
    test_str += '0' if test_label['label'][i] == 0 else '1'
    test_str += '\n'

with open('../../data/BCB-F/test.txt', 'w', encoding='utf-8') as f:
    f.write(test_str)
