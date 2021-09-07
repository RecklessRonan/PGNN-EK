from itertools import cycle
import yaml
import warnings
from datetime import datetime
import logging
import os
import torch
import torch.nn.functional as F
from torch.nn import (
    Module, Linear, CrossEntropyLoss)
from transformers import (RobertaTokenizer, RobertaConfig, RobertaModel,
                          DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup)
from torch_geometric.data import Data, DataLoader
from pgnn import GNNEncoder
from torch.utils.data import RandomSampler
import numpy as np
from tqdm import tqdm
import argparse
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Generate features for model')
parser.add_argument('--dataset', type=str, help='BCB or BCB-F')
args = parser.parse_args()
if args.dataset == 'BCB':
    dataset_url = '../../data/BCB/'
    features_url = '../features/BCB/'
elif args.dataset == 'BCB-F':
    dataset_url = '../../data/BCB-F/'
    features_url = '../features/BCB-F/'
else:
    print('Wrong dataset name')


# Configuration
config_file = '../configs/config_cs.yml'
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
info_prefix = config['logs']['info_prefix']
# Data source
TRAIN_DIR = dataset_url + 'train.jsonl'
VALID_DIR = dataset_url + 'valid.jsonl'
TEST_DIR = dataset_url + 'test.jsonl'
# Preprocess
max_source_length = config['preprocess']['max_source_length']
divide_node_num = config['preprocess']['divide_node_num']
max_node_num = config['preprocess']['max_node_num']
max_subgraph_num = int(max_node_num/divide_node_num)
# Model design
graph_embedding_size = config['model']['graph_embedding_size']
lstm_hidden_size = config['model']['lstm_hidden_size']
gnn_layers_num = config['model']['gnn_layers_num']
lstm_layers_num = config['model']['lstm_layers_num']
decoder_input_size = config['model']['decoder_input_size']
siamese_input_size = config['model']['siamese_input_size']
# Training setting
batch_size = config['training']['batch_size']
lr = config['training']['lr']
warmup_steps = config['training']['warmup_steps']
train_steps = config['training']['train_steps']
weight_decay = config['training']['weight_decay']
adam_epsilon = config['training']['adam_epsilon']
valid_loss_steps = config['training']['valid_loss_steps']
valid_bleu_steps = config['training']['valid_bleu_steps']

# Logs
run_id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
log_file = '../../logs/' + run_id + '.log'
exp_dir = '../../runs/' + run_id
os.mkdir(exp_dir)


class Info(object):
    def __init__(self, info_prefix=''):
        self.info_prefix = info_prefix

    def print_msg(self, msg):
        text = self.info_prefix + ' ' + msg
        print(text)
        logging.info(text)


logging.basicConfig(format='%(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, level=logging.DEBUG)
msgr = Info(info_prefix)

msgr.print_msg('run_id : {}'.format(run_id))
msgr.print_msg('log_file : {}'.format(log_file))
msgr.print_msg('exp_dir: {}'.format(exp_dir))
msgr.print_msg(str(config))


# Define a pairdata for pytorch geometric
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


# Load features
train_features = torch.load(
    features_url + 'train_features.pt')
valid_features = torch.load(
    features_url + 'valid_features.pt')
test_features = torch.load(
    features_url + 'test_features.pt')


class MixEncoder(Module):
    def __init__(self, gnn_encoder, roberta, decoder_input_size, hidden_size, siamese_input_size):
        super(MixEncoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.roberta = roberta
        self.fc = Linear(decoder_input_size + hidden_size, siamese_input_size)

    def forward(self, x, edge_index, edge_attr, subgraph_node_num, real_graph_num, batch, ptr, source_ids, source_mask):
        output1 = self.gnn_encoder(
            x, edge_index, edge_attr, subgraph_node_num, real_graph_num, batch, ptr)
        output2 = self.roberta(source_ids, source_mask).pooler_output
        output = torch.cat((output1, output2), dim=1)
        return self.fc(output)


class Model(Module):
    def __init__(self, mix_encoder, siamese_input_size):
        super(Model, self).__init__()
        self.mix_encoder = mix_encoder
        self.fc1 = Linear(2 * siamese_input_size, siamese_input_size)
        self.fc2 = Linear(siamese_input_size, 2)

    def forward(self, x1, edge_index1, edge_attr1, subgraph_node_num1, real_graph_num1, batch1, ptr1, source_ids1, source_mask1,
                x2, edge_index2, edge_attr2, subgraph_node_num2, real_graph_num2, batch2, ptr2, source_ids2, source_mask2):
        output1 = self.mix_encoder(x1, edge_index1, edge_attr1, subgraph_node_num1,
                                   real_graph_num1, batch1, ptr1, source_ids1, source_mask1)
        output2 = self.mix_encoder(x2, edge_index2, edge_attr2, subgraph_node_num2,
                                   real_graph_num2, batch2, ptr2, source_ids2, source_mask2)
        output = torch.cat((output1, output2), dim=1)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


# Load CodeBERT related
checkpoint = 'microsoft/codebert-base'
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
ast_tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
roberta = RobertaModel.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
roberta_config = RobertaConfig.from_pretrained(checkpoint)
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


# Model Initilization
device = torch.device('cuda:0')
gnn_encoder = GNNEncoder(vocab_len=tokenizer.vocab_size+num_added_toks, graph_embedding_size=graph_embedding_size,
                         gnn_layers_num=gnn_layers_num, lstm_layers_num=lstm_layers_num, lstm_hidden_size=lstm_hidden_size,
                         decoder_input_size=decoder_input_size, device=device)
mix_encoder = MixEncoder(gnn_encoder=gnn_encoder, roberta=roberta, decoder_input_size=decoder_input_size,
                         hidden_size=roberta_config.hidden_size, siamese_input_size=siamese_input_size)
model = Model(mix_encoder=mix_encoder, siamese_input_size=siamese_input_size)
model.to(device)


# optimizer and schedule
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [
            p
            for n, p in model.named_parameters()
            if all(nd not in n for nd in no_decay)
        ],
        'weight_decay': weight_decay,
    },
    {
        'params': [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay': 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000,
#                                             num_training_steps=30000)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=train_steps)


# used for pytorch geometric data
def get_ptr_from_batch(batch):
    #     print(batch[-1])
    ptr = [batch.tolist().index(i) for i in range(batch[-1] + 1)]
    ptr.append(batch.size(0))
    return torch.tensor(ptr, dtype=torch.long)


# Start training
msgr.print_msg("***** Running training *****")
msgr.print_msg("  Num examples = {}".format(len(train_features)))
msgr.print_msg("  Batch size = {}".format(batch_size))
msgr.print_msg("  lr= {}".format(lr))
msgr.print_msg("  Num epoch = {}".format(batch_size//len(train_features)))
model.train()
nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_f1, best_loss = 0, 0, 0, 0, 0, 1e6
bar = tqdm(range(train_steps), total=train_steps)
train_dataloader = DataLoader(
    train_features, batch_size=batch_size, shuffle=True, follow_batch=['x_s', 'x_t'])
train_dataloader = cycle(train_dataloader)
loss_func = CrossEntropyLoss()
output_dir = exp_dir

for step in bar:
    data = next(train_dataloader)
    data = data.to(device)
    x1 = data.x_s
    edge_index1 = data.edge_index_s
    edge_attr1 = data.edge_attr_s
    subgraph_node_num1 = torch.stack(torch.split(
        data.subgraph_node_num_s, max_subgraph_num))
    real_graph_num1 = torch.stack(torch.split(data.real_graph_num_s, 1))
    source_ids1 = torch.stack(torch.split(
        data.source_ids_s, max_source_length))
    source_mask1 = torch.stack(torch.split(
        data.source_ids_s.ne(1), max_source_length))
    batch1 = data.x_s_batch
    ptr1 = get_ptr_from_batch(batch1).to(device)

    x2 = data.x_t
    edge_index2 = data.edge_index_t
    edge_attr2 = data.edge_attr_t
    subgraph_node_num2 = torch.stack(torch.split(
        data.subgraph_node_num_t, max_subgraph_num))
    real_graph_num2 = torch.stack(torch.split(data.real_graph_num_t, 1))
    source_ids2 = torch.stack(torch.split(
        data.source_ids_t, max_source_length))
    source_mask2 = torch.stack(torch.split(
        data.source_ids_t.ne(1), max_source_length))
    batch2 = data.x_t_batch
    ptr2 = get_ptr_from_batch(batch2).to(device)

    probs = model(x1, edge_index1, edge_attr1, subgraph_node_num1, real_graph_num1, batch1, ptr1, source_ids1, source_mask1,
                  x2, edge_index2, edge_attr2, subgraph_node_num2, real_graph_num2, batch2, ptr2, source_ids2, source_mask2)
    loss = loss_func(probs, data.label)
#     print('probs', probs)
#     print('labels', data.label)
    tr_loss += loss.item()
#     print('loss', loss.item())
    train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
    bar.set_description('loss {}'.format(train_loss))
    nb_tr_examples += data.label.size(0)
    nb_tr_steps += 1
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    global_step += 1

    if (global_step + 1) % valid_loss_steps == 0:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        valid_sampler = RandomSampler(
            valid_features, replacement=True, num_samples=10000)
        valid_dataloader = DataLoader(
            valid_features, sampler=valid_sampler, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
        msgr.print_msg("\n***** Running evaluation *****")
        msgr.print_msg("  Num examples = {}".format(len(valid_features)))
        msgr.print_msg("  Batch size = {}".format(batch_size))

        # Start Evaling model
        model.eval()
        valid_loss, tokens_num = 0, 10000
        logits = []
        y_trues = []
        for data in valid_dataloader:
            data = data.to(device)
            x1 = data.x_s
            edge_index1 = data.edge_index_s
            edge_attr1 = data.edge_attr_s
            subgraph_node_num1 = torch.stack(torch.split(
                data.subgraph_node_num_s, max_subgraph_num))
            real_graph_num1 = torch.stack(
                torch.split(data.real_graph_num_s, 1))
            source_ids1 = torch.stack(torch.split(
                data.source_ids_s, max_source_length))
            source_mask1 = torch.stack(torch.split(
                data.source_ids_s.ne(1), max_source_length))
            batch1 = data.x_s_batch
            ptr1 = get_ptr_from_batch(batch1).to(device)

            x2 = data.x_t
            edge_index2 = data.edge_index_t
            edge_attr2 = data.edge_attr_t
            subgraph_node_num2 = torch.stack(torch.split(
                data.subgraph_node_num_t, max_subgraph_num))
            real_graph_num2 = torch.stack(
                torch.split(data.real_graph_num_t, 1))
            source_ids2 = torch.stack(torch.split(
                data.source_ids_t, max_source_length))
            source_mask2 = torch.stack(torch.split(
                data.source_ids_t.ne(1), max_source_length))
            batch2 = data.x_t_batch
            ptr2 = get_ptr_from_batch(batch2).to(device)

            with torch.no_grad():
                probs = model(x1, edge_index1, edge_attr1, subgraph_node_num1, real_graph_num1, batch1, ptr1, source_ids1, source_mask1,
                              x2, edge_index2, edge_attr2, subgraph_node_num2, real_graph_num2, batch2, ptr2, source_ids2, source_mask2)
            loss = loss_func(probs, data.label)
            probs = F.softmax(probs)
            logits.append(probs.cpu().numpy())
            y_trues.append(data.label.cpu().numpy())
            valid_loss += loss.item()
        valid_loss /= tokens_num
        result = {'valid_loss': valid_loss,
                  'global_step': global_step+1,
                  'train_loss': round(train_loss, 5)}
        for key in sorted(result.keys()):
            msgr.print_msg("{}= {}".format(key, str(result[key])))
        msgr.print_msg("  "+"*"*20)

        # save last checkpoint
        last_output_dir = os.path.join(output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        if valid_loss < best_loss:
            msgr.print_msg("  Best valid_loss:{}".format(valid_loss))
            msgr.print_msg("  " + "*" * 20)
            best_loss = valid_loss
            # Save best checkpoint for best loss
            best_output_dir = os.path.join(output_dir, 'checkpoint-best-loss')
            if not os.path.exists(best_output_dir):
                os.makedirs(best_output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                best_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        logits = np.concatenate(logits, 0)
        y_trues = np.concatenate(y_trues, 0)
#         msgr.print_msg("logits: {}".format(logits[0:100]))
#         msgr.print_msg("y_trues: {}".format(y_trues[0:100]))
        best_threshold = 0
        best_f1 = 0
        for i in range(1, 100):
            threshold = i/100
            y_preds = logits[:, 1] > threshold
            from sklearn.metrics import recall_score
            recall = recall_score(y_trues, y_preds, average='macro')
            from sklearn.metrics import precision_score
            precision = precision_score(y_trues, y_preds, average='macro')
            from sklearn.metrics import f1_score
            f1 = f1_score(y_trues, y_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        y_preds = logits[:, 1] > best_threshold
        from sklearn.metrics import recall_score
        recall = recall_score(y_trues, y_preds, average='macro')
        from sklearn.metrics import precision_score
        precision = precision_score(y_trues, y_preds, average='macro')
        from sklearn.metrics import f1_score
        f1 = f1_score(y_trues, y_preds, average='macro')
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,

        }

        msgr.print_msg("***** Eval results *****")
        for key in sorted(result.keys()):
            msgr.print_msg("{}= {}".format(key, str(round(result[key], 4))))

        if f1 > best_f1:
            msgr.print_msg("  Best f1:{}".format(f1))
            msgr.print_msg("  "+"*"*20)
            best_f1 = f1
            # Save best checkpoint for best bleu
            best_output_dir = os.path.join(output_dir, 'checkpoint-best-f1')
            if not os.path.exists(best_output_dir):
                os.makedirs(best_output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                best_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        model.train()


def test(checkpoint_name):
    model_name = output_dir + checkpoint_name
    model.load_state_dict(torch.load(model_name))
    model.eval()

    logits = []
    y_trues = []
    for data in valid_dataloader:
        data = data.to(device)
        x1 = data.x_s
        edge_index1 = data.edge_index_s
        edge_attr1 = data.edge_attr_s
        subgraph_node_num1 = torch.stack(torch.split(
            data.subgraph_node_num_s, max_subgraph_num))
        real_graph_num1 = torch.stack(torch.split(data.real_graph_num_s, 1))
        source_ids1 = torch.stack(torch.split(
            data.source_ids_s, max_source_length))
        source_mask1 = torch.stack(torch.split(
            data.source_ids_s.ne(1), max_source_length))
        batch1 = data.x_s_batch
        ptr1 = get_ptr_from_batch(batch1).to(device)

        x2 = data.x_t
        edge_index2 = data.edge_index_t
        edge_attr2 = data.edge_attr_t
        subgraph_node_num2 = torch.stack(torch.split(
            data.subgraph_node_num_t, max_subgraph_num))
        real_graph_num2 = torch.stack(torch.split(data.real_graph_num_t, 1))
        source_ids2 = torch.stack(torch.split(
            data.source_ids_t, max_source_length))
        source_mask2 = torch.stack(torch.split(
            data.source_ids_t.ne(1), max_source_length))
        batch2 = data.x_t_batch
        ptr2 = get_ptr_from_batch(batch2).to(device)

        with torch.no_grad():
            probs = model(x1, edge_index1, edge_attr1, subgraph_node_num1, real_graph_num1, batch1, ptr1, source_ids1, source_mask1,
                          x2, edge_index2, edge_attr2, subgraph_node_num2, real_graph_num2, batch2, ptr2, source_ids2, source_mask2)
        loss = loss_func(probs, data.label)
        probs = F.softmax(probs)
        logits.append(probs.cpu().numpy())
        y_trues.append(data.label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    best_threshold = 0.5
    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    msgr.print_msg("***** Eval results *****")
    for key in sorted(result.keys()):
        msgr.print_msg("{}= {}".format(key, str(round(result[key], 4))))


test_dataloader = DataLoader(
    test_features, batch_size=batch_size, shuffle=True, follow_batch=['x_s', 'x_t'])
msgr.print_msg("\n***** Running testing *****")
msgr.print_msg("  Num examples = {}".format(len(test_features)))
msgr.print_msg("  Batch size = {}".format(batch_size))

checkpoint_list = ['/checkpoint-best-loss/pytorch_model.bin',
                   '/checkpoint-best-bleu/pytorch_model.bin', '/checkpoint-last/pytorch_model.bin']
for c in checkpoint_list:
    test(checkpoint_name=c)
