from itertools import cycle
import yaml
import warnings
from datetime import datetime
import logging
import os
import torch
from torch.nn import (TransformerDecoderLayer, TransformerDecoder)
from transformers import (RobertaTokenizer, RobertaConfig, RobertaModel,
                          DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup)
from torch_geometric.data import DataLoader
from pgnn import GNNEncoder
from torch.utils.data import SequentialSampler
import numpy as np
import json
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import argparse

from codebert_seq2seq import Seq2Seq
import bleu
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Generate features for model')
parser.add_argument('--dataset', type=str, help='TLC or CSN')
args = parser.parse_args()
if args.dataset == 'TLC':
    dataset_url = '../../data/TLC/'
    features_url = '../features/TLC/'
    config_file = '../configs/config_cs_tlc.yml'
elif args.dataset == 'CSN':
    dataset_url = '../../data/CSN/'
    features_url = '../features/CSN/'
    config_file = '../configs/config_cs_csn.yml'
else:
    print('Wrong dataset name')

# Configuration
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
info_prefix = config['logs']['info_prefix']
# Data source
TRAIN_DIR = dataset_url + 'train_enhanced.jsonl'
VALID_DIR = dataset_url + 'valid_enhanced.jsonl'
TEST_DIR = dataset_url + 'test_enhanced.jsonl'
# Preprocess
max_source_length = config['preprocess']['max_source_length']
max_target_length = config['preprocess']['max_target_length']
divide_node_num = config['preprocess']['divide_node_num']
max_node_num = config['preprocess']['max_node_num']
max_subgraph_num = int(max_node_num/divide_node_num)
# Model design
graph_embedding_size = config['model']['graph_embedding_size']
lstm_hidden_size = config['model']['lstm_hidden_size']
gnn_layers_num = config['model']['gnn_layers_num']
lstm_layers_num = config['model']['lstm_layers_num']
decoder_input_size = config['model']['decoder_input_size']
# Training setting
batch_size = config['training']['batch_size']
beam_size = config['training']['beam_size']
lr = config['training']['lr']
gnn_lr = config['training']['gnn_lr']
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


train_examples = read_examples(TRAIN_DIR)
valid_examples = read_examples(VALID_DIR)
test_examples = read_examples(TEST_DIR)
msgr.print_msg('train size: {}, valid size: {}, test size: {}'.format(
    len(train_examples), len(valid_examples), len(test_examples)))

# Load features
train_features = torch.load(
    features_url + 'train_features.pt')
valid_features = torch.load(
    features_url + 'valid_features.pt')
test_features = torch.load(
    features_url + 'test_features.pt')


# Load CodeBERT related
# checkpoint = 'microsoft/codebert-base'
checkpoint = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhurenyu/huggingface-models/codebert'
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

device = torch.device('cuda:1')
gnn_encoder = GNNEncoder(vocab_len=tokenizer.vocab_size+num_added_toks, graph_embedding_size=graph_embedding_size,
                         gnn_layers_num=gnn_layers_num, lstm_layers_num=lstm_layers_num, lstm_hidden_size=lstm_hidden_size,
                         decoder_input_size=decoder_input_size, device=device)
decoder_layer = TransformerDecoderLayer(
    d_model=roberta_config.hidden_size, nhead=roberta_config.num_attention_heads)
decoder = TransformerDecoder(decoder_layer, num_layers=6)
model = Seq2Seq(encoder=roberta, decoder=decoder, gnn_encoder=gnn_encoder, config=roberta_config, beam_size=10, max_length=max_target_length,
                sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
model.to(device)
# print(model)


# optimizer and schedule
no_decay = ['bias', 'LayerNorm.weight']
all_param_optimizer = list(model.named_parameters())
gnn_param_optimizer = list(model.gnn_encoder.named_parameters())
other_param_optimizer = []

for op in all_param_optimizer:
    # print(op[0])
    if 'gnn_encoder' not in op[0]:
        other_param_optimizer.append(op)

optimizer_grouped_parameters = [
    {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay, 'lr': lr},
    {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': lr},
    
    {'params': [p for n, p in gnn_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay, 'lr': gnn_lr},
    {'params': [p for n, p in gnn_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': gnn_lr}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000,
#                                             num_training_steps=30000)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=train_steps)

# Start training
msgr.print_msg("***** Running training *****")
msgr.print_msg("  Num examples = {}".format(len(train_features)))
msgr.print_msg("  Batch size = {}".format(batch_size))
msgr.print_msg("  lr= {}".format(lr))
msgr.print_msg("  Num epoch = {}".format(batch_size//len(train_features)))
model.train()
valid_dataset = {}
nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
# bar = tqdm(range(train_steps), total=train_steps)
train_dataloader = DataLoader(train_features, batch_size=batch_size)
train_dataloader = cycle(train_dataloader)
output_dir = exp_dir

# for step in bar:
for step in range(train_steps):
    data = next(train_dataloader)
    data = data.to(device)
    # print(torch.split(
    #     data.subgraph_node_num, max_subgraph_num))
    subgraph_node_num = torch.stack(torch.split(
        data.subgraph_node_num, max_subgraph_num))
    real_graph_num = torch.stack(torch.split(data.real_graph_num, 1))
    source_ids = torch.stack(torch.split(data.source_ids, max_source_length))
    source_mask = torch.stack(torch.split(data.source_mask, max_source_length))
    target_ids = torch.stack(torch.split(data.target_ids, max_target_length))
    target_mask = torch.stack(torch.split(data.target_mask, max_target_length))
    loss, _, _, = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, subgraph_node_num=subgraph_node_num,
                        real_graph_num=real_graph_num, batch=data.batch, ptr=data.ptr, source_ids=source_ids, source_mask=source_mask,
                        target_ids=target_ids, target_mask=target_mask)

    tr_loss += loss.item()
    train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
    # bar.set_description('loss {}'.format(train_loss))
    msgr.print_msg("train loss= {}".format(train_loss))

    nb_tr_examples += data.x.size(0)
    nb_tr_steps += 1
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    global_step += 1

    if (global_step + 1) % valid_loss_steps == 0:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        valid_sampler = SequentialSampler(valid_features)
        valid_dataloader = DataLoader(
            valid_features, sampler=valid_sampler, batch_size=batch_size)

        msgr.print_msg("\n***** Running evaluation *****")
        msgr.print_msg("  Num examples = {}".format(len(valid_features)))
        msgr.print_msg("  Batch size = {}".format(batch_size))

        # Start Evaling model
        model.eval()
        valid_loss, tokens_num = 0, 0
        for data in valid_dataloader:
            data = data.to(device)
            subgraph_node_num = torch.stack(torch.split(
                data.subgraph_node_num, max_subgraph_num))
            real_graph_num = torch.stack(torch.split(data.real_graph_num, 1))
            source_ids = torch.stack(torch.split(
                data.source_ids, max_source_length))
            source_mask = torch.stack(torch.split(
                data.source_mask, max_source_length))
            target_ids = torch.stack(torch.split(
                data.target_ids, max_target_length))
            target_mask = torch.stack(torch.split(
                data.target_mask, max_target_length))

            with torch.no_grad():
                _, loss, num = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                     subgraph_node_num=subgraph_node_num, real_graph_num=real_graph_num,  ptr=data.ptr,
                                     source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
            valid_loss += loss.sum().item()
            tokens_num += num.sum().item()
        # Pring loss of valid dataset
        # model.train()
        valid_loss /= tokens_num
        result = {'valid_loss': valid_loss,
                  'valid_ppl': round(np.exp(valid_loss), 5),
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
            msgr.print_msg("  Best ppl:{}".format(
                round(np.exp(valid_loss), 5)))
            msgr.print_msg("  " + "*" * 20)
            best_loss = valid_loss
            # Save best checkpoint for best ppl
            best_output_dir = os.path.join(output_dir, 'checkpoint-best-ppl')
            if not os.path.exists(best_output_dir):
                os.makedirs(best_output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                best_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

    if (global_step + 1) % valid_bleu_steps == 0 and (global_step + 1) >= train_steps * 0.6:
        model.eval()
        p = []
        for data in valid_dataloader:
            data = data.to(device)
            subgraph_node_num = torch.stack(torch.split(
                data.subgraph_node_num, max_subgraph_num))
            real_graph_num = torch.stack(torch.split(data.real_graph_num, 1))
            source_ids = torch.stack(torch.split(
                data.source_ids, max_source_length))
            source_mask = torch.stack(torch.split(
                data.source_mask, max_source_length))
            target_ids = torch.stack(torch.split(
                data.target_ids, max_target_length))
            target_mask = torch.stack(torch.split(
                data.target_mask, max_target_length))
            with torch.no_grad():
                preds = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                              subgraph_node_num=subgraph_node_num, real_graph_num=real_graph_num, ptr=data.ptr,
                              source_ids=source_ids, source_mask=source_mask)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(
                        t, clean_up_tokenization_spaces=False)
                    p.append(text)
        predictions = []
        with open(os.path.join(output_dir, "valid.output"), 'w', encoding='utf-8') as f, open(os.path.join(output_dir, "valid.gold"), 'w', encoding='utf-8') as f1:
            for ref, gold in zip(p, valid_examples):
                predictions.append(str(gold.idx)+'\t'+ref)
                f.write(str(gold.idx)+'\t'+ref+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')

        (goldMap, predictionMap) = bleu.computeMaps(
            predictions, os.path.join(output_dir, "valid.gold"))
        valid_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        msgr.print_msg("  {} = {}".format("bleu-4", str(valid_bleu)))
        msgr.print_msg("  "+"*"*20)
        if valid_bleu > best_bleu:
            msgr.print_msg("  Best bleu:{}".format(valid_bleu))
            msgr.print_msg("  "+"*"*20)
            best_bleu = valid_bleu
            # Save best checkpoint for best bleu
            best_output_dir = os.path.join(output_dir, 'checkpoint-best-bleu')
            if not os.path.exists(best_output_dir):
                os.makedirs(best_output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                best_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
    model.train()


# nltk used for TLC
# we follow the original code in TLC 'evaluation.py'
def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


def nltk_corpus_bleu(hypotheses, references, order=4):
    refs = []
    count = 0.0
    total_score = 0.0
    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1
    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hypotheses)
    return corpus_bleu, avg_score


# Test
def test(checkpoint_name):
    checkpoint_dir = os.path.join(output_dir, checkpoint_name) 
    model_name = checkpoint_dir + '/pytorch_model.bin'

    model.load_state_dict(torch.load(model_name))
    model.eval()
    p = []
    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        data = data.to(device)
        subgraph_node_num = torch.stack(torch.split(
            data.subgraph_node_num, max_subgraph_num))
        real_graph_num = torch.stack(torch.split(data.real_graph_num, 1))
        source_ids = torch.stack(torch.split(
            data.source_ids, max_source_length))
        source_mask = torch.stack(torch.split(
            data.source_mask, max_source_length))
        with torch.no_grad():
            preds = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, subgraph_node_num=subgraph_node_num,
                          real_graph_num=real_graph_num, batch=data.batch, ptr=data.ptr, source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    model.train()
    predictions = []
    with open(os.path.join(checkpoint_dir, "test.output"), 'w', encoding='utf-8') as f, open(os.path.join(checkpoint_dir, "test.gold"), 'w', encoding='utf-8') as f1:
        for ref, gold in zip(p, test_examples):
            predictions.append(str(gold.idx)+'\t'+ref)
            f.write(str(gold.idx)+'\t'+ref+'\n')
            f1.write(str(gold.idx)+'\t'+gold.target+'\n')

    (goldMap, predictionMap) = bleu.computeMaps(
        predictions, os.path.join(checkpoint_dir, "test.gold"))

    if args.dataset == 'TLC':
        new_golds = []
        golds = []
        with open(os.path.join(checkpoint_dir, "test.gold"), 'r', encoding='utf-8') as f:
            for v in f.readlines():
                golds.append(v)
        for g in golds:
            t = tokenizer.tokenize(g.split('\t', 1)[1])[: 30]
            ids = tokenizer.convert_tokens_to_ids(t)
            tt = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            if not tt.endswith('\n'):
                tt += '\n'
            new_golds.append(str(g.split('\t', 1)[0])+'\t'+tt)
        nltk_golds = []
        nltk_preds = []
        for g, p in zip(new_golds, predictions):
            nltk_golds.append(g.split('\t', 1)[1])
            nltk_preds.append(p.split('\t', 1)[1])
        dev_bleu = nltk_corpus_bleu(nltk_golds, nltk_preds)[1] * 100
    else:
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    msgr.print_msg(" {} = {} ".format("bleu-4", str(dev_bleu)))
    msgr.print_msg("  "+"*"*20)


# Calculate bleu
test_sampler = SequentialSampler(test_features)
test_dataloader = DataLoader(
    test_features, sampler=test_sampler, batch_size=batch_size)

msgr.print_msg("\n***** Running testing *****")
msgr.print_msg("  Num examples = {}".format(len(test_features)))
msgr.print_msg("  Batch size = {}".format(batch_size))

checkpoint_list = ['checkpoint-best-ppl',
                   'checkpoint-best-bleu', 'checkpoint-last']
for c in checkpoint_list:
    test(checkpoint_name=c)
