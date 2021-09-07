# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import torch
import torch.nn as nn
from codebert_seq2seq import Seq2Seq
from torchsummary import summary
import json
from tqdm import tqdm, trange
from itertools import cycle
from datetime import datetime
import logging
import os
import bleu
import numpy as np
import random


# %%
class Info(object):
    def __init__(self, info_prefix=''):
        self.info_prefix = info_prefix

    def print_msg(self, msg):
        text = self.info_prefix + ' ' + msg
        print(text)
        logging.info(text)


run_id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
log_file = 'logs/' + run_id + '.log'
exp_dir = 'runs/' + run_id
os.mkdir(exp_dir)
logging.basicConfig(format='%(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, level=logging.DEBUG)
msgr = Info('codebert enhanced')

msgr.print_msg('run_id : {}'.format(run_id))
msgr.print_msg('log_file : {}'.format(log_file))
msgr.print_msg('exp_dir: {}'.format(exp_dir))


# %%
checkpoint = 'microsoft/codebert-base'
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
roberta = RobertaModel.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
config = RobertaConfig.from_pretrained(checkpoint)


# %%
max_source_length = 256
max_target_length = 64
batch_size = 64
beam_size = 10
lr = 5e-5
warmup_steps = 0
train_steps = 10000
weight_decay = 0.0
adam_epsilon = 1e-8
valid_steps = 1000
train_url = '/data/code/represent-code-in-human/data/code-summarization-enhanced-middle/train.jsonl'
valid_url = '/data/code/represent-code-in-human/data/code-summarization-enhanced-middle/valid.jsonl'
test_url = '/data/code/represent-code-in-human/data/code-summarization-enhanced-middle/test.jsonl'
output_dir = 'codebert_checkpoint/'


# %%
decoder_layer = nn.TransformerDecoderLayer(
    d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model = Seq2Seq(encoder=roberta, decoder=decoder, config=config, beam_size=10, max_length=max_target_length,
                sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)


# %%
# summary(model)


# %%
device = torch.device('cuda: 0')
model.to(device)


# %%
class Example(object):
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


# %%
def read_examples(filename):
    examples = []
    with open(filename, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            ast_des = js['ast_des'].strip()
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=ast_des,
                    target=nl,
                )
            )
    return examples


# %%
class InputFeatures(object):
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


# %%
def convert_examples_to_features(examples, tokenizer, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[
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
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask
            )
        )
    return features

# %% [markdown]
# training


# %%
train_examples = read_examples(train_url)
train_features = convert_examples_to_features(
    train_examples, tokenizer, stage='train')
all_source_ids = torch.tensor(
    [f.source_ids for f in train_features], dtype=torch.long)
all_source_mask = torch.tensor(
    [f.source_mask for f in train_features], dtype=torch.long)
all_target_ids = torch.tensor(
    [f.target_ids for f in train_features], dtype=torch.long)
all_target_mask = torch.tensor(
    [f.target_mask for f in train_features], dtype=torch.long)
train_data = TensorDataset(
    all_source_ids, all_source_mask, all_target_ids, all_target_mask)


# %%
train_dataloader = DataLoader(train_data, batch_size=batch_size)


# %%
# optimizer and schedule
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=train_steps)


# %%
# Start training
msgr.print_msg("***** Running training *****")
msgr.print_msg("  Num examples = {}".format(len(train_examples)))
msgr.print_msg("  Batch size = {}".format(batch_size))
msgr.print_msg("  Num epoch = {}".format(batch_size//len(train_examples)))
model.train()
valid_dataset = {}
nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
bar = tqdm(range(train_steps), total=train_steps)
train_dataloader = cycle(train_dataloader)

for step in bar:
    batch = next(train_dataloader)
    batch = tuple(t.to(device) for t in batch)
    source_ids, source_mask, target_ids, target_mask = batch
    loss, _, _, = model(source_ids, source_mask, target_ids, target_mask)

    tr_loss += loss.item()
    train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
    bar.set_description('loss {}'.format(train_loss))
    nb_tr_examples += source_ids.size(0)
    nb_tr_steps += 1
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    global_step += 1

    if (global_step + 1) % valid_steps == 0:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        if 'valid_loss' in valid_dataset:
            valid_examples, valid_data = valid_dataset['valid_loss']
        else:
            valid_examples = read_examples(valid_url)
            valid_features = convert_examples_to_features(
                valid_examples, tokenizer, stage='valid')
            all_source_ids = torch.tensor(
                [f.source_ids for f in valid_features], dtype=torch.long)
            all_source_mask = torch.tensor(
                [f.source_mask for f in valid_features], dtype=torch.long)
            all_target_ids = torch.tensor(
                [f.target_ids for f in valid_features], dtype=torch.long)
            all_target_mask = torch.tensor(
                [f.target_mask for f in valid_features], dtype=torch.long)
            valid_data = TensorDataset(
                all_source_ids, all_source_mask, all_target_ids, all_target_mask)
            valid_dataset['valid_loss'] = valid_examples, valid_data
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size)

        msgr.print_msg("***** Running evaluation *****")
        msgr.print_msg("  Num examples = {}".format(len(valid_examples)))
        msgr.print_msg("  Batch size = {}".format(batch_size))

        # Start Evaling model
        model.eval()
        valid_loss, tokens_num = 0, 0
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            with torch.no_grad():
                _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                     target_ids=target_ids, target_mask=target_mask)
            valid_loss += loss.sum().item()
            tokens_num += num.sum().item()
        # Pring loss of valid dataset
        model.train()
        valid_loss = valid_loss / tokens_num
        result = {'valid_ppl': round(np.exp(valid_loss), 5),
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
            output_dir = os.path.join(output_dir, 'checkpoint-best-ppl')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # Calculate bleu
        if 'valid_bleu' in valid_dataset:
            valid_examples, valid_data = valid_dataset['valid_bleu']
        else:
            valid_examples = read_examples(valid_url)
            valid_examples = random.sample(
                valid_examples, min(1000, len(valid_examples)))
            valid_features = convert_examples_to_features(
                valid_examples, tokenizer, stage='test')
            all_source_ids = torch.tensor(
                [f.source_ids for f in valid_features], dtype=torch.long)
            all_source_mask = torch.tensor(
                [f.source_mask for f in valid_features], dtype=torch.long)
            valid_data = TensorDataset(all_source_ids, all_source_mask)
            valid_dataset['valid_bleu'] = valid_examples, valid_data

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size)

        model.eval()
        p = []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                preds = model(source_ids=source_ids, source_mask=source_mask)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(
                        t, clean_up_tokenization_spaces=False)
                    p.append(text)
        model.train()
        predictions = []
        with open(os.path.join(output_dir, "valid.output"), 'w') as f, open(os.path.join(output_dir, "valid.gold"), 'w') as f1:
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
            output_dir = os.path.join(output_dir, 'checkpoint-best-bleu')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)


# %%
test_examples = read_examples(test_url)
test_features = convert_examples_to_features(
    test_examples, tokenizer, stage='test')
all_source_ids = torch.tensor(
    [f.source_ids for f in test_features], dtype=torch.long)
all_source_mask = torch.tensor(
    [f.source_mask for f in test_features], dtype=torch.long)
test_data = TensorDataset(all_source_ids, all_source_mask)

# Calculate bleu
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(
    test_data, sampler=test_sampler, batch_size=batch_size)

model.eval()
p = []
for batch in tqdm(test_dataloader, total=len(test_dataloader)):
    batch = tuple(t.to(device) for t in batch)
    source_ids, source_mask = batch
    with torch.no_grad():
        preds = model(source_ids=source_ids, source_mask=source_mask)
        for pred in preds:
            t = pred[0].cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
            p.append(text)
model.train()
predictions = []
with open(os.path.join(output_dir, "test.output"), 'w') as f, open(os.path.join(output_dir, "test.gold"), 'w') as f1:
    for ref, gold in zip(p, test_examples):
        predictions.append(str(gold.idx)+'\t'+ref)
        f.write(str(gold.idx)+'\t'+ref+'\n')
        f1.write(str(gold.idx)+'\t'+gold.target+'\n')

(goldMap, predictionMap) = bleu.computeMaps(
    predictions, os.path.join(output_dir, "test.gold"))
dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
msgr.print_msg(" {} = {} ".format("bleu-4", str(dev_bleu)))
msgr.print_msg("  "+"*"*20)


# %%
