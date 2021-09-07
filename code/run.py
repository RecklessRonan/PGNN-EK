from data_preprocess import get_asts_and_labels, create_ggnn_cs_data, get_code_token_vocab
from gnn import GGNN, RNNModule, GGNNRNN, GCN, GCNRNN
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import ignite

train_asts, valid_asts, test_asts, train_labels, valid_labels, test_labels, text_vocab_len, text_vocab_dict = get_asts_and_labels()
code_vocab_len, code_vocab_dict = get_code_token_vocab(
    train_asts, valid_asts, test_asts)
train_data, valid_data, test_data = create_ggnn_cs_data(
    train_asts, valid_asts, test_asts, train_labels, valid_labels, test_labels, code_vocab_dict)

device = torch.device('cuda:0')
# ggnn = GGNN(vocablen=code_vocab_len, embedding_dim=50,
#             num_layers=4, device=device)
# ggnn.to(device)
# rnn = RNNModule(text_vocab_len, seq_size=32, embedding_size=50, lstm_size=64)
# rnn.to(device)

batch_size = 24
seq_size = 32
epoches = 10


# model = GCN(vocab_len=code_vocab_len, embedding_dim=50,
#             hidden_channels=64, output_dim=64, batch_size=batch_size)
# model = GGNNRNN(vocablen=code_vocab_len, embedding_dim=50, num_layers=4,
#                 device=device, n_vocab=text_vocab_len, seq_size=seq_size, lstm_size=64, batch_size=batch_size)
model = GCNRNN(vocab_len=code_vocab_len, embedding_dim=50, hidden_channels=64, output_dim=64, device=device,
               n_vocab=text_vocab_len, seq_size=seq_size, lstm_size=64, batch_size=batch_size, num_layers=2)
print(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
metric = ignite.metrics.nlp.Bleu(ngram=4, smooth="smooth1")
train_loader = DataLoader(train_data, batch_size=batch_size)
valid_loader = DataLoader(valid_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

for e in range(epoches):
    print('epoch ', e)
    model.train()
    train_loss = 0
    for data in train_loader:
        # print('data', data)
        data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # print('output', output.shape)

        label_split = torch.stack(list(torch.split(data.y, 32, dim=0)), dim=0)
        # print('label', label_split.shape)
        # print(label_split)
        output = output.view(-1, output.shape[-1])
        loss = criterion(output, data.y)
        # print('loss', loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('train loss', train_loss)

    model.eval()
    valid_loss = 0
    for data in valid_loader:
        data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        output = output.view(-1, output.shape[-1])
        loss = criterion(output, data.y)
        # print('loss', loss.item())
        valid_loss += loss.item()

    print('valid loss', valid_loss)


print('start testing...')
for data in test_loader:
    data.to(device)
    output = model(data.x, data.edge_index, data.edge_attr, data.batch)
    predict = torch.argmax(output, dim=-1)
