# static GNN
# add RNN to GGNN to do text generation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn import MessagePassing, GatedGraphConv, GCNConv, global_mean_pool


class GGNN(nn.Module):
    def __init__(self, vocab_len, embedding_dim, num_layers, device):
        super(GGNN, self).__init__()
        self.device = device
        self.embed = nn.Embedding(vocab_len, embedding_dim)
        self.edge_embed = nn.Embedding(20, embedding_dim)
        self.ggnnlayer = GatedGraphConv(embedding_dim, num_layers)
        self.mlp_gate = nn.Sequential(
            nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

    def forward(self, data):
        x, edge_index, edge_attr = data
        # print('x', x)
        # print('edge_index', edge_index)
        # print('edge_attr', edge_attr)
        x = self.embed(x)
        x = x.squeeze(1)
        if type(edge_attr) == type(None):
            edge_weight = None
        else:
            edge_weight = self.edge_embed(edge_attr)
            edge_weight = edge_weight.squeeze(1)
        x = self.ggnnlayer(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        hg = self.pool(x, batch=batch)
        return hg


class GCN(nn.Module):
    def __init__(self, vocab_len, embedding_dim, hidden_channels, output_dim, batch_size):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.embed = nn.Embedding(vocab_len, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dim)
        self.batch_size = batch_size

    def forward(self, x, edge_index, edge_attr, batch):
        # obtain node embedding
        x = self.embed(x)
        x = x.squeeze(1)
        # print('after embed x', x.shape)
        x = self.conv1(x, edge_index)
        x = x.relu()
        # print('after conv1 x', x.shape)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # print('after conv3 x', x.shape)

        # readout layer
        # batch = torch.tensor(self.batch_size, dtype=torch.long)
        x = global_mean_pool(x, batch=batch)

        # output layer
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


# use RNN to generate code summarization
class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        logits = self.dense(output)
        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


# combine ggnn and rnn
class GGNNRNN(nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device, n_vocab, seq_size, lstm_size, batch_size):
        super(GGNNRNN, self).__init__()
        self.device = device
        self.embed = nn.Embedding(vocablen, embedding_dim)
        self.edge_embed = nn.Embedding(20, embedding_dim)
        self.ggnnlayer = GatedGraphConv(embedding_dim, num_layers)
        self.mlp_gate = nn.Sequential(
            nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_dim, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
        self.batch_size = batch_size

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data
        print('before gnn x', x.shape)
        print('before gnn edge_index', edge_index.shape)
        print('before gnn edge_attr', edge_attr.shape)
        x = self.embed(x)
        x = x.squeeze(1)
        if type(edge_attr) == type(None):
            edge_weight = None
        else:
            edge_weight = self.edge_embed(edge_attr)
            edge_weight = edge_weight.squeeze(1)
        x = self.ggnnlayer(x, edge_index)
        print('after gnn x', x.shape)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        print('batch', batch)
        hg = self.pool(x, batch=batch)
        print('hg', hg.shape)
        # hg = hg.repeat(self.seq_size, 1)
        hg = torch.unsqueeze(hg, 0)
        state_h = torch.zeros(1, self.batch_size,
                              self.lstm_size).to(self.device)
        state_c = torch.zeros(1, self.batch_size,
                              self.lstm_size).to(self.device)
        output, _ = self.lstm(hg, (state_h, state_c))
        logits = self.dense(output)

        return logits


# combine gcn and rnn
class GCNRNN(nn.Module):
    def __init__(self, vocab_len, embedding_dim, hidden_channels, output_dim, device, n_vocab, seq_size, lstm_size, batch_size, num_layers):
        super(GCNRNN, self).__init__()
        self.device = device
        torch.manual_seed(12345)
        self.embed = nn.Embedding(vocab_len, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dim)
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(output_dim, lstm_size,
                            num_layers, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
        self.batch_size = batch_size
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr, batch):
        # obtain node embedding
        x = self.embed(x)
        x = x.squeeze(1)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # print('x', x.shape)
        hg = torch.unsqueeze(x, 1)
        # print('hg before', hg.shape)
        # expand the output vector of gnn 'seq_size' times
        hg = hg.expand(-1, self.seq_size, -1)
        # print('hg after', hg.shape)
        state_h = torch.zeros(self.num_layers, hg.shape[0],
                              self.lstm_size).to(self.device)
        state_c = torch.zeros(self.num_layers, hg.shape[0],
                              self.lstm_size).to(self.device)
        output, _ = self.lstm(hg, (state_h, state_c))
        logits = self.dense(output)
        return logits
