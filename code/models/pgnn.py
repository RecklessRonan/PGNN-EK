import torch
from torch.nn import (Module, Embedding, LSTM, Sequential, Linear, Sigmoid)
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn.glob import GlobalAttention


class GNNEncoder(Module):
    def __init__(self, vocab_len, graph_embedding_size, gnn_layers_num, lstm_layers_num, lstm_hidden_size, decoder_input_size, device):
        super(GNNEncoder, self).__init__()
        self.device = device
        self.embeddings = Embedding(vocab_len, graph_embedding_size)
        # only two edge types to be set weights, which are AST edge and data flow edge
        self.edge_embed = Embedding(4, 1)
        self.ggnnlayer = GatedGraphConv(graph_embedding_size, gnn_layers_num)
        self.mlp_gate = Sequential(
            Linear(graph_embedding_size, 300), Sigmoid(), Linear(300, 1), Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)
        self.lstm = LSTM(input_size=graph_embedding_size,
                         hidden_size=lstm_hidden_size, num_layers=lstm_layers_num)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers_num = lstm_layers_num
        self.fc = Linear(graph_embedding_size +
                         lstm_hidden_size, decoder_input_size)

    def subgraph_forward(self, x, edge_index, edge_attr, batch):
        if type(edge_attr) == type(None):
            edge_weight = None
        else:
            edge_weight = self.edge_embed(edge_attr)
            edge_weight = edge_weight.squeeze(1)
        x = self.ggnnlayer(x, edge_index, edge_weight)
        return self.pool(x, batch=batch)

    # partitioning multiple subgraphs by dynamic allocating edges
    def partition_graph(self, x, edge_index, edge_attr, subgraph_node_num, real_graph_num, ptr):
        nodes_list = []  # record all nodes number for each subgraph in total batch
        subgraph_num = max(real_graph_num)

        batch_size = subgraph_node_num.size(0)
        start_node_num = [1 for _ in range(batch_size)]
        for i in range(subgraph_num):
            subgraph_nodes_list = []
            for j in range(batch_size):
                if subgraph_node_num[j][i] != 0:
                    for k in range(ptr[j]+start_node_num[j], ptr[j]+start_node_num[j]+subgraph_node_num[j][i]):
                        subgraph_nodes_list.append(k)
                    start_node_num[j] += subgraph_node_num[j][i]
            nodes_list.append(subgraph_nodes_list)

        # only count the edge whose target node in subgraph
        sub_edge_src = [[] for _ in range(subgraph_num)]
        sub_edge_tgt = [[] for _ in range(subgraph_num)]
        sub_edge_attr = [[] for _ in range(subgraph_num)]
        # print('nodes_list', nodes_list)
        node_num = len(x)
        # use a list to store the subgraph numbers for all nodes
        node_subgraph_index = [0 for _ in range(node_num)]
        for i in range(len(nodes_list)):
            for node in nodes_list[i]:
                node_subgraph_index[node] = i

        for i in range(len(edge_index[1])):
            src = edge_index[0][i].item()
            tgt = edge_index[1][i].item()
            sub_edge_src[node_subgraph_index[tgt]].append(src)
            sub_edge_tgt[node_subgraph_index[tgt]].append(tgt)
            sub_edge_attr[node_subgraph_index[tgt]].append(edge_attr[i].item())
        edge_index_list = []
        edge_attr_list = []
        for i in range(subgraph_num):
            edge_index_list.append(torch.tensor(
                [sub_edge_src[i], sub_edge_tgt[i]], dtype=torch.long))
            edge_attr_list.append(torch.tensor(
                sub_edge_attr[i], dtype=torch.long))
        # print('nodes_list', nodes_list)
        return edge_index_list, edge_attr_list

    def forward(self, x, edge_index, edge_attr, subgraph_node_num, real_graph_num, batch, ptr):
        edge_index_list, edge_attr_list = self.partition_graph(
            x, edge_index, edge_attr, subgraph_node_num, real_graph_num, ptr)
        # print('edge_index_list', edge_index_list)
        # print('edge_attr_list', edge_attr_list)
        x = self.embeddings(x)
        x = x.squeeze(1)
        subgraph_pool_list = [
            self.subgraph_forward(x, edge_index_list[i].to(
                self.device), edge_attr_list[i].to(self.device), batch)
            for i in range(len(edge_index_list))
        ]
        graph_pool = self.subgraph_forward(x, edge_index, edge_attr, batch)
        # print('graph_pool', graph_pool.shape)
        subgraph_pool_seq = torch.stack(subgraph_pool_list)
        # print('subgraph_pool_seq', subgraph_pool_seq.shape)
        h0 = torch.zeros(self.lstm_layers_num, subgraph_pool_seq.size(
            1), self.lstm_hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm_layers_num, subgraph_pool_seq.size(
            1), self.lstm_hidden_size).to(self.device)
        subgraph_output, (_, _) = self.lstm(subgraph_pool_seq, (h0, c0))
        return self.fc(torch.cat((subgraph_output[-1], graph_pool), dim=1))
