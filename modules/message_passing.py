import torch
import torch.nn as nn

class GraphMessagePassing(nn.Module):
    def __init__(self):
        super(GraphMessagePassing, self).__init__()

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        # (batch_size, num_edges, hidden_size)
        node2edge_emb = torch.bmm(node2edge, node_state)
        # self-loop
        nrm = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, node2edge_emb) + node_state) / nrm
        return agg_state
