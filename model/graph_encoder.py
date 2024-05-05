import torch
from torch import nn

import config
from utils import Logger

class GraphNN(nn.Module):
    def __init__(self, device: torch.device, logger: Logger):
        super(GraphNN, self).__init__()
        self.device = device
        self.logger = logger
        self.hidden_size = config.GRAPH_HIDDEN_SIZE
        self.graph_hops = config.GRAPH_HOPS

        # self.f_ans_pool = config['f_ans_pool'] # False
        # self.graph_direction = config.get('graph_direction', 'all') # "all"
        # self.graph_type = config['graph_type'] # "static"

        self.linear_max = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        #self.static_graph_mp = GraphMessagePassing(config) #TODO
        #self.static_gru_step = GRUStep(hidden_size, hidden_size) #TODO
        #self.static_gated_fusion = GatedFusion(hidden_size) #TODO
        #self.graph_pool = self.graph_maxpool #TODO

        self.logger.log(f"Using a bidirectional graph encoder with {self.graph_hops} hops.")

    def forward(self, node_state, edge_vec, adj, node_mask=None, ans_state=None):
        node_state, graph_embedding = self.graph_update(
            node_state,
            edge_vec,
            adj,
            node_mask=node_mask,
            ans_state=ans_state,
        )
        return node_state, graph_embedding

    def graph_update(self):
        pass
