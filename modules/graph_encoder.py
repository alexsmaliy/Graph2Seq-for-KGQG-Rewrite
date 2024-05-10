import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .gated_fusion import GatedFusionModule
from .gated_recurrent_unit import GRUModule
from .message_passing import GraphMessagePassing
from utils import Logger, send_to_device

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
        self.static_graph_mp = GraphMessagePassing()
        self.static_gru_step = GRUModule(self.hidden_size, self.hidden_size)
        self.static_gated_fusion = GatedFusionModule(self.hidden_size)

        self.logger.log(f"Using a bidirectional graph encoder with {self.graph_hops} hops.")

    def forward(self, node_state, edge_vec, adj):
        node_state, graph_embedding = self.graph_update(node_state, edge_vec, adj)
        return node_state, graph_embedding

    def graph_update(self, node_state, edge_vec, adj):
        node2edge, edge2node = adj
        # (batch_size, num_edges, num_entities)
        node2edge = send_to_device(
            torch.stack(
                [torch.Tensor(x.A) for x in node2edge],
                dim=0,
            ),
            self.device,
        )
        # (batch_size, num_entities, num_edges)
        edge2node = send_to_device(
            torch.stack(
                [torch.Tensor(x.A) for x in edge2node],
                dim=0,
            ),
            self.device,
        )

        for _ in range(self.graph_hops):
            bw_agg_state = \
                self.static_graph_mp.msg_pass(node_state, edge_vec, node2edge, edge2node)

            # this direction is transposed to reverse the edges
            fw_agg_state = self.static_graph_mp.msg_pass(
                node_state,
                edge_vec,
                edge2node.transpose(1, 2),
                node2edge.transpose(1, 2),
            )

            agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
            node_state = self.static_gru_step(node_state, agg_state)

        graph_embedding = self.graph_max_pool(node_state).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def graph_max_pool(self, node_state):
        # (batch_size, hidden_size, num_entities)
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(
            node_embedding_p,
            kernel_size=node_embedding_p.size(-1),
        ).squeeze(-1)
        return graph_embedding
