import torch
import torch.nn as nn

from utils import INF

class AttentionModule(nn.Module):
    # attn_type='add'
    def __init__(self, hidden_size, h_state_embed_size, in_memory_embed_size):
        super(AttentionModule, self).__init__()
        _W = torch.Tensor(h_state_embed_size, hidden_size)
        self.W = nn.Parameter(nn.init.xavier_uniform_(_W))

        _W = torch.Tensor(in_memory_embed_size, hidden_size)
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(_W))

        _W = torch.Tensor(hidden_size, 1)
        self.W3 = nn.Parameter(nn.init.xavier_uniform_(_W))

    def forward(self, query_embed, in_memory_embed, attn_mask=None, addition_vec=None):
        att = in_memory_embed.view(
            -1,
            in_memory_embed.size(-1),
        )

        att = torch.mm(att, self.W2)

        att = att.view(
            in_memory_embed.size(0),
            -1,
            self.W2.size(-1),
        )

        att = att + torch.mm(query_embed, self.W).unsqueeze(1)

        if addition_vec is not None:
            att = att + addition_vec

        att = torch.tanh(att)
        att = att.view(-1, att.size(-1))
        att = torch.mm(att, self.W3)
        att = att.view(att.size(0), -1)

        if attn_mask is not None:
            att = attn_mask * att - (1 - attn_mask) * INF

        return att
