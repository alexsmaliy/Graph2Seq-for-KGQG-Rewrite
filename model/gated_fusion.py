import torch
import torch.nn as nn

class GatedFusionModule(nn.Module):
    def __init__(self, hidden_size: int):
        super(GatedFusionModule, self).__init__()
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        ts = [h_state, input, h_state * input, h_state - input]
        cat_ts = torch.cat(ts, -1)
        fx = self.fc_z(cat_ts)
        z = torch.sigmoid(fx)
        h_state = (1 - z) * h_state + z * input
        return h_state
