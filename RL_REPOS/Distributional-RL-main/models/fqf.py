import torch
from torch import nn
from torch.nn import functional as F
from .base_model import BaseModel
import math


class FQFModel(BaseModel):
    def __init__(self, state_shape, num_actions, num_embedding, num_quants):
        super(FQFModel, self).__init__(state_shape)
        self.num_embedding = num_embedding
        self.fp_layer = FractionProposalModel(self.flatten_size,
                                              num_quants
                                              )

        self.z = nn.Linear(512, num_actions)
        self.phi = nn.Linear(num_embedding, self.flatten_size)

        nn.init.orthogonal_(self.z.weight, 0.01)
        self.z.bias.data.zero_()
        nn.init.orthogonal_(self.phi.weight, gain=nn.init.calculate_gain("relu"))
        self.phi.bias.data.zero_()

    def forward(self, inputs):
        states, taus = inputs
        x = states / 255
        x = self.conv_net(x)
        state_feats = x.view(x.size(0), -1)

        #  view(...) for broadcasting later when it multiplies to taus
        i_pi = math.pi * torch.arange(1, 1 + self.num_embedding, device=taus.device).view(1, 1, self.num_embedding)
        taus = torch.unsqueeze(taus, -1)
        x = torch.cos(i_pi * taus).view(-1, self.num_embedding)
        phi = F.relu(self.phi(x))

        x = state_feats.view(state_feats.size(0), 1, -1) * phi.view(states.size(0), taus.size(1), -1)
        x = x.view(-1, phi.size(-1))
        x = F.relu(self.fc(x))
        z = self.z(x)

        return z.view(states.size(0), taus.size(1), -1)

    def get_qvalues(self, x):
        taus, tau_hats, *_ = self.get_taus(x)
        z = self.forward((x, tau_hats))
        q_values = torch.sum((taus[:, 1:, None] - taus[:, :-1, None]) * z, dim=1)
        return q_values

    def get_taus(self, x):
        with torch.no_grad():
            x = x / 255
            x = self.conv_net(x)
            state_feats = x.view(x.size(0), -1)
        return self.fp_layer(state_feats)


class FractionProposalModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FractionProposalModel, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        self.layer.bias.data.zero_()

    def forward(self, x):
        # raw logits -> stable log‐softmax + exp
        logits = self.layer(x)
        #use log softmax instead
        temp = 1.0
        log_probs = F.log_softmax(logits / temp, dim=-1)
        raw_probs = torch.exp(log_probs)

        # clamp to avoid exact zeros (and infinities in log)
        eps = 1e-6
        clamped_probs = raw_probs.clamp(min=eps, max=1.0)

        #recompute log-probs of the clamped distribution
        clamped_log_probs = torch.log(clamped_probs)

        # entropy: safe because log_probs is finite and probs > 0
        ent = -(clamped_probs * clamped_log_probs).sum(dim=-1)

        # build cumulative fractions
        taus = torch.cumsum(clamped_probs, dim=-1)
        batch, N1 = taus.shape
        taus = torch.cat([
            torch.zeros(batch, 1, device=taus.device),
            taus
        ], dim=-1)

        # mid‐points for quantile regression
        tau_hats  = 0.5 * (taus[:, :-1] + taus[:, 1:])

        return taus, tau_hats, ent, raw_probs, clamped_probs
