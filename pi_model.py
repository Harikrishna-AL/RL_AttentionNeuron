from base_pi_model import BasePiModel

import torch
import numpy as np
import torch.nn as nn


def pos_table(n, dim):
    def get_angle(x, h):
        return x / np.power(10000, 2 * (h // 2) / dim)
    
    def get_angle_vec(x):
        return [get_angle(x, h) for h in range(dim)]
    
    tab = np.array([get_angle_vec(i) for i in range(n)])
    tab[:, 0::2] = np.sin(tab[:, 0::2])
    tab[:, 1::2] = np.cos(tab[:, 1::2])
    return tab


class AttentionMatrix(nn.Module):
    def __init__(self, dim_q, dim_k, msg_dim, bias=True, scale=True):
        super(AttentionMatrix, self).__init__()
        self.proj_q = nn.Linear(dim_q, msg_dim, bias=bias)
        self.proj_k = nn.Linear(dim_k, msg_dim, bias=bias)
        if scale:
            self.msg_dim = msg_dim
        else:
            self.msg_dim = 1
        
    def forward(self, q, k):
        q = self.proj_q(q)
        k = self.proj_k(k)
        if q.ndim == k.ndim == 2:
            dot = torch.matmul(q, k.T)
        else:
            dot = torch.bmm(q, k.permute(0, 2, 1))
        return torch.div(dot, np.sqrt(self.msg_dim))
        

class SelfAttentionMatrix(nn.Module):
    def __init__(self, dim_in, msg_dim, bias=True, scale=True):
        super(SelfAttentionMatrix, self).__init__(
            dim_q=dim_in,
            dim_k=dim_in,
            msg_dim=msg_dim,
            bias=bias,
            scale=scale,
        )


class AttentionNeuron(nn.Module):
    def __init__(self,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_emb_dim,
                 bias=True,
                 scale=True):
        super(AttentionNeuron, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_emb_dim = pos_emb_dim
        self.pos_embedding = torch.from_numpy(
            pos_table(self.act_dim, self.pos_emb_dim)
            ).float()
        self.hx = None
        self.lstm = nn.LSTMCell(
            input_size=1 + self.act_dim, hidden_size=pos_emb_dim
        )
        self.attention = SelfAttentionMatrix(
            dim_in=pos_emb_dim,
            msg_dim=msg_dim,
            bias=bias,
            scale=scale,
        )

    def forward(self, obs, prev_act):
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs.copy()).float().unsqueeze(-1)
        else:
            x = obs.unsqueeze(-1)
        obs_dim = x.shape[0]

        x_aug = torch.cat([x, torch.vstack([prev_act] * obs_dim)], dim=-1)
        if self.hx is None:
            self.hx = (
                torch.zeros(obs_dim, self.pos_emb_dim),
                torch.zeros(obs_dim, self.pos_emb_dim),
            )
        self.hx = self.lstm(x_aug, self.hx)

        w = torch.tanh(self.attention(
            q=self.pos_embedding.to(x.device), k=self.hx[0]
        ))
        output = torch.matmul(w, x)
        return torch.tanh(output)
    
    def reset(self):
        self.hx = None
        
            
        