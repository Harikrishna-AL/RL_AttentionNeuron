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
    def __init__(
        self, act_dim, hidden_dim, msg_dim, pos_emb_dim, bias=True, scale=True
    ):
        super(AttentionNeuron, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_emb_dim = pos_emb_dim
        self.pos_embedding = torch.from_numpy(
            pos_table(self.act_dim, self.pos_emb_dim)
        ).float()
        self.hx = None
        self.lstm = nn.LSTMCell(input_size=1 + self.act_dim, hidden_size=pos_emb_dim)
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

        w = torch.tanh(self.attention(q=self.pos_embedding.to(x.device), k=self.hx[0]))
        output = torch.matmul(w, x)
        return torch.tanh(output)

    def reset(self):
        self.hx = None


class BasePiTorchModel(BasePiModel):
    def __init__(self, device):
        self.modules_to_learn = []
        self.device = torch.device(device)

    def get_action(self, obs):
        with torch.no_grad():
            return self._get_action(obs)

    def get_params(self):
        params = []
        with torch.no_grad():
            for layer in self.modules_to_learn:
                for p in layer.parameters():
                    params.append(p.cpu().numpy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        assert isinstance(params, np.ndarray)
        ss = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                ee = ss + np.prod(p.shape)
                p.data = (
                    torch.from_numpy(params[ss:ee].reshape(p.shape))
                    .float()
                    .to(self.device)
                )
                ss = ee
        assert ss == params.size

    def save(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def load(self, filename):
        with np.load(filename) as data:
            params = data["params"]
            self.set_params(params)

    def get_num_params(self):
        return self.get_params().size

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass


class RL_agent(BasePiTorchModel):
    def __init__(
        self,
        device,
        act_dim,
        hidden_dim,
        msg_dim,
        pos_em_dim,
        num_hidden_layers=1,
        pi_layer_bias=True,
        pi_layer_scale=True,
    ):
        super(RL_agent, self).__init__(device=device)
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.prev_act = torch.zeros(1, self.act_dim)

        self.att_neuron = AttentionNeuron(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_emb_dim=pos_em_dim,
            bias=pi_layer_bias,
            scale=pi_layer_scale,
        )
        self.modules_to_learn.append(self.pi_layer)

        hidden_lyrs = []
        for layer in range(num_hidden_layers):
            hidden_lyrs.extend(
                [
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.Tanh(),
                ]
            )
        self.net = nn.Sequential(
            *hidden_lyrs,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.net)

    def _get_action(self, obs):
        x = self.att_neuron(obs=obs, prev_act=self.prev_act)
        self.prev_act = self.net(x.T)
        return self.prev_act.squeeze(0).cpu().numpy()

    def reset(self):
        self.prev_act = torch.zeros(1, self.act_dim)
        self.att_neuron.reset()
