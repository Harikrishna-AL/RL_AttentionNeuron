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


class SelfAttentionMatrix(AttentionMatrix):
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
        self, act_dim, hidden_dim, msg_dim, pos_emb_dim, bias=True, scale=True, RL=False
    ):
        super(AttentionNeuron, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_emb_dim = pos_emb_dim
        self.rl = RL
        self.pos_embedding = torch.from_numpy(
            pos_table(self.act_dim, self.pos_emb_dim)
        ).float()
        self.hx = None
        if self.rl == True:
            self.lstm = nn.LSTMCell(
                input_size=1 + self.act_dim, hidden_size=pos_emb_dim
            )
        else:
            self.lstm = nn.LSTMCell(input_size=16, hidden_size=pos_emb_dim)
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
        x = x.view(x.shape[0], x.shape[1])
        # print(x.shape)
        obs_dim = x.shape[0]

        if self.rl == True:
            x_aug = torch.cat([x, torch.vstack([prev_act] * obs_dim)], dim=-1)
        else:
            x_aug = x
        if self.hx is None:
            self.hx = (
                torch.zeros(obs_dim, self.pos_emb_dim),
                torch.zeros(obs_dim, self.pos_emb_dim),
            )

        new_hx = self.lstm(x_aug, self.hx)
        self.hx = (new_hx[0].detach(), new_hx[1].detach())

        w = torch.tanh(self.attention(q=self.pos_embedding.to(x.device), k=self.hx[0]))
        output = torch.matmul(w, x)
        return torch.tanh(output)

    def reset(self):
        self.hx = None


class AttentionLayer(nn.Module):
    """The attention mechanism."""

    def __init__(self, dim_in_q, dim_in_k, dim_in_v, msg_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.attention_matrix = AttentionMatrix(
            dim_q=dim_in_q,
            dim_k=dim_in_k,
            msg_dim=msg_dim,
        )
        self.proj_v = nn.Linear(in_features=dim_in_v, out_features=out_dim)
        self.mostly_attended_entries = None

    def forward(self, data_q, data_k, data_v):
        a = torch.softmax(self.attention_matrix(q=data_q, k=data_k), dim=-1)
        self.mostly_attended_entries = set(torch.argmax(a, dim=-1).numpy())
        v = self.proj_v(data_v)
        return torch.matmul(a, v)


class VisionAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        msg_dim,
        pos_emb_dim,
        patch_size,
        stack_k=1,
        stack_dim_first=False,
    ):
        super(VisionAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_emb_dim = pos_emb_dim
        self.patch_size = patch_size
        self.stack_k = stack_k
        self.stack_dim_first = stack_dim_first
        self.pos_embedding = torch.from_numpy(
            pos_table(self.pos_emb_dim, self.pos_emb_dim)
        ).float()
        self.attention = AttentionLayer(
            dim_in_q=self.pos_emb_dim,
            dim_in_k=self.stack_k * self.patch_size**2,
            dim_in_v=self.stack_k * self.patch_size**2,
            msg_dim=self.msg_dim,
            out_dim=self.msg_dim,
        )
        self.input_ln = nn.LayerNorm(
            normalized_shape=self.patch_size**2,
            elementwise_affine=True,
        )
        self.input_ln.eval()
        self.output_ln = nn.LayerNorm(
            normalized_shape=self.msg_dim,
            elementwise_affine=True,
        )
        self.output_ln.eval()

    def get_patches(self, x):
        h, w, c = x.size()
        # print(x.unfold(0, self.patch_size, self.patch_size))
        patches = x.unfold(0, self.patch_size, self.patch_size).permute(0, 3, 1, 2)
        patches = patches.unfold(2, self.patch_size, self.patch_size).permute(
            0, 2, 1, 4, 3
        )
        return patches.reshape((-1, self.patch_size, self.patch_size, c))

    def forward(self, x):
        # if self.stack_dim_first:
        #     x = x.permute(1, 0, 2, 3)
        # else:
        #     x = x.permute(0, 3, 1, 2)
        # print("input shape: ", x.shape)
        x = self.get_patches(x)
        # print("shape after patching: ",x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.input_ln(x)
        # print("shape after falttening: ",x.shape)
        x = torch.tanh(
            self.attention(
                data_q=self.pos_embedding.to(x.device),
                data_k=x,
                data_v=x,
            )
        )
        # print(x.shape)
        x = self.output_ln(x)
        # if self.stack_dim_first:
        #     x = x.permute(1, 0, 2)
        # else:
        #     x = x.permute(0, 2, 3, 1)
        return x


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


class RL_agent(nn.Module):
    def __init__(
        self,
        act_dim,
        hidden_dim,
        msg_dim,
        pos_em_dim,
        patch_size,
        num_classes,
        device="cpu",
        num_hidden_layers=1,
        pi_layer_bias=True,
        pi_layer_scale=True,
        rl=False,
    ):
        super(RL_agent, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.prev_act = torch.zeros(1, self.act_dim)
        self.device = device

        self.att_neuron = AttentionNeuron(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_emb_dim=pos_em_dim,
            bias=pi_layer_bias,
            scale=pi_layer_scale,
            RL=rl,
        )
        self.modules_to_learn = []
        self.modules_to_learn.append(self.att_neuron)

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
        self.out = nn.Linear(
            in_features=hidden_dim * (self.patch_size**2),
            out_features=self.num_classes,
        )
        self.modules_to_learn.append(self.net)
        self.modules_to_learn.append(self.out)

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

    def get_patches(self, x):
        h, w, c = x.size()
        patches = x.unfold(0, self.patch_size, self.patch_size).permute(0, 3, 1, 2)
        patches = patches.unfold(2, self.patch_size, self.patch_size).permute(
            0, 2, 1, 4, 3
        )
        return patches.reshape((-1, self.patch_size, self.patch_size, c))

    def _get_action(self, obs):
        obs = self.get_patches(obs.permute(1, 2, 0)).permute(0, 3, 1, 2)
        obs = torch.flatten(obs, start_dim=1)
        x = self.att_neuron(obs=obs, prev_act=self.prev_act)
        # print(x.shape)
        x = x.T
        x = self.net(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=0)
        output = self.out(x)
        return output

    def reset(self):
        self.prev_act = torch.zeros(1, self.act_dim)
        self.att_neuron.reset()

    # def parameters(self):
    #     return self.modules_to_learn[0].parameters()

    def forward(self, obs):
        return self._get_action(obs)

    def get_loss(self, action, target):
        return torch.nn.functional.cross_entropy(action, target)


class FaceAgent(nn.Module):
    
    def __init__(self,
                 device,
                 act_dim,
                 msg_dim,
                 pos_emb_dim,
                patch_size=6,
                stack_k=1,
                feat_dim=20,
    ):
        super(FaceAgent, self).__init__()
        self.device = device
        self.act_dim = act_dim
        self.msg_dim = msg_dim
        self.pos_emb_dim = pos_emb_dim
        self.patch_size = patch_size
        self.stack_k = stack_k
        self.feat_dim = feat_dim

        self.attn_neuron = VisionAttentionLayer(
            hidden_dim=feat_dim**2,
            msg_dim=msg_dim,
            pos_emb_dim=pos_emb_dim,
            patch_size=patch_size,
            stack_k=stack_k,
            stack_dim_first=False,
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(4,4),
                stride=(2,2),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1),
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=18496, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=act_dim),
        )

    def forward(self, obs):

        x = self.attn_neuron(x = obs) 
        x = x.reshape(40,40,1).unsqueeze(0)
        x = torch.relu(x.permute(0, 3, 1, 2))
        x = self.cnn(x)
        # x = torch.softmax(x, dim=-1)
        return x
        