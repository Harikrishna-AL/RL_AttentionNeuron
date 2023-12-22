import gymnasium as gym
from tqdm import tqdm
import numpy as np
from gym.utils.play import play
from pi_model import RL_agent, FaceAgent
import cma
import torch
import matplotlib.pyplot as plt
from utils import load_dataset, Dataset, train_sgd, test_model, train_cma, fitness

# env = gym.make("CartPole-v1", render_mode="human")
# obs, info = env.reset()


act_dim = 40
hidden_dim = 40
msg_dim = 40
pos_em_dim = 40
num_hidden_layers = 1
num_classes = 40
pi_layer_bias = True
pi_layer_scale = True
device = "cuda:0"
rl = False

agent = RL_agent(
    act_dim=act_dim,
    hidden_dim=hidden_dim,
    msg_dim=msg_dim,
    pos_em_dim=pos_em_dim,
    patch_size=4,
    device=device,
    num_classes=num_classes,
    num_hidden_layers=num_hidden_layers,
    pi_layer_bias=pi_layer_bias,
    pi_layer_scale=pi_layer_scale,
    rl=rl,
).to(device)


# for _ in tqdm(range(500)):
#     global observation

#     if _ % 100 == 0:
#         observation = env.reset()
#         observation = observation[0]
#     action = agent.get_action(observation)

#     if action > 0.5:
#         action = 1
#     else:
#         action = 0
#     print(action)
#     obs, reward, done, truncated, info = env.step(action)
#     observation = obs

# env.close()

# sample_image = torch.randn(3, 100, 100)
# for _ in tqdm(range(50)):
#     obs = agent.get_action(sample_image)
#     print(obs.shape)

import torch.autograd

torch.autograd.set_detect_anomaly(True)

face_agent = FaceAgent(
    device=device,
    act_dim=act_dim,
    msg_dim=msg_dim,
    pos_emb_dim=pos_em_dim,
    patch_size=8,

)
# faces, target = load_dataset("../../Documents/face_dataset/")
# dataset = Dataset(faces, target)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# for i, (x, y) in enumerate(dataloader):
#     # print(x.shape)
#     x = x.reshape(64, 64,1)
#     y_pred = face_agent(x)

def main(device):
    faces, target = load_dataset("../../Documents/face_dataset/")

    dataset = Dataset(faces, target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = torch.nn.MSELoss()
    config = {"num_epochs": 30, "num_iters": 10, "lr": 0.005}

    train_sgd(config, criterion, face_agent, dataloader, device)


# main(device)

# faces, target = load_dataset("../../Documents/face_dataset/")
# dataset = Dataset(faces, target)

# test_model(agent, face=dataset[0][0].reshape(1, 64, 64), target=dataset[0][1])
# face_agent = FaceAgent(agent, device)
main(device)

def main_cma():
    config = {
        "num_iters": 100,
        "pop_size": 10,
        "sigma_init": 0.1,
        "seed": 0,
    }
    train_cma(config, agent, fitness)


