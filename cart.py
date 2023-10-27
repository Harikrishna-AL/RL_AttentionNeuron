import gymnasium as gym
from tqdm import tqdm
import numpy as np
from gym.utils.play import play
from pi_model import RL_agent
import cma
import torch
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()


act_dim = 10
hidden_dim = 10
msg_dim = 10
pos_em_dim = 10
num_hidden_layers = 1
pi_layer_bias = True
pi_layer_scale = True
device = "cpu"
rl = False

agent = RL_agent(
    device=device,
    act_dim=act_dim,
    hidden_dim=hidden_dim,
    msg_dim=msg_dim,
    pos_em_dim=pos_em_dim,
    patch_size=9,
    num_hidden_layers=num_hidden_layers,
    pi_layer_bias=pi_layer_bias,
    pi_layer_scale=pi_layer_scale,
    rl=rl,
)


for _ in tqdm(range(500)):
    global observation

    if _ % 100 == 0:
        observation = env.reset()
        observation = observation[0]
    action = agent.get_action(observation)

    if action > 0.5:
        action = 1
    else:
        action = 0
    print(action)
    obs, reward, done, truncated, info = env.step(action)
    observation = obs

env.close()
sample_image = torch.randn(3,100,100)
for _ in tqdm(range(50)):
    obs = agent.get_action(sample_image)
    print(obs.shape)
