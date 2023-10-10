import gymnasium as gym
from tqdm import tqdm
import numpy as np
from gym.utils.play import play
from pi_model import RL_agent

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()


act_dim = 1
hidden_dim = 1
msg_dim = 1
pos_em_dim = 5
num_hidden_layers = 1
pi_layer_bias = True
pi_layer_scale = True
device = "cpu"

agent = RL_agent(
    device=device,
    act_dim=act_dim,
    hidden_dim=hidden_dim,
    msg_dim=msg_dim,
    pos_em_dim=pos_em_dim,
    num_hidden_layers=num_hidden_layers,
    pi_layer_bias=pi_layer_bias,
    pi_layer_scale=pi_layer_scale,
)

for _ in tqdm(range(500)):
    global observation

    if _%100 == 0:
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
