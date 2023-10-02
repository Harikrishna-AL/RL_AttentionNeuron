import gymnasium as gym
from tqdm import tqdm
import numpy as np
from gym.utils.play import play


env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in tqdm(range(500)):
    action = env.action_space.sample()
    # print(type(action))
    obs, reward, done, truncated, info = env.step(action)
    # print(obs, action)
    # print(info)
    if done or truncated:
        obs, info = env.reset()

env.close()
# env.render()

# gym.utils.play.play(env, zoom=3)
