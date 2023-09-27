import gymnasium as gym
from tqdm import tqdm

env = gym.make('CartPole-v1',render_mode="human")
obs, info = env.reset()

for _ in tqdm(range(1000)):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(info)
    if done:
        obs, info = env.reset()

env.close()
# env.render()