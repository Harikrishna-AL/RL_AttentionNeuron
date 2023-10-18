import gymnasium as gym
from tqdm import tqdm
import numpy as np
from gym.utils.play import play
from pi_model import RL_agent
import cma

# env = gym.make("CartPole-v1", render_mode="human")
# obs, info = env.reset()


act_dim = 10
hidden_dim =10
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
    num_hidden_layers=num_hidden_layers,
    pi_layer_bias=pi_layer_bias,
    pi_layer_scale=pi_layer_scale,
    rl=rl,
)


def fitness(params):
    params, num_episodes = params
    agent.set_params(params)
    total_reward = 0
    for _ in range(num_episodes):
        observation = env.reset()
        observation = observation[0]
        done = False
        while not done:
            action = agent.get_action(observation)
            if action > 0.5:
                action = 1
            else:
                action = 0
            obs, reward, done, truncated, info = env.step(action)
            observation = obs
            total_reward += reward
    return total_reward / num_episodes


def train_cma(config):
    num_params = agent.get_num_params()
    init_params = agent.get_params()
    cma = cma.CMAEvolutionStrategy(x0=np.zeros(num_params) if init_params is None else init_params,
                                   sigma0=config.sigma_init,
                                   inopts={'popsize': config.pop_size,
                                           'seed' : config.seed,
                                           'randn': np.random.randn},
                                    )
    for _ in range(config.num_iters):
        solutions = cma.ask()
        fitnesses = fitness(zip(solutions, 100))
        cma.tell(solutions, -fitnesses)


def train_sgd(config, criterion, model):
    pass



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

for _ in tqdm(range(50)):
    obs = agent.get_action(np.array([0, 0, 0, 0]))
    print(obs, obs.shape)
