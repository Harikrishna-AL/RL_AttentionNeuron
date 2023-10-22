import numpy as np
import torch
import matplotlib.pyplot as plt
import cma


def fitness(params, agent, env):
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


def train_cma(config, agent, fitness):
    num_params = agent.get_num_params()
    init_params = agent.get_params()
    cma = cma.CMAEvolutionStrategy(
        x0=np.zeros(num_params) if init_params is None else init_params,
        sigma0=config.sigma_init,
        inopts={
            "popsize": config.pop_size,
            "seed": config.seed,
            "randn": np.random.randn,
        },
    )
    for _ in range(config.num_iters):
        solutions = cma.ask()
        fitnesses = fitness(zip(solutions, 100))
        cma.tell(solutions, -fitnesses)


def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def train_sgd(config, criterion, model, dataloader):
    # classifcatio model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.num_epochs):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")
