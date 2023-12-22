import numpy as np
import torch
import matplotlib.pyplot as plt
import cma
from torch.utils.data import Dataset, DataLoader


def fitness(params, num_episodes, agent, obs, target):
    params, num_episodes = params, num_episodes
    agent.set_params(params)
    loss = []
    for _ in range(num_episodes):
        action = agent.get_action(obs)
        loss_item = agent.get_loss(action, target)
        loss.append(loss_item)
    return np.mean(loss)


def train_cma(config, agent, fitness, test_data):
    num_params = agent.get_num_params()
    init_params = agent.get_params()
    cmaes = cma.CMAEvolutionStrategy(
        x0=np.zeros(num_params) if init_params is None else init_params,
        sigma0=config["sigma_init"],
        inopts={
            "popsize": config["pop_size"],
            "seed": config["seed"],
            "randn": np.random.randn,
        },
    )
    for _ in range(config["num_iters"]):
        solutions = cmaes.ask()
        fitnesses = fitness(
            solutions, 100, agent, test_data[_][0].reshape(1, 64, 64), test_data[_][1]
        )
        cmaes.tell(solutions, -fitnesses)
        cmaes.logger.add()
        cmaes.disp()
    cmaes.result_pretty()


def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def train_sgd(config, criterion, model, dataloader, device):
    # classifcatio model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    for epoch in range(config["num_epochs"]):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.reshape(64, 64, 1)
            # y = torch.softmax(y, dim=1)
            # print(y)
            y_pred = model(x)
            y_pred = y_pred.reshape(1, 40)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        # print(y_pred.argmax(), y.argmax())
        # print(y_pred, y)
        print(f"Epoch {epoch}, Loss {loss.item()}")


def load_dataset(train_path):
    faces = np.load(train_path + "olivetti_faces.npy")
    target = np.load(train_path + "olivetti_faces_target.npy")
    return faces, target


def test_model(model, face, target):
    # face = torch.tensor(face, dtype=torch.float32)
    # target = torch.tensor(target, dtype=torch.float32)
    model.eval()
    y_pred = model(face)
    y_pred = y_pred.reshape(1, 40)
    print(y_pred.argmax(), target.argmax())


class Dataset(Dataset):
    def __init__(self, faces, target):
        self.faces = faces
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        target_arr = np.zeros(40)
        target_arr[self.target[idx]] = 1
        faces = torch.tensor(self.faces[idx], dtype=torch.float32) / 255
        target_arr = torch.tensor(target_arr, dtype=torch.float32)
        return faces, target_arr
