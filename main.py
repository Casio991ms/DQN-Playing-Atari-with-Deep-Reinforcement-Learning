import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, env, epsilon_schedule=None):
        super().__init__()
        self.env = env
        self.epsilon_schedule = epsilon_schedule
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, env.single_action_space.n)
        )

    def forward(self, x):
        return self.network(x / 255.0)