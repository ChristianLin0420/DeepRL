
import numpy as np
import random
import itertools
from collections import deque

import gym
import slimevolleygym

import torch
from torch import nn

# Training parameters
# ENV_NAME = "SlimeVolleyNoFrameskip-v0"
ENV_NAME = "CartPole-v0"

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 100000

class Network(nn.Module):
    
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype = torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim = 1)[0]
        action = max_q_index.detach().item()


env = gym.make(ENV_NAME)

replay_buffer = deque(maxlen = BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen = 100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr = 5e-4)

# Initialize Replay Buffer
obs = env.reset()

for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, reward, done, info = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

# Main Training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()

    if random_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, reward, done, _ = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += reward

    if done:
        obs = env.reset()

        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    observations = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_observations = np.asarray([t[4] for t in transitions])

    observations_t = torch.as_tensor(observations, dtype = torch.float32)
    actions_t = torch.as_tensor(actions, dtype = torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype = torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype = torch.float32).unsqueeze(-1)
    new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32)

    # Compute Targets
    target_q_values = target_net(new_observations_t)
    max_target_q_values = target_q_values.max(dim = 1, keepdim = True)[0]

    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(observations_t)

    action_q_values = torch.gather(input = q_values, dim = 1, index = actions_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Reward', np.mean(reward_buffer))
