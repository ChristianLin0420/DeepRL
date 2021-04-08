
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import gym
import gym.spaces

import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout

# Training parameters
ENV_NAME = "SlimeVolleyNoFrameskip-v0"

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10 ** 4 * 4
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000
LEARNING_STARTS = 50000

EPSILON_DECAY = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

# OpenAI Gym Wrappers
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        print(env.unwrapped.get_action_meanings())
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

# DQN Architecture
class DeepQnetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DeepQnetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        print("conv_out_size: {}".format(input_shape))

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, * shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


# Experience replay
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


# Agent
class Agent:
    def __init__(self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self._reset()
        self.last_action = 0

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        Select action
        Execute action and step environment
        Add state/action/reward to experience replay
        """
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.replay_memory.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

# Loss function
def calculate_loss(batch, net, target_net, device="cpu"):
    """
    Calculate MSE between actual state action values,
    and expected state action values from DQN
    """
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]
    next_state_values[done] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


# Training loop
if __name__ == "__main__":

    device = torch.device("cpu")

    # Make Gym environement and DQNs
    env = gym.make(ENV_NAME)
    net = DeepQnetwork(env.observation_space.shape, env.action_space.n).to(device)
    # target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    
    # print(net)

    # replay_memory = ExperienceReplay(REPLAY_SIZE)
    # agent = Agent(env, replay_memory)
    # epsilon = EPSILON_START

    # optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # total_rewards = []
    # best_mean_reward = None
    # frame_idx = 0
    # timestep_frame = 0
    # timestep = time.time()

    # while True:
    #     frame_idx += 1
    #     epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY)

    #     reward = agent.play_step(net, epsilon, device=device)
    #     if reward is not None:
    #         total_rewards.append(reward)
    #         speed = (frame_idx - timestep_frame) / (time.time() - timestep)
    #         timestep_frame = frame_idx
    #         timestep = time.time()
    #         mean_reward = np.mean(total_rewards[-100:])
    #         print("{} frames: done {} games, mean reward {}, eps {}, speed {} f/s".format(
    #             frame_idx, len(total_rewards), round(mean_reward, 3), round(epsilon,2), round(speed, 2)))

    #         if best_mean_reward is None or best_mean_reward < mean_reward or len(total_rewards) % 25 == 0:
    #             torch.save(net.state_dict(), args.env + "-" + str(len(total_rewards)) + ".dat")
    #             if COLAB:
    #                 gsync.update_file_to_folder(args.env + "-" + str(len(total_rewards)) + ".dat")
    #             if best_mean_reward is not None:
    #                 print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3), round(mean_reward, 3)))
    #             best_mean_reward = mean_reward

    #         if mean_reward > args.reward and len(total_rewards) > 10:
    #             print("Game solved in {} frames! Average score of {}".format(frame_idx, mean_reward))
    #             break

    #     if len(replay_memory) < LEARNING_STARTS:
    #         continue

    #     if frame_idx % TARGET_UPDATE_FREQ == 0:
    #         target_net.load_state_dict(net.state_dict())

    #     optimizer.zero_grad()
    #     batch = replay_memory.sample(BATCH_SIZE)
    #     loss_t = calculate_loss(batch, net, target_net, device=device)
    #     loss_t.backward()
    #     optimizer.step()

    env.close()




