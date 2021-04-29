
from __future__ import division

import os
import math
import cv2
import argparse
import bz2
import pickle
from datetime import datetime
import numpy as np

import atari_py
from tqdm import trange
from collections import deque
import random

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

#### training package
import time

import gym
from gym import spaces
from pynput.keyboard import Controller
from pynput.keyboard import Key
from selenium import webdriver

# commend
# python 107062240_hw3_train.py --target-update 2000 --T-max 100000 --learn-start 1600 --memory-capacity 100000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 10000

'''
    QWOP
'''

PORT = 8000
PRESS_DURATION = 0.1
MAX_EPISODE_DURATION_SECS = 120
STATE_SPACE_N = 71
ACTIONS = {
    0: 'qw',
    1: 'qo',
    2: 'qp',
    3: 'q',
    4: 'wo',
    5: 'wp',
    6: 'w',
    7: 'op',
    8: 'o',
    9: 'p',
    10: '',
}

MODEL_NAME = 'FastDQN_2hr'
EXPLORATION_FRACTION = 0.3
LEARNING_STARTS = 3000
EXPLORATION_INITIAL_EPS = 0.01
EXPLORATION_FINAL_EPS = 0.001
BUFFER_SIZE = 300000
BATCH_SIZE = 64
TRAIN_FREQ = 4
LEARNING_RATE = 0.0001
TRAIN_TIME_STEPS = 600000
MODEL_PATH = os.path.join('models', MODEL_NAME)
TENSORBOARD_PATH = './tensorboard/'

def get_new_model():

    # Initialize env and model
    env = QWOPEnv()
    model = DQN(
        CustomDQNPolicy,
        env,
        prioritized_replay=True,
        verbose=1,
        tensorboard_log=TENSORBOARD_PATH,
    )

    return model

def run_train(model_path=MODEL_PATH):

    model = get_new_model()
    model.learning_rate = LEARNING_RATE
    model.learning_starts = LEARNING_STARTS
    model.exploration_initial_eps = EXPLORATION_INITIAL_EPS
    model.exploration_final_eps = EXPLORATION_FINAL_EPS
    model.buffer_size = BUFFER_SIZE
    model.batch_size = BATCH_SIZE
    model.train_freq = TRAIN_FREQ

    # Train and save
    t = time.time()

    model.learn(
        total_timesteps=TRAIN_TIME_STEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )
    model.save(MODEL_PATH)

    print(f"Trained {TRAIN_TIME_STEPS} steps in {time.time()-t} seconds.")

class QWOPEnv(gym.Env):

    meta_data = {'render.modes': ['human']}
    pressed_keys = set()

    def __init__(self):

        # Open AI gym specifications
        super(QWOPEnv, self).__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[STATE_SPACE_N], dtype=np.float32
        )
        self.num_envs = 1

        # QWOP specific stuff
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.evoke_actions = True

        # Open browser and go to QWOP page
        self.driver = webdriver.Chrome("/Users/christianlin/Downloads/chromedriver")
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')

        # Wait a bit and then start game
        time.sleep(2)
        self.driver.find_element_by_xpath("//body").click()

        self.keyboard = Controller()
        self.last_press_time = time.time()

    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name};')

    def _get_state_(self):

        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')

        # Get done
        if (
            (game_state['gameEnded'] > 0)
            or (game_state['gameOver'] > 0)
            or (game_state['scoreTime'] > MAX_EPISODE_DURATION_SECS)
        ):
            self.gameover = done = True
        else:
            self.gameover = done = False

        # Get reward
        torso_x = body_state['torso']['position_x']
        torso_y = body_state['torso']['position_y']

        # Reward for moving forward
        reward1 = max(torso_x - self.previous_torso_x, 0)

        # # Penalize for low torso
        # if torso_y > 0:
        #     reward2 = -torso_y / 5
        # else:
        #     reward2 = 0

        # # Penalize for torso vertical velocity
        # reward3 = -abs(torso_y - self.previous_torso_y) / 4

        # # Penalize for bending knees too much
        # if (
        #     body_state['joints']['leftKnee'] < -0.9
        #     or body_state['joints']['rightKnee'] < -0.9
        # ):
        #     reward4 = (
        #         min(body_state['joints']['leftKnee'], body_state['joints']['rightKnee'])
        #         / 6
        #     )
        # else:
        #     reward4 = 0

        # Combine rewards
        reward = reward1 * 2  # + reward2 + reward3 + reward4

        # print(
        #     'Rewards: {:3.1f}, {:3.1f}, {:3.1f}, {:3.1f}, {:3.1f}'.format(
        #         reward1, reward2, reward3, reward4, reward
        #     )
        # )

        # Update previous scores
        self.previous_torso_x = torso_x
        self.previous_torso_y = torso_y
        self.previous_score = game_state['score']
        self.previous_time = game_state['scoreTime']

        # Normalize torso_x
        for part, values in body_state.items():
            if 'position_x' in values:
                values['position_x'] -= torso_x

        # print('Positions: {:3.1f}, {:3.1f}, {:3.1f}'.format(
        #     body_state['torso']['position_x'],
        #     body_state['leftThigh']['position_x'],
        #     body_state['rightCalf']['position_x']
        # ))

        # print('Knee angles: {:3.2f}, {:3.2f}'.format(
        #     body_state['joints']['leftKnee'],
        #     body_state['joints']['rightKnee']
        # ))

        # Convert body state
        state = []
        for part in body_state.values():
            state = state + list(part.values())
        state = np.array(state)

        return state, reward, done, {}

    def _release_all_keys_(self):

        for char in self.pressed_keys:
            self.keyboard.release(char)

        self.pressed_keys.clear()

    def send_keys(self, keys):

        # Release all keys
        self._release_all_keys_()

        # Hold down current key
        for char in keys:
            self.keyboard.press(char)
            self.pressed_keys.add(char)

        # print('pressed for', time.time() - self.last_press_time)
        # self.last_press_time = time.time()
        time.sleep(PRESS_DURATION)

    def reset(self):

        # Send 'R' key press to restart game
        self.send_keys(['r', Key.space])
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self._release_all_keys_()

        return self._get_state_()[0]

    def step(self, action_id):

        # send action
        keys = ACTIONS[action_id]

        if self.evoke_actions:
            self.send_keys(keys)
        else:
            time.sleep(PRESS_DURATION)

        return self._get_state_()

    def render(self, mode='human'):
        pass

    def close(self):
        pass


'''
    MODEL
'''

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

'''
    AGENT
'''
class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()

    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
        # Calculate nth next state probabilities
        pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
        dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
        argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
        self.target_net.reset_noise()  # Sample new target net noise
        pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
        pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

        # Compute Tz (Bellman operator T applied to z)
        Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = states.new_zeros(self.batch_size, self.atoms)
        offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

'''
    ENVIRONMENT
'''
class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
        
        # Perform up to 30 random no-ops before starting
        for _ in range(random.randrange(30)):
            self.ale.act(0)  # Assumes raw action 0 is always no-op
            if self.ale.game_over():
                self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False

        for t in range(4):
            reward += self.ale.act(self.actions.get(action))

            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()

            if done:
                break

        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()

            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True

            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()



'''
    REPLAY MEMORY
'''
Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
        self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # True if future frame has timestep 0
        
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
        transitions[blank_mask] = blank_trans
        
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
            if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)
        # Discrete actions to be used as index
        actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
        
        transitions[blank_mask] = blank_trans
        state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        
        return state

#   next = __next__  # Alias __next__ for Python 2 compatibility

'''
    TEST
'''
def test(args, T, dqn, val_mem, metrics, results_dir, evaluate=False):
    env = Env(args)
    env.eval()
    metrics['steps'].append(T)
    T_rewards, T_Qs = [], []

    # Test performance over several episodes
    done = True
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False

        action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
        state, reward, done = env.step(action)  # Step
        reward_sum += reward
        
        if args.render:
            env.render()

        if done:
            T_rewards.append(reward_sum)
            break
    env.close()

    # Test Q-values over validation memory
    for state in val_mem:  # Iterate over valid states
        T_Qs.append(dqn.evaluate_q(state))

    avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
    if not evaluate:
        # Save model parameters if improved
        if avg_reward > metrics['best_avg_reward']:
            metrics['best_avg_reward'] = avg_reward
            dqn.save(results_dir)

        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        metrics['Qs'].append(T_Qs)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot
        _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
        _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

    # Return average reward and Q-value
    return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


'''
    MAIN
'''
# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')

for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

results_dir = os.path.join('results', args.id)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))

# if torch.cuda.is_available() and not args.disable_cuda:
#     args.device = torch.device('cuda')
#     torch.cuda.manual_seed(np.random.randint(1, 10000))
#     torch.backends.cudnn.enabled = args.enable_cudnn
# else:
#     args.device = torch.device('cpu')

args.device = torch.device('cpu')

# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


# Environment
env =  QWOPEnv()##Env(args)
env.train()
action_space = env.action_space()

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

    mem = load_memory(args.memory, args.disable_bzip_memory)

else:
    mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True

while T < args.evaluation_size:
    if done:
        state = env.reset()

    next_state, _, done = env.step(np.random.randint(0, action_space))
    val_mem.append(state, -1, 0.0, done)
    state = next_state
    T += 1

if args.evaluate:
    dqn.eval()  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    # Training loop
    dqn.train()
    T, done = 0, True

    print("Start trainning")

    for T in trange(1, args.T_max + 1):
        if done:
            state = env.reset()

        if T % args.replay_frequency == 0:
            dqn.reset_noise()  # Draw a new set of noisy weights

        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done = env.step(action)  # Step
        
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        
        mem.append(state, action, reward, done)  # Append transition to memory

        # Train and test
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                dqn.learn(mem)  # Train with n-step distributional double-Q learning

            if T % args.evaluation_interval == 0:
                dqn.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                dqn.train()  # Set DQN (online network) back to training mode

                # If memory path provided, save it
                if args.memory is not None:
                    save_memory(mem, args.memory, args.disable_bzip_memory)

        # Update target network
        if T % args.target_update == 0:
            dqn.update_target_net()

        # Checkpoint the network
        if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
            dqn.save(results_dir, 'checkpoint.pth')

        state = next_state

env.close()