
import os
import math
import random
from time import time
import numpy as np
import pandas as pd
from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

import argparse
import itertools

from osim.env import L2M2019Env

'''
    Custom operations
'''
def flatten_list(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item

def flatten(mydict):
    new_dict = {}
    for key, value in mydict.items():
        if type(value) == list:
            print("key: {}, value: {}".format(key, value))
            new_dict[key] = flatten_list(value)
        else:
            new_dict[key] = value
    return new_dict

def FF(ss):
    state = flatten(ss)
    # print(state)

    new_state = []

    for v in state.values():       
        # print(type(v))
        if type(v) == dict:
            temp = pd.json_normalize(v, sep = '_')
            temp = list(temp.values)
            for item in temp:
                if type(item) == np.ndarray:
                    for val in item:
                        if type(val) == list:
                            for t in val:
                                new_state.append(float(t))
                        else:
                            new_state.append(float(val))
                else:
                    new_state.append(item)
        else:
            # print("asdf")
            for arr in v:
                for a in arr:
                    for val in a:
                        new_state.append(float(val))

    return new_state

'''
    Model
'''

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_skills, hidden_dim):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_skills)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x # score, unnormalized

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, context, action):
        xu = torch.cat([state, context, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, context):
        xu = torch.cat([state, context], 1)
        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, context):
        mean, log_std = self.forward(state, context)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

'''
    Replay Memory
'''
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, context, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (context, state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        print("start sampling.....")
        batch = random.sample(self.buffer, batch_size)
        context, state, action, reward, next_state, done = map(np.stack, zip(*batch))
        print("finish sampling.....")
        return context, state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

'''
    Utility
'''
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


'''
    SAC
'''
class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cpu") 

        self.critic = QNetwork(num_inputs + args.num_skills, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs + args.num_skills, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.disc = Discriminator(num_inputs, args.num_skills, args.hidden_size).to(device=self.device)
        self.disc_optim = Adam(self.disc.parameters(), lr=args.lr)

        self.disc_target = Discriminator(num_inputs, args.num_skills, args.hidden_size).to(device=self.device)
        hard_update(self.disc_target, self.disc)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs + args.num_skills, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs + args.num_skills, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, context, eval=False):
        state = FF(state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        context = torch.FloatTensor(context).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state, context)
        else:
            _, _, action = self.policy.sample(state, context)
        return action.detach().cpu().numpy()[0]

    def state_prob(self, context, state):
        state = FF(state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        context = torch.FloatTensor(context).to(self.device).unsqueeze(0)
        score = self.disc_target(state)
        prob = F.softmax(score, dim=1) 
        prob = prob * context
        prob, _ = torch.max(prob, dim=1, keepdim=False)
        return prob.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        context_batch, state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        import time

        # print("start updating parameters......")
        now = time.time()
        temp = []
        for a in state_batch:
            b = FF(a)
            temp.append(b)


        temp1 = []
        for a in next_state_batch:
            b = FF(a)
            temp1.append(b)

        context_batch = torch.FloatTensor(context_batch).to(self.device)
        state_batch = torch.FloatTensor(temp).to(self.device)
        next_state_batch = torch.FloatTensor(temp1).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, context_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, context_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, context_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch, context_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, context_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        score_vector = self.disc(state_batch)
        context_index = torch.argmax(context_batch, dim=1)
        disc_loss = F.cross_entropy(score_vector, context_index)

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        self.critic_optim.zero_grad()
        self.policy_optim.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        policy_loss.backward()
        self.critic_optim.step()        
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.disc_target, self.disc, self.tau)

        end = time.time()

        # print("finish updating parameters......")
        # print("update duration: {}".format(end - now))

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), disc_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="hw4", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

'''
    Main
'''

torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="2d-navigation-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--num_skills', type=int, default=20,
                    help='Number of skills to learn')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.05, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=8192, metavar='N',
                    help='batch size (default: 8192)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--buffer_size', type=int, default=100000, metavar='N',
                    help='Replay buffer for discriminator')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = L2M2019Env(visualize=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model("models/sac_actor_107062240_", "models/sac_critic_107062240_")

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    episode_sr = 0 # pseudo reward
    done = False
    state = env.reset()

    context_index = np.random.randint(0, high=args.num_skills)
    context = np.zeros(args.num_skills)
    context[context_index] = 1.
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state, context)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, disc_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        state_prob = agent.state_prob(context, state)
        pseudo_reward = np.log(max(state_prob, 1E-6)) + np.log(args.num_skills)
        episode_sr += pseudo_reward

        mask = 1 if episode_steps == 100 else float(not done)

        memory.push(context, state, action, pseudo_reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, sr: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), round(episode_sr, 2)))

    # if i_episode > 10:
    #     disc_loss, disc_loss_old = agent.update_disc(buffer, args.batch_size * 10, steps=50)    
    # else:        
    #     disc_loss = 0.
    #     disc_loss_old = 0.
    # writer.add_scalar('loss/disc', disc_loss, i_episode)
    # writer.add_scalar('loss/disc_delta', disc_loss - disc_loss_old, i_episode)

    if i_episode % 5 == 0 and args.eval == True:
        avg_reward = 0.
        avg_sr = 0.
        episodes = args.num_skills
        for i  in range(episodes):

            state = env.reset()
            traj = []
            traj.append([state, None, 0.0, False])
            episode_reward = 0
            episode_sr = 0
            done = False
            context = np.zeros(args.num_skills)
            context[i] = 1.
            while not done:
                action = agent.select_action(state, context, eval=False)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                traj.append([next_state, action, reward, done])

                state_prob = agent.state_prob(context, next_state)
                pseudo_reward = np.log(max(state_prob, 1E-6)) + np.log(args.num_skills)
                episode_sr += pseudo_reward

                state = next_state
            avg_reward += episode_reward
            avg_sr += episode_sr

        avg_reward /= episodes
        avg_sr /= episodes

        agent.save_model("107062240")

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Avg. SR: {}".format(episodes, round(avg_reward, 2), round(avg_sr, 2)))
        print("----------------------------------------")

env.close()