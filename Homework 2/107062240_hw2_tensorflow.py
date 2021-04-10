
import gym
import slimevolleygym

import torch
from torch.autograd import Variable

import time
import math
import copy
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
# import torchvision.transform as T



# Demonstration
# env = gym.envs.make("CartPole-v1") 
env = gym.envs.make("SlimeVolleyNoFrameskip-v0")

# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
episodes = 200
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001

print("n_state: {}".format(n_state))
print("n_action: {}".format(n_action))

def get_screen():
    ''' Extract one step of the simulation.'''
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    return torch.from_numpy(screen)

# Speify the number of simulation steps
# num_steps = 2

# # Show several steps
# for i in range(num_steps):
#     clear_output(wait=True)
#     env.reset()
#     plt.figure()
#     plt.imshow(get_screen().cpu().permute(1, 2, 0).numpy(),
#                interpolation='none')
#     plt.title('CartPole-v0 Environment')
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()

def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

# def random_search(env, episodes, 
#                   title='Random Strategy'):
#     """ Random search strategy implementation."""
#     final = []

#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         total = 0
#         while not done:
#             # Sample random actions
#             action = env.action_space.sample()
#             # Take action and extract results
#             next_state, reward, done, _ = env.step(action)
#             # Update reward
#             total += reward
#             if done:
#                 break
#         # Add to the final reward
#         final.append(total)
        
#         if episode == episodes - 1:
#             plot_res(final,title)
    
#     return final

# # Get random search results
# random_s = random_search(env, episodes)

class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim*2, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)



    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        state = observation(state)
        state = state.reshape((84, 84))
        y_pred = self.model(torch.Tensor(state))
        # print(y_pred)
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            state = observation(state)
            state = state.reshape((84, 84))
            # print("predict state shape: {}".format(state.shape))
            return self.model(torch.Tensor(state))

# Expand DQL class with a replay function.
class DQN_replay(DQN):
    
    #new replay function
    def replay(self, memory, size, gamma=0.9):
        """New replay function"""
        #Try to improve replay speed
        if len(memory)>=size:
            batch = random.sample(memory,size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            states = observation(states)
            next_states = observation(next_states)

            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            #Update q values
            all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
        
            
            self.update(states.tolist(), all_q_values.tolist())

def observation(frame):
    """
    returns the current observation from a frame
    :param frame: ([int] or [float]) environment frame
    :return: ([int] or [float]) the observation
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = []
    episode_i=0
    sum_total_replay_time=0

    for episode in range(episodes):

        print("q_learning episode: {}".format(episode))

        episode_i += 1

        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()

        if double and soft:
            model.target_update()
        
        # Reset state
        state = env.reset()

        # state = observation(state)

        # print(state.shape)

        done = False
        total = 0
        
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action % 6)
            # next_state = observation(next_state)


            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()
             
            if done:
                if not replay:
                    q_values[int(action / 6)][int(action % 6)] = reward
                    # Update network weights
                    model.update(state, q_values)
                break

            if replay:
                t0 = time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1 = time.time()
                sum_total_replay_time+=(t1-t0)
            else: 
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                q_values[int(action / 6)][int(action % 6)] = reward + gamma * torch.max(q_values_next).item()
                # print(q_va)
                model.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        # plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time/episode_i)
        
    return final

# # Get DQN results
# simple_dqn = DQN(n_state, n_action, n_hidden, lr)
# simple = q_learning(env, simple_dqn, episodes, gamma=.9, epsilon=0.3)



# Get replay results
dqn_replay = DQN_replay(n_state, n_action, n_hidden, lr)
replay = q_learning(env, dqn_replay, 
                    episodes, gamma=.9, 
                    epsilon=0.2, replay=True, 
                    title='DQL with Replay')

## https://github.com/ritakurban/Practical-Data-Science/blob/master/DQL_CartPole.ipynb