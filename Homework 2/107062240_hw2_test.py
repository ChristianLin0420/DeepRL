
import cv2
import copy
import random

import torch
import gym

import torch.nn as nn

# import slimevolleygym

# env = gym.envs.make("SlimeVolleyNoFrameskip-v0")

# # Number of states
# n_state = env.observation_space.shape[0]
# # Number of actions
# n_action = env.action_space.n

# print(n_state)
# print(n_action)

class DQN(nn.Module):
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):

            super(DQN, self).__init__()

            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim*2, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), 0.001)



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

class DQN_double(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.target = copy.deepcopy(self.model)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            s = observation(s)
            s = s.reshape((84, 84))
            return self.target(torch.Tensor(s))
        
    def target_update(self):
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[int(action / 6)][int(action % 6)] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[int(action / 6)][int(action % 6)] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            for s, t in zip(states, targets):
                self.update(s, t)
            # self.update(states, targets)

def observation(frame):
    """
    returns the current observation from a frame
    :param frame: ([int] or [float]) environment frame
    :return: ([int] or [float]) the observation
    """
    # print(frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
      self.model = DQN_double(84, 6, 50, 0.001)
      # self.model = torch.load("107062240_hw2_data.pth")
      self.model.load_state_dict(torch.load("107062240_hw2_data.pth"), strict=False)
    #   print(self.model)

    def act(self, observation, reward, done):
        return torch.argmax(self.model.predict(observation)).item() % 6

# obs = env.reset()
# done = False
# total_reward = 0

# agent = Agent()
# reward = 0
# done = False

# while not done:
#   action = agent.act(obs, reward, done)

#   obs, reward, done, info = env.step(action)
#   total_reward += reward
#   env.render()

# print("score:", total_reward)
