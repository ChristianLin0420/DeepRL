
import torch
import gym
import slimevolleygym

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.model = torch.load("107062240_hw2.pth")

    def act(self, observation, reward, done):
        return self.model(observation)

env = gym.make("SlimeVolleyNoFrameskip-v0"))

obs = env.reset()
done = False
total_reward = 0

agent = Agent()

while not done:
  action = my_policy(obs)
  obs, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print("score:", total_reward)