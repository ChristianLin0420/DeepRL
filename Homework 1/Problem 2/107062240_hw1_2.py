import numpy as np
import gym
import matplotlib.pyplot as plt

SHOW_ENV_DISPLAY_FREQUENCY = 100
DEFAULT_ITERATION_COUNT = 100000

Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

# Creates a table of Q_values (state-action) initialized with zeros
# Initialize Q(s, a), for all s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, ·) = 0.
def createQ_table():
    q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
    # print("q_table shape: ", q_table.shape)
    # print("q_tabel content: ", q_table[:1])

    return q_table

def get_discrete_state(state):
    discrete_state = state / np_array_win_size + np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

# Choosing action using policy
# Sutton's code pseudocode: Choose A from S using policy derived from Q (e.g., ε-greedy)
# %10 exploration to avoid stucking at a local optima
def epsilon_greedy_policy(discrete_state, q_table, epsilon = 0.1):
    # choose a random float from an uniform distribution [0.0, 1.0)
    explore_exploit = np.random.random()

    if explore_exploit < epsilon:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(q_table[discrete_state])

    return action

def q_learning(num_episodes = DEFAULT_ITERATION_COUNT, gamma_discount = 0.9, alpha = 0.5, epsilon = 0.1):

    reward_cache = list()
    step_cache = list()
    q_table = createQ_table()
    discrete_state = get_discrete_state(env.reset())

    for episode in range(0, num_episodes):
        discrete_state = get_discrete_state(env.reset())
        done = False
        reward_cum = 0
        step_cum = 0

        while(done == False):
            action = epsilon_greedy_policy(discrete_state, q_table)
            new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.
            reward_cum += reward
            step_cum += 1
            new_discrete_state = get_discrete_state(new_state)

            if episode % SHOW_ENV_DISPLAY_FREQUENCY == 0: #render
                env.render()

            if not done: #update q-table
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - alpha) * current_q + alpha * (reward + gamma_discount * max_future_q)
                q_table[discrete_state + (action,)] = new_q

            discrete_state = new_discrete_state

        reward_cache.append(reward_cum)
        step_cache.append(step_cum)

        if episode % SHOW_ENV_DISPLAY_FREQUENCY == 0: #render
            print("q_learning episodes count: ", episode)

    return q_table, reward_cache, step_cache

def sarsa(num_episodes = DEFAULT_ITERATION_COUNT, gamma_discount = 0.9, alpha = 0.5, epsilon = 0.1):
    
    reward_cache = list()
    step_cache = list()
    q_table = createQ_table()
    discrete_state = get_discrete_state(env.reset())

    for episode in range(0, num_episodes):
        discrete_state = get_discrete_state(env.reset())
        done = False
        reward_cum = 0
        step_cum = 0

        while(done == False):
            action = epsilon_greedy_policy(discrete_state, q_table)
            new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.
            reward_cum += reward
            step_cum += 1
            new_discrete_state = get_discrete_state(new_state)

            if episode % SHOW_ENV_DISPLAY_FREQUENCY == 0: #render
                env.render()

            if not done: #update q-table
                next_state_value = q_table[new_discrete_state][action]
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - alpha) * current_q + alpha * (reward + gamma_discount * next_state_value)
                q_table[discrete_state + (action,)] = new_q

            discrete_state = new_discrete_state

        reward_cache.append(reward_cum)
        step_cache.append(step_cum)

        if episode % SHOW_ENV_DISPLAY_FREQUENCY == 0: #render
            print("sarsa episodes count: ", episode)

    return q_table, reward_cache, step_cache

def plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA):
    """
    Visualizes the reward convergence
    
    Args:
        reward_cache -- type(list) contains cumulative_reward
    """
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0
            
    cum_rewards_SARSA = []
    rewards_mean = np.array(reward_cache_SARSA).mean()
    rewards_std = np.array(reward_cache_SARSA).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0      
    # prepare the graph    
    plt.plot(cum_rewards_q, label = "q_learning")
    plt.plot(cum_rewards_SARSA, label = "SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    plt.savefig('cumulative_reward.png')
    
def plot_number_steps(step_cache_qlearning, step_cache_SARSA):
    """
        Visualize number of steps taken
    """    
    cum_step_q = []
    steps_mean = np.array(step_cache_qlearning).mean()
    steps_std = np.array(step_cache_qlearning).std()
    count = 0 # used to determine the batches
    cur_step = 0 # accumulate reward for the batch
    for cache in step_cache_qlearning:
        count = count + 1
        cur_step += cache
        if(count == 10):
            # normalize the sample
            normalized_step = (cur_step - steps_mean)/steps_std
            cum_step_q.append(normalized_step)
            cur_step = 0
            count = 0
            
    cum_step_SARSA = []
    steps_mean = np.array(step_cache_SARSA).mean()
    steps_std = np.array(step_cache_SARSA).std()
    count = 0 # used to determine the batches
    cur_step = 0 # accumulate reward for the batch
    for cache in step_cache_SARSA:
        count = count + 1
        cur_step += cache
        if(count == 10):
            # normalize the sample
            normalized_step = (cur_step - steps_mean)/steps_std
            cum_step_SARSA.append(normalized_step)
            cur_step = 0
            count = 0      
    # prepare the graph    
    plt.plot(cum_step_q, label = "q_learning")
    plt.plot(cum_step_SARSA, label = "SARSA")
    plt.ylabel('Number of iterations')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Iteration number untill game ends")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    plt.savefig('number_steps.png')
    

    
def plot_qlearning_smooth(reward_cache):
    """
    Visualizes the reward convergence using weighted average of previous 10 cumulative rewards
    NOTE: Normalization gives better visualization
    
    Args:
        reward_cache -- type(list) contains cumulative_rewards for episodes
    """
    mean_rev = (np.array(reward_cache[0:11]).sum())/10
    # initialize with cache mean
    cum_rewards = [mean_rev] * 10
    idx = 0
    for cache in reward_cache:
        cum_rewards[idx] = cache
        idx += 1
        smooth_reward = (np.array(cum_rewards).mean())
        cum_rewards.append(smooth_reward)
        if(idx == 10):
            idx = 0
        
    plt.plot(cum_rewards)
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning  Convergence of Cumulative Reward")
    plt.legend(loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    #SARSA
    q_table_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa()
    # QLEARNING
    q_table_qlearning, reward_cache_qlearning, step_cache_qlearning = q_learning()

    plot_number_steps(step_cache_qlearning, step_cache_SARSA)
    # Visualize the result
    plot_cumreward_normalized(reward_cache_qlearning,reward_cache_SARSA)
