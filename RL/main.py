import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
from reinforce import REINFORCE
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from Quadratic_2D_env import QuadraticEnv
import gym

def main():

    # create env
    env = QuadraticEnv()
    env.seed(456)
    torch.manual_seed(456)
    np.random.seed(456)

    hidden_size = 32

    # get env info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space

    print("number of actions:{0}, dim of states: {1}".format(action_dim,state_dim))

    # create policy
    policy = REINFORCE(state_dim, hidden_size, action_dim)

    

    # start of training: Keep looping until desired amount of episodes reached
    max_episodes = 1000
    total_episodes = 0 # keep track of amount of episodes that we have done
    max_reward = 0
    max_reward_ep = 0
    reward_list = []
    successful_episodes = []
    while total_episodes < max_episodes:

        obs = env.reset()
        done = False
        trajectory = [] # trajectory info for reinforce update
        episode_reward = 0 # keep track of rewards per episode

        while not done:
            action, ln_prob = policy.select_action(np.array(obs))
            next_state, reward, done = env.step(action)
            trajectory.append([obs, action, ln_prob, reward, next_state, done])
            obs = next_state
            episode_reward += reward
        reward_list.append(env.reward())
        print(f'Episode: {total_episodes} Reward: {episode_reward} function: {env.a}x^2 + {env.b}y^2 + {env.c}xy + {env.d}x + {env.e}y + {env.f} x:{env.state[0]} y:{env.state[1]} x_min:{env.x_min} y_min:{env.y_min}')
        if episode_reward > max_reward:
            max_reward = episode_reward
            max_reward_ep = total_episodes
        if abs(env.state[0]-env.x_min)<0.1 and abs(env.state[1]-env.y_min)<0.1 :
            successful_episodes.append(total_episodes)
        total_episodes += 1
        policy_loss = policy.train(trajectory)
    
    print(f'Max Reward is {max_reward} occured on episode {max_reward_ep}')
    eps = [ep for ep in range(1,max_episodes+1)]
    plt.plot(eps,reward_list)
    plt.show()
    print('Successful Episodes:')
    print(successful_episodes)

    print('Testing starts:')
    #start of testing
    max_episodes = 1000
    total_episodes = 0 # keep track of amount of episodes that we have done
    max_reward = 0
    max_reward_ep = 0
    reward_list = []
    succesful_episodes = []
    while total_episodes < max_episodes:

        obs = env.reset()
        done = False
        episode_reward = 0 # keep track of rewards per episode

        while not done:
            action, ln_prob = policy.select_action(np.array(obs))
            next_state, reward, done = env.step(action)
            trajectory.append([obs, action, ln_prob, reward, next_state, done])
            obs = next_state
            episode_reward += reward
        reward_list.append(env.reward())
        print(f'Episode: {total_episodes} Reward:{episode_reward} function: {env.a}x^2 + {env.b}y^2 + {env.c}xy + {env.d}x + {env.e}y + {env.f} x:{env.state[0]} y:{env.state[1]} x_min:{env.x_min} y_min:{env.y_min}')
        if abs(env.state[0]-env.x_min) < 0.1 and (env.state[1]-env.y_min) < 0.1:
          succesful_episodes.append(total_episodes)
        if episode_reward > max_reward:
            max_reward = episode_reward
            max_reward_ep = total_episodes

        total_episodes += 1
    
    print(f'Max Reward is {max_reward} occured on episode {max_reward_ep}')
    eps = [ep for ep in range(1,max_episodes+1)]
    plt.plot(eps,reward_list)
    plt.show()
    print('Succesful Episodes:')
    print(succesful_episodes)



if __name__ == '__main__':

    main()