import argparse
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
    env = QuadraticEnv(1,4,0,0,0,0)
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

    

    # start of experiment: Keep looping until desired amount of episodes reached
    max_episodes = 400
    total_episodes = 0 # keep track of amount of episodes that we have done
    max_reward = 0
    max_reward_ep = 0
    while total_episodes < max_episodes:

        obs = env.reset()
        done = False
        trajectory = [] # trajectory info for reinforce update
        episode_reward = 0 # keep track of rewards per episode
        env.start_time()
        env.end_time()

        while not done:
            action, ln_prob = policy.select_action(np.array(obs))
            next_state, reward, done = env.step(action)
            trajectory.append([obs, action, ln_prob, reward, next_state, done])
            obs = next_state
            episode_reward += reward
            env.end_time()
        print(f'Episode: {total_episodes} Reward: {episode_reward}')
        if episode_reward > max_reward:
            max_reward = episode_reward
            max_reward_ep = total_episodes

        total_episodes += 1
        policy_loss = policy.train(trajectory)
    
    print(f'Max Reward is {max_reward} occured on episode {max_reward_ep}')


if __name__ == '__main__':

    main()
