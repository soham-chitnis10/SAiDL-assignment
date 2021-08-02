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
import Quadratic_2D_env
import os
import gym

def evaluate_policy(policy, env, eval_episodes = 10):
    '''
        function to return the average reward of the policy over 10 runs
    '''
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, log_prob = policy.select_action(np.array(obs) )
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("the average reward is: {0}".format(avg_reward))
    #return avg_reward

def render_policy(policy,env):
    '''
        Function to see the policy in action
    '''
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action,_,_,_ = policy.select_action(np.array(obs))
        obs, reward, done, _ = env.step(action)

    env.close()

def main(args):

    # create env
    env = Quadratic_2D_env()
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # follow different logic depending on action space of env
    hidden_size = args.hidden_size

        # get env info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    max_action = (env.action_space.high)
    min_action = (env.action_space.low)

    print("number of actions:{0}, dim of states: {1},max_action: {2}, min_action: {3}".format(action_dim,state_dim,max_action,min_action))

    # create policy
    policy = REINFORCE(state_dim, hidden_size, action_dim)

    

    # start of experiment: Keep looping until desired amount of episodes reached
    max_episodes = args.num_episodes
    total_episodes = 0 # keep track of amount of episodes that we have done

    while total_episodes < max_episodes:

        obs = env.reset()
        done = False
        trajectory = [] # trajectory info for reinforce update
        episode_reward = 0 # keep track of rewards per episode

        while not done:
            action, ln_prob = policy.select_action(np.array(obs))
            next_state, reward, done, _ = env.step(action)
            trajectory.append([np.array(obs), action, ln_prob, reward, next_state, done])

            obs = next_state
            episode_reward += reward

        total_episodes += 1


        if total_episodes % 10 == 0:
            evaluate_policy(policy,env)

        env.close()


if __name__ == '__main__':

    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env-name', default="HalfCheetah-v2", help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=456, metavar='N', help='random seed (default: 456)')
    parser.add_argument('--num-episodes', type=int, default=2000, metavar='N',
                        help='maximum number of episodes (default:2000)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--action-space', type=str, default = 'continuous', help = "Whether the action space is continuous or not to use the appropriate REINFORCE algo")

    args = parser.parse_args()

    main(args)