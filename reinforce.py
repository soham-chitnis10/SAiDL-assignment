from policy import Gaussian_Policy
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class REINFORCE:
    '''
    Implementation of the basic online reinforce algorithm for Gaussian policies.
    '''

    def __init__(self, num_inputs, hidden_size, action_space, lr_pi = 3e-4,gamma = 0.99):

        self.gamma = gamma
        self.action_space = action_space
        self.policy = Gaussian_Policy(num_inputs, hidden_size, action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr_pi)


    def select_action(self,state):

        state = torch.from_numpy(state).float().unsqueeze(0) # just to make it a Tensor obj
        # get mean and std
        mean, std = self.policy.forward(state)

        # create normal distribution
        normal = Normal(mean, std)

        # sample action
        action = normal.sample()

        # get log prob of that action
        ln_prob = normal.log_prob(action)
        ln_prob = ln_prob.sum()
	# squeeze action into [-1,1]
        action = torch.tanh(action)
        # turn actions into numpy array
        action = action.numpy()

        return action, ln_prob #, mean, std

    def train(self, trajectory):

        '''
        The training is done using the rewards-to-go formulation of the policy gradient update of Reinforce.
        trajectory: a list of the form [( state , action , lnP(a_t|s_t), reward ), ...  ]
        '''

        log_probs = [item[2] for item in trajectory]
        rewards = [item[3] for item in trajectory]
        states = [item[0] for item in trajectory]
        actions = [item[1] for item in trajectory]

	#calculate rewards to go
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).cuda()

        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            log_prob = log_prob.cuda()
            policy_loss.append( - log_prob * R)


        policy_loss = torch.stack( policy_loss ).sum()
        # update policy weights
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss
