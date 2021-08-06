from gym import Env
from gym.utils import seeding
from gym.spaces import Box
import numpy as np
import random


class QuadraticEnv(Env):
    def __init__(self):
        self.a = 1
        self.b = 4
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.x_init = random.uniform(-4,4)
        self.y_init = random.uniform(-4,4)
        self.x = self.x_init
        self.y = self.y_init
        self.x_min = 0
        self.y_min = 0
        self.state = [self.x,self.y,self.a,self.b,self.c,self.d,self.e,self.f]
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.low_state = np.array([self.x_init-5,self.y_init-5,self.a,self.b,self.c,self.d,self.e,self.f])
        self.high_state = np.array([self.x_init+5,self.y_init+5,self.a,self.b,self.c,self.d,self.e,self.f])
        self.observation_space = Box(low=self.low_state, high=self.high_state)
        self.seed()
    def seed(self,seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        self.state[0]-= action[0]
        self.state[1]-= action[1]
        reward = self.reward()
        return reward,False
    def render(self):
        pass
    def reset(self):
        self.state =np.array([random.uniform(-4,4),random.uniform(-4,4),self.a,self.b,self.c,self.d,self.e,self.f])
    def reward(self):
        state = self.state
        dist = np.sqrt((state[0]-self.x_min)**2 + (state[1]-self.y_min)**2)
        reward = 1/dist
        return reward
    def _get_obs(self):
        pass
# env = QuadraticEnv()
# EPISODES =10
# for eps in range(EPISODES):
#     env.reset()
#     print(f'Episode: {eps}')
#     env.step()