from gym import Env
from gym.utils import seeding
from gym.spaces import Box
import numpy as np
import random
import time


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
        self.state = np.array([self.x,self.y,self.a,self.b,self.c,self.d,self.e,self.f])
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.low_state = np.array([self.x_init-5,self.y_init-5,self.a,self.b,self.c,self.d,self.e,self.f],dtype=np.float32)
        self.high_state = np.array([self.x_init+5,self.y_init+5,self.a,self.b,self.c,self.d,self.e,self.f],dtype=np.float32)
        self.observation_space = Box(low=self.low_state, high=self.high_state)
        self.seed()
    def seed(self,seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        done = False
        self.state[0]-= action[0][0]
        self.state[1]-= action[0][1]
        reward = self.reward()
        if self.state[0]<0.1 and self.state[1]<0.1 :
            done = True
        elif (self.end - self.start) >= 120:
            done = True

        return self.state,reward,done
    def render(self):
        pass
    def reset(self):
        self.state = np.array([random.uniform(-4,4),random.uniform(-4,4),self.a,self.b,self.c,self.d,self.e,self.f])
        return self.state
    def reward(self):
        state = self.state
        dist = np.sqrt((state[0]-self.x_min)**2 + (state[1]-self.y_min)**2)
        reward = 1/dist
        return reward
    def start_time(self):
        self.start = time.time()
    def end_time(self):
        self.end = time.time()
env = QuadraticEnv()
print(env.observation_space.shape[0])
# EPISODES =10
# for eps in range(EPISODES):
#     obs = env.reset()
#     print(f'Episode: {eps}')
#     done = False
#     R = 0
#     env.start_time()
#     env.end_time()
#     while not done:
#         action = np.array([[0.9,0.6]])
#         obs,reward,done =env.step(action)
#         R += reward
#         env.end_time()
#     print(f'Total Reward: {R}')

