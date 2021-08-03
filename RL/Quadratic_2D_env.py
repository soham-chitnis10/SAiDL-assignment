from gym import Env
from gym.utils import seeding
from gym.spaces import Box
import numpy as np
import random
import time



class QuadraticEnv(Env):
    def __init__(self):
        self.seed()
        self.a = random.uniform(0,10)
        self.b = random.uniform(0,10)
        self.c = random.uniform(-10,10)
        self.d = random.uniform(-10,10)
        self.e = random.uniform(-10,10)
        self.f = random.uniform(-10,10)
        self.x = random.uniform(-4,4)
        self.y = random.uniform(-4,4)
        self.state = np.array([self.x,self.y,self.a,self.b,self.c,self.d,self.e,self.f])
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.low_state = np.array([random.uniform(-4,4)-5,random.uniform(-4,4)-5,self.a,self.b,self.c,self.d,self.e,self.f],dtype=np.float32)
        self.high_state = np.array([random.uniform(-4,4)+5,random.uniform(-4,4)+5,self.a,self.b,self.c,self.d,self.e,self.f],dtype=np.float32)
        self.observation_space = Box(low=self.low_state, high=self.high_state)
        self.set_minima()
    def set_minima(self):

        det = 4 * self.a * self.b - self.c * self.c
        while det == 0:
            self.reset()
            det = 4 * self.a * self.b - self.c * self.c
        
        self.x_min = (-2 * self.b * self.d + self.c * self.e)/det
        self.y_min = (self.c * self.d - 2 * self.a * self.e)/det   
    def seed(self,seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        done = False
        self.state[0]-= action[0][0]
        self.state[1]-= action[0][1]
        reward = self.reward()
        if abs(self.state[0]-self.x_min)<0.1 and abs(self.state[1]-self.y_min)<0.1 :
            done = True
        elif (self.end - self.start) >= 60:
            done = True

        return self.state,reward,done
    def render(self):
        pass
    def reset(self):
        self.a = random.uniform(0,10)
        self.b = random.uniform(0,10)
        self.c = random.uniform(-10,10)
        self.d = random.uniform(-10,10)
        self.e = random.uniform(-10,10)
        self.f = random.uniform(-10,10)
        self.set_minima()
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

