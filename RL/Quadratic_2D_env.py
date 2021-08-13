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
        self.steps = 0
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
        reward = self.reward_updated(action)
        self.state[0]-= action[0][0]
        self.state[1]-= action[0][1]
        self.steps += 1
        if abs(self.state[0]-self.x_min)<0.1 and abs(self.state[1]-self.y_min)<0.1 :
            done = True
        elif self.steps > 200:
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
        self.steps = 0
        return self.state
    def reward(self):
        '''
        Reward function:
        Depends only on the state 
        Reward is given by th inverse of the distance between the state and the goal.
        '''
        state = self.state
        dist = np.sqrt((state[0]-self.x_min)**2 + (state[1]-self.y_min)**2)
        reward = 1/dist
        return reward
    def reward_new(self,action):
        '''
        Reward function:
        Depends on state as well as action
        Reward is given by exponentiation of the difference in coordinates with action along both directions.
        Reward is restricted to be less than 1e+17
        '''
        reward = np.exp(action[0][0]*(-(self.x_min - self.state[0]))) + np.exp(action[0][1]*(-(self.y_min - self.state[1])))
        while reward > 1e+17:
          reward /= 1e+17
        return reward
    def reward_updated(self,action):
      '''
      Reward function:
      Depends on state as well as action
      Reward is awarded as 0 when action is the opposite direction e.g x coordinate is supposed to increased to reach the minima but the action is to reduce the x cooordinate in that case reward is 0.
      '''
      if (self.state[0]-self.x_min)*action[0][0] > 0:
        r_x = (self.state[0]-self.x_min)*action[0][0]
      else:
        r_x = 0
      if (self.state[1]-self.y_min)*action[0][1] > 0:
        r_y = (self.state[1]-self.y_min)*action[0][1]
      else:
        r_y = 0
      reward = r_x + r_y
      return reward
    def reward_func_4(self,action):
      '''
      Reward function:
      Depends on state as well as action
      Reward is awarded as 0 when action is the opposite direction e.g x coordinate is supposed to increased to reach the minima but the action is to reduce the x cooordinate in that case reward is 0.
      '''
      if action[0][0]/(self.state[0]-self.x_min + 1e-9) > 0:
        r_x = action[0][0]/(self.state[0]-self.x_min + 1e-9)
      else:
        r_x = 0
      if action[0][1]/(self.state[1]-self.y_min + 1e-9) > 0:
        r_y = action[0][1]/(self.state[1]-self.y_min + 1e-9)
      else:
        r_y = 0
      reward = r_x + r_y
      return reward
