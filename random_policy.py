import gym # openAi gym
from gym import envs
import numpy as np 
import pandas as pd 
import random

import warnings
warnings.filterwarnings('ignore')

env = gym.make('Taxi-v3')   # Here you set the environment 
env.reset()

numOfSteps = np.zeros(1000)

for i_episode in range(1000):
    observation = env.reset()
    t=0
    while True:
        env.render()
        action = env.action_space.sample()  # Random action
        observation, reward, done, info = env.step(action)
        t+=1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            numOfSteps[i_episode] = t+1
            break
env.close()

print(np.average(numOfSteps))
