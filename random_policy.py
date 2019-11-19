import gym # openAi gym
from gym import envs
import numpy as np 
import pandas as pd 
import random

import warnings
warnings.filterwarnings('ignore')

env = gym.make('Taxi-v3')   # Here you set the environment
env._max_episode_steps = 40000
env.reset()

numOfSteps = np.zeros(1000)

for i_episode in range(1000):
    observation = env.reset()
    t=0
    while True:
        # env.render()
        action = env.action_space.sample()  # Random action
        observation, reward, done, info = env.step(action)
        t+=1
        if done:
            print(info, reward)
            print("Episode finished after {} timesteps".format(t))
            numOfSteps[i_episode] = t
            break
env.close()

print(np.average(numOfSteps))
