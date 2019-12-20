import time
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gym
agent.saver.restore(agent.sess, 'model/chase'+str(size)+'.ckpt')

for _ in range(5):
    r = 0
    state, goal = env.reset()['observation'],env.reset()['desired_goal'] 
    
    # run through the episode
    for _ in range(50):
        env.render()
        state=state.reshape(1,25)
        goal=goal.reshape(1,3)
        action = agent.choose_action(state, goal, 0)
        next_state, reward, done = env.step(action)
        r += reward
        state = next_state
        time.sleep(0.04)

        # render the final result
        if done:
            env.render()
            break
    print("reward : %06.2f"%r, " success :", reward==0)
