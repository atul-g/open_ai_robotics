import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from network_classes import create_actor_network, create_critic_network
from tensorflow.keras import losses

import gym


# REPLAY BUFFER
class Episode_experience():
    def __init__(self):
        self.memory = []  #INITIALIZED

    def add(self, state, action, reward, next_state, done, goal):
        # add tuple of experience to buffer
        self.memory += [(state, action, reward, next_state, done, goal)]

    def clear(self):
        # clear the buffer
        self.memory = []



class DDPGAgent:
    def __init__(self,action_low=-1, action_high=1, gamma=0.98, actor_learning_rate=0.001,
                 critic_learning_rate=0.001, tau=1e-3):


        # initialize limits for action clipping
        self.action_low = action_low          ############## DOUBT
        self.action_high = action_high

        # setup variables for RL
        self.tau = tau
        self.gamma = gamma
        self.batch_size = 128
        self.gradient_norm_clip = None
        self.a_learning_rate = actor_learning_rate
        self.c_learning_rate = critic_learning_rate #


        # initialize experience buffer
        self.memory = []
        self.buffer_size = int(5e4)
        
        #make the networks
        self.actor_model=create_actor_network()
        self.critic_model=create_critic_network()
        
        self.target_actor_model=create_actor_network()
        self.target_actor_model.set_weights(self.actor_model.get_weights())

        self.target_critic_model=create_critic_network()
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        q_target=reward+self.gamma+(1-done)*target_critic_model.predict() ##################
        
        model.compile(loss=losses.mean_squared_error(q_target, ))
        # execute noisy version of policy output
    def choose_action(self, state, goal, variance):
        input=numpy.concatenate((state, goal), axis=0) #shape becomes (28,)
        input.resize(1,28)
        action = self.actor_model.predict(input)[0]
        return np.clip(np.random.normal(action, variance), self.action_low, self.action_high)
            
    def remember(self, ep_experience):
        self.memory += ep_experience.memory
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:] # empty the first memories
            
        # network update step from experience replay
    def replay(self, optimization_steps=1):
        # if there's no enough transitions, do nothing
        if len(self.memory) < self.batch_size: 
            return 0, 0
                
        # perform optimization for optimization_steps
        a_losses = 0
        c_losses = 0
        
        for _ in range(optimization_steps):
            # get a minibatch
            minibatch = np.vstack(random.sample(self.memory, self.batch_size))

            # stack states, actions and rewards
            ss = np.vstack(minibatch[:,0])
            acs = np.vstack(minibatch[:,1])
            rs = minibatch[:,2]
            nss = np.vstack(minibatch[:,3])
            ds = minibatch[:,4]
            gs = np.vstack(minibatch[:,5])           #####################
            
            actor_model.fit() ##########################
            critic_model.fit() ######################
            
            
    def update_target_nework(self):
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        


################### MAIN


env=gym.make('FetchReachAndPick-v1')
agent=DDPGAgent()


# variables for network training
num_epochs = 200
num_episodes = 80
optimization_steps = 20

ep_mean_r = []
success_rate = []


# initialize buffers for episode experience
ep_experience = Episode_experience()

# time the performance of network
total_step = 0
start = time.clock()

#use HER
her=True
            
        # loop for num_epochs
for i in range(num_epochs):

    # tracking successes per epoch
    successes = 0
    ep_total_r = 0
        
    # loop over episodes
    for n in range(num_episodes):
        # reset env
        a=env.reset()
        state, goal = a['observation'], a['desired_goal']
        done=False

        # run env for episode_length steps
        while not done:
            # track number of samples
            total_step += 1

            # obtain action by agent
            action = agent.choose_action(state, goal, variance) 

            # execute action in env
            obs, reward, done, info= env.step(action)

            # track reward and add regular experience
            ep_total_r += reward
            next_state = obs['observation']
            ep_experience.add(state, action, reward, next_state, done, goal)
            state = next_state

            # add experience using her
            if total_step % 100 == 0 or done:
                if use_her: 
                    # add additional experience for each time step
                    for t in range(len(ep_experience.memory)):
                        # get K future states per time step
                        ################
                        # Can we improve the K-future strategy? K=4 here
                        ################
                        for _ in range(4):
                            # get random future t
                            future = np.random.randint(t, len(ep_experience.memory))

                            # get new goal at t_future
                            goal_ = ep_experience.memory[future][3] 
                            state_ = ep_experience.memory[t][0]
                            action_ = ep_experience.memory[t][1]
                            next_state_ = ep_experience.memory[t][3]
                            done_, reward_ = True, 0

                            # add new experience to her
                            ep_experience_her.add(state_, action_, reward_, 
                                                  next_state_, done_, goal_)

                    # add this her experience to agent buffer
                    agent.remember(ep_experience_her)
                    ep_experience_her.clear()

                # add regular experience to agent buffer
                agent.remember(ep_experience)
                ep_experience.clear()

                # perform optimization step
                variance *= 0.9995
                agent.update_target_net()

            # if episode ends start new episode
            if done:
                break

        # keep track of successes
        successes += reward==0 and done

    # obtain success rate per epoch
    success_rate.append(successes/num_episodes)
    ep_mean_r.append(ep_total_r/num_episodes)
        
    # print statistics per epoch
    print("\repoch", i+1, "success rate", success_rate[-1], 
          "ep_mean_r %.2f"%ep_mean_r[-1], 'exploration %.2f'%variance, end=' '*10)
          
     # output total training time
    print("Training time : %.2f"%(time.clock()-start), "s")
        
    plt.figure()
    plt.plot(success_rate)
    plt.title('Success Rates')
    plt.show()
    
    plt.figure()
    plt.plot(ep_mean_r)
    plt.title('Mean Episode Rewards')
    plt.show()
    
    
    
actor_model.save("actor_model.h5")
critic_model.save("critic_model.h5")
      
        
        
        
        



























