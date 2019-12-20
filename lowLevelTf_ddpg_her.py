#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import libraries
import time
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gym

# create replay buffer
class Episode_experience():
    def __init__(self):
        # initialize buffer
        self.memory = []
        
    def add(self, state, action, reward, next_state, done, goal):
        # add tuple of experience to buffer
        self.memory += [(state, action, reward, next_state, done, goal)]
        
    def clear(self):
        # clear the buffer
        self.memory = []
        
class DDPGAgent:
    def __init__(self, action_low=-1, action_high=1, gamma=0.98, actor_learning_rate=0.001, 
                 critic_learning_rate=0.001, tau=1e-3):
        
        # initialize limits for action clipping
        self.action_low = action_low
        self.action_high = action_high
        
        # setup variables for RL
        self.tau = tau 
        self.gamma = gamma 
        self.batch_size = 64
        self.gradient_norm_clip = None
        self.a_learning_rate = actor_learning_rate
        self.c_learning_rate = critic_learning_rate 
        
        # initialize experience buffer
        self.memory = []
        self.buffer_size = int(5e5)
        
        # create a neural network
        self._construct_nets()
        
    def _construct_nets(self):
        # initialize computation graph
        #tf.reset_default_graph()
        self.sess = tf.Session()
        
        # initialize palce holders for computation
        self.R = tf.placeholder(tf.float32, [None, ], 'r')
        self.D = tf.placeholder(tf.float32, [None, ], 'done')
        self.G = tf.placeholder(tf.float32, [None, 3], 'goal')
        self.S = tf.placeholder(tf.float32, [None, 25], 'state')
        self.S_ = tf.placeholder(tf.float32, [None, 25], 'next_state')
        
        # create actor and critic networks along with target networks
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, self.G, scope='eval')
            self.a_ = self._build_a(self.S_, self.G, scope='target')
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, self.a, self.G, scope='eval')
            self.q_ = self._build_c(self.S_, self.a_, self.G, scope='target')
        
        # get list of parameters for each network
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           scope='Critic/target')

        # soft update operation of target networks
        self.soft_update_op = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), 
                                tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # operation to compute target q value
        q_target = self.R + self.gamma * (1-self.D) * self.q_
        
        # loss function for actor and critic networks
        self.c_loss = tf.losses.mean_squared_error(q_target, self.q)
        self.a_loss = - tf.reduce_mean(self.q)
        
        # perform optimization based on gradient clipping
        if self.gradient_norm_clip is not None:
            # initialize critic optimizer
            c_optimizer = tf.train.AdamOptimizer(self.c_learning_rate)
            c_gradients = c_optimizer.compute_gradients(self.c_loss, 
                                                        var_list=self.ce_params)
            
            # perform gradient clipping
            for i, (grad, var) in enumerate(c_gradients):
                if grad is not None:
                    c_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip), var)
            self.c_train = c_optimizer.apply_gradients(c_gradients)
            
            # initialize actor optimizer
            a_optimizer = tf.train.AdamOptimizer(self.a_learning_rate)
            a_gradients = c_optimizer.compute_gradients(self.a_loss, 
                                                        var_list=self.ae_params)
            
            # perform gradient clipping
            for i, (grad, var) in enumerate(a_gradients):
                if grad is not None:
                    a_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip), var)
            self.a_train = a_optimizer.apply_gradients(a_gradients)
        else:
            # perform optimization without gradient clipping
            self.c_train = tf.train.AdamOptimizer(self.c_learning_rate).minimize(self.c_loss, var_list=self.ce_params)
            self.a_train = tf.train.AdamOptimizer(self.a_learning_rate).minimize(self.a_loss, var_list=self.ae_params)
            
        # initialize model saver
        self.saver = tf.train.Saver()        
        
        # variable initializer for session
        self.sess.run(tf.global_variables_initializer())
    
    def _build_a(self, s, g, scope): 
        # actor network based on UVFA (Schaul et al. 2015)
        with tf.variable_scope(scope):
            # use both state and goal as input for network
            net = tf.concat([s, g], 1)                  ################################
            net = tf.layers.dense(net, 256, tf.nn.relu)
            net = tf.layers.dense(net, 256, tf.nn.relu)
            net = tf.layers.dense(net, 256, tf.nn.relu)
            a = tf.layers.dense(net, 4, tf.nn.tanh)
            return a * (self.action_high-self.action_low)/2 + (self.action_high+self.action_low)/2
    
    def _build_c(self, s, a, g, scope): 
        # critic network based on UVFA (Schaul et al. 2015)
        with tf.variable_scope(scope):
            net = tf.concat([s, a, g], 1)                 ################################
            net = tf.layers.dense(net, 256, tf.nn.relu)
            net = tf.layers.dense(net, 256, tf.nn.relu)
            net = tf.layers.dense(net, 256, tf.nn.relu)
            return tf.layers.dense(net, 1)
    
    # execute noisy version of policy output
    def choose_action(self, state, goal, variance): 
        action = self.sess.run(self.a, {self.S: state, self.G: goal})[0]
        return np.clip(np.random.normal(action, variance), self.action_low, self.action_high)
    
    # append episode experience to replay buffer
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
            gs = np.vstack(minibatch[:,5])            ################################
            
            # obtain the losses and perform one gradient update step
            a_loss, c_loss, _, _ = self.sess.run([self.a_loss, self.c_loss, self.a_train, self.c_train],
                                                 {self.S: ss, self.a: acs, self.R: rs,
                                                  self.S_: nss, self.D: ds, self.G: gs})
            
            # accumulate losses over steps
            a_losses += a_loss
            c_losses += c_loss
            
        return a_losses/optimization_steps, c_losses/optimization_steps
    
    # utility function to update target network
    def update_target_net(self):
        self.sess.run(self.soft_update_op)
'''
# environment to evaluate ddpg
class ChaseEnv():
    def __init__(self):

        # threshold for detecting success
        self.thr = 1 
        
    def reset(self):
        # reset goal and state at end of episode
        self.goal = self.size * (2*np.random.random(2)-1) 
        self.state = self.size * (2*np.random.random(2)-1)
        return np.copy(self.state/self.size), np.copy(self.goal/self.size)

    def reward_func(self, state, goal):
        # define two types of states
        good_done = np.linalg.norm(state-goal) <= self.thr
        bad_done = np.max(np.abs(state)) > self.size
        
        if self.reward_type == 'sparse':
            # output binary reward for sparse
            reward = 0 if good_done else -1
        else:
            # output dense reward for other cases
            reward = 5*self.size if good_done else -10 if bad_done else -np.linalg.norm(state-goal)/200
            
        # return done flag as well
        return good_done or bad_done, reward

    def step(self, action, scale=4):
        # step through the env
        self.state += action/scale
        
        # obtain reward and done flag
        done, reward = self.reward_func(self.state, self.goal)
        
        # return update
        return np.copy(self.state/self.size), reward, done
    
    def render(self):
        # render state of env
        print("\rstate :", np.array_str(self.state), 
              "goal :", np.array_str(self.goal), end=' '*10)
              
'''
        
def main():

    reward_type='sparse'
    env=gym.make('FetchPickAndPlace-v1')

    # initialize DDPG agent
    agent = DDPGAgent(actor_learning_rate=0.0001, 
                      critic_learning_rate=0.0001, tau=0.1)
    variance = 5
    
    # use hindsight experience replay or not
    use_her = True
    
    # variables for network training
    num_epochs = 400
    num_episodes = 40
    optimization_steps = 40
    episode_length = 50
    
    # implement K-future strategy for HER
    K = 4
    '''
    the ratio between HER replays and regular replays (e.g. K = 4 -> 4 times
            as many HER replays as regular replays are used)
            
    '''

    # initialize buffers for tracking progress
    a_losses = []
    c_losses = []
    ep_mean_r = []
    success_rate = []

    # initialize buffers for episode experience
    ep_experience = Episode_experience()
    ep_experience_her = Episode_experience()

    # flags for training
    train = True

    if train:
        # time the performance of network
        total_step = 0
        start = time.clock()

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
                    state=state.reshape(1,25)
                    goal=goal.reshape(1,3)
                    action = agent.choose_action(state, goal, variance)      ########@$%@%$#@$#@%@

                    # execute action in env
                    obs, reward, done, info = env.step(action)
                    next_state=obs['observation'].reshape(1,25)
                    
                    # track reward and add regular experience
                    ep_total_r += reward
                    ep_experience.add(state, action, reward, next_state, done, goal)
                    state = next_state

                    # add experience using her
                    if total_step % 100 == 0 or done:
                        if use_her: 
                            # add additional experience for each time step
                            for t in range(len(ep_experience.memory)):
                                # get K future states per time step
                                ################
                                # Can we improve the K-future strategy?
                                ################
                                for _ in range(K):
                                    # get random future t
                                    future = np.random.randint(t, len(ep_experience.memory))

                                    # get new goal at t_future
                                    goal_ = ep_experience.memory[future][3][0,3:6]  ##########################
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
                        a_loss, c_loss = agent.replay(optimization_steps)
                        a_losses += [a_loss]
                        c_losses += [c_loss]
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
        
        # plot the performance stats
        plt.figure()
        plt.plot(success_rate)
        plt.title('Success Rates')
        plt.show()
        
        plt.figure()
        plt.plot(a_losses)
        plt.title('Actor Losses')
        plt.show()
        
        plt.figure()
        plt.plot(ep_mean_r)
        plt.title('Mean Episode Rewards')
        plt.show()
        
    # perform test inference
    agent.saver.save(agent.sess, 'checkpoints/fetchpickandplace.ckpt')
    # load saved model
    agent.saver.restore(agent.sess, 'checkpoints/fetchpickandplace.ckpt')

    # evaluate network for 5 episodes
    for _ in range(5):
        r = 0
        z=env.reset()
        state, goal = z['observation'],z['desired_goal'] 
        
        # run through the episode
        for _ in range(50):
            env.render()

            state=state.reshape(1,25)
            goal=goal.reshape(1,3)
            action = agent.choose_action(state, goal, 0)
            obs, reward, done, _ = env.step(action)
            r += reward
            next_state=obs['observation']
            state = next_state
            time.sleep(0.04)

            # render the final result
            if done:
                env.render()
                break
        print("reward : %06.2f"%r, " success :", reward==0)


if __name__=='__main__':
    main()
