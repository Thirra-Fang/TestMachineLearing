#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: ThirraFang
LastEditTime: 2025-06-09 10:54:57
@Discription: 机器学习课件，gym经典控制杆不倒问题。
@Environment: python 3.7.7
@StudentTestEnvironment:python 3.7.9;pip 20.1.1;setuptools 47.1.0;gym 0.20.0;torch 1.13.1+cu117;matplotlib 3.5.3;seaborn 0.12.2;pyglet 2.0dev23;PyOpenGL 3.1.9;cuda 11.7.0
'''
import sys, os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path


import gym
import torch
import datetime

import numpy as np
#if not hasattr(np,'bool8'):
#    np.bool8 = np.bool

from common.utils import save_results, make_dir
from common.plot import plot_rewards
from agent import DQN
from agent import DDQN

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

class DQNConfig:
    def __init__(self):
        self.algo = "DDQN"  # name of algo
        self.env = 'CartPole-v1'
        self.result_path = curr_path+"\\outputs\\" + self.env + \
            '\\'+curr_time+'\\results\\'  # path to save results
        self.model_path = curr_path+"\\outputs\\" + self.env + \
            '\\'+curr_time+'\\models\\'  # path to save models
        self.train_eps = 300  # max trainng episodes
        self.end_reward = 450 #当十步平均reward大于这个值时结束训练避免后期奖励下降
        self.eval_eps = 50 # number of episodes for evaluating
        self.gamma = 0.95
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 100000  # capacity of Replay Memory
        self.batch_size = 64
        self.target_update = 5 # update frequency of target net，目标网络更新间隔
        self.DDQN_turn_target_and_policy = 100 #DDQN算法下适用，切换训练网络的间隔
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu
            #"cuda"
        self.hidden_dim = 256  # hidden size of net
        self.isdemo = False
        self.demo_path = curr_path+"\\outputs\\" + self.env + \
            '\\'+"20250609-152653"+'\\models\\'
        #''
        
def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    #env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if cfg.algo == "DQN":
        agent = DQN(state_dim,action_dim,cfg)
    elif cfg.algo == "DDQN":
        agent = DDQN(state_dim,action_dim,cfg)
    return env,agent
    
def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while True:
            #env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        if (i_ep+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep+1)%10 == 0:
            print('Episode:{}/{}, Reward:{}'.format(i_ep+1, cfg.train_eps, ep_reward))
            if ep_reward > 400:
                break
        rewards.append(ep_reward)
        # save ma rewards
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    if cfg.isdemo == True:
        eval_eps = 5
        display = True
    else:
        eval_eps = cfg.eval_eps
        display = False
    rewards = []  
    ma_rewards = [] # moving average rewards
    for i_ep in range(eval_eps):
        ep_reward = 0  # reward per episode
        state = env.reset()  
        while True:
            if display:
                env.render()
            action = agent.predict(state) 
            next_state, reward, done, _ = env.step(action)  
            state = next_state  
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%1 == 0:
            print(f"Episode:{i_ep+1}/{eval_eps}, reward:{ep_reward:.1f}")
    env.close()
    print('Complete evaling！')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = DQNConfig()
    if cfg.isdemo == False:
        # train
        env,agent = env_agent_config(cfg,seed=1)
        rewards, ma_rewards = train(cfg, env, agent)
        make_dir(cfg.result_path, cfg.model_path)
        agent.save(path=cfg.model_path)
        save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
        plot_rewards(rewards, ma_rewards, tag="train",env=cfg.env,
                     algo=cfg.algo, path=cfg.result_path)
    
    
        # eval
        env, agent = env_agent_config(cfg, seed=10)
        agent.load(path=cfg.model_path)
        rewards, ma_rewards = eval(cfg, env, agent)
        save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
        plot_rewards(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    else:
        # demo
        env, agent = env_agent_config(cfg, seed=3407)
        agent.load(path = cfg.demo_path)
        eval(cfg, env, agent)


