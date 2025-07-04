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
from common.plot import plot_q_values
from agent import DQN, DDQN,REINFORCE

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

class TestConfig:
    def __init__(self):
        self.algo = "DDQN"  # name of algo
        self.env = 'CartPole-v1'
        self.result_path = curr_path+"\\outputs\\" + self.env + \
            '\\'+self.algo+'_'+curr_time+'\\results\\'  # path to save results
        self.model_path = curr_path+"\\outputs\\" + self.env + \
            '\\'+self.algo+'_'+curr_time+'\\models\\'  # path to save models
        self.train_eps = 5000  # max trainng episodes
        self.end_reward = 475 #当ma_reward大于这个值时结束训练
        self.eval_eps = 50 # number of episodes for evaluating
        self.gamma = 0.95
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 100000  # capacity of Replay Memory
        self.batch_size = 64
        self.target_update = 5 # update frequency of target net，DQN算法下适用，目标网络更新间隔
        self.DDQN_turn_target_and_policy = 5 #DDQN算法下适用，切换训练网络的间隔
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu
            #"cuda"
        self.hidden_dim = 256  # hidden size of net
        self.isdemo = False
        self.demo_path = curr_path+"\\outputs\\" + self.env + \
            '\\'+"DDQN_20250610-003217"+'\\models\\'
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
    elif cfg.algo == "REINFORCE":
        agent = REINFORCE(state_dim, action_dim, cfg)
    else:
        agent = DQN(state_dim, action_dim, cfg)
    return env,agent
    
def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    aver_q_values = []
    aver_pre_q_values = []
    count100 = 0
    count200 = 0
    count300 = 0
    count400 = 0
    count500 = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_aver_q = 0
        ep_aver_pre_q = 0
        i_step = 0
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
        }
        while True:
            i_step+=1
            #env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if cfg.algo == "DQN" or cfg.algo == "DDQN":
                agent.memory.push(state, action, reward, next_state, done)
            elif cfg.algo == "REINFORCE":
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['rewards'].append(reward)
            state = next_state

            if cfg.algo == "DQN" or cfg.algo == "DDQN":
                #获取每次迭代q平均输出值和预测值
                average_q_values, average_predict_q_values = agent.update()
                ep_aver_q += average_q_values
                ep_aver_pre_q += average_predict_q_values
            if done:
                break

        if cfg.algo == "REINFORCE":
            #REINFORCE每轮迭代一次
            agent.update(transition_dict)

        #保存并输出q输出值和预测值
        ep_aver_q /= i_step
        ep_aver_pre_q /= i_step
        aver_q_values.append(ep_aver_q)
        aver_pre_q_values.append(ep_aver_pre_q)

        #拷贝参数至目标值网络
        if cfg.algo == "DQN" and (i_ep+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        #交换当前值网络与目标值网络的参数
        if cfg.algo == "DDQN" and (i_ep+1) % cfg.DDQN_turn_target_and_policy == 0:
            agent.inverse_policy_target()

        if (i_ep+1)%10 == 0:
            print('Episode:{}/{}, Reward:{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ep_reward >= 100:
            count100+=1
        if ep_reward >= 200:
            count200+=1
        if ep_reward >= 300:
            count300+=1
        if ep_reward >= 400:
            count400+=1
        if ep_reward >= 500:
            count500+=1
        # save ma rewards
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if ma_rewards[-1] > cfg.end_reward:
            break
    print('Complete training！')
    print('100以上的回合数：',count100)
    print('200以上的回合数：',count200)
    print('300以上的回合数：',count300)
    print('400以上的回合数：',count400)
    print('500以上的回合数：',count500)
    return rewards, ma_rewards,aver_q_values, aver_pre_q_values

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
    cfg = TestConfig()
    if cfg.isdemo == False:
        # train
        env,agent = env_agent_config(cfg,seed=1)
        rewards, ma_rewards,aver_q_values, aver_pre_q_values = train(cfg, env, agent)
        make_dir(cfg.result_path, cfg.model_path)
        agent.save(path=cfg.model_path)
        save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
        plot_rewards(rewards, ma_rewards, tag="train",env=cfg.env,
                     algo=cfg.algo, path=cfg.result_path)
        plot_q_values(aver_q_values,aver_pre_q_values,env=cfg.env,algo=cfg.algo,
                      path=cfg.result_path)


    
    
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


