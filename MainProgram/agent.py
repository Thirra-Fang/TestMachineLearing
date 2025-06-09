#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2021-05-07 16:30:05
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''




import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from common.memory import ReplayBuffer
from common.model import MLP
from common.model import Actor, Critic


def safe_multinomial(probs, num_samples=1, min_prob=1e-10):
    """
    安全的多项式采样函数，处理各种边界情况

    :param probs: 概率张量，形状为 [batch_size, num_classes] 或 [num_classes]
    :param num_samples: 采样数量
    :param min_prob: 最小概率值，防止数值问题
    :return: 采样结果
    """
    # 确保是浮点类型
    probs = probs.float()

    # 1. 处理负值：将负值设为0
    probs = torch.clamp(probs, min=0.0)

    # 2. 处理NaN和Inf
    probs = torch.nan_to_num(probs, nan=min_prob, posinf=min_prob, neginf=min_prob)

    # 3. 添加最小概率值防止完全为0
    probs = probs + min_prob

    # 4. 归一化概率
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / probs_sum

    # 5. 确保没有无效值
    assert torch.all(probs >= 0), "Negative probabilities found after cleaning"
    assert not torch.any(torch.isnan(probs)), "NaN values found after cleaning"
    assert not torch.any(torch.isinf(probs)), "Inf values found after cleaning"

    #print(probs)
    # 6. 执行采样
    return torch.multinomial(probs, num_samples=num_samples).squeeze(-1)
class DQN:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子

        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)

        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): # copy params from policy net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)#迭代决策模型的参数所用的优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)
        

    def choose_action(self, state):
        '''选择动作,概率选择预测最优动作或者随机动作
        '''
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            action = random.randrange(self.action_dim)
        return action
    def predict(self,state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action
    def update(self):

        #如果记忆不足batch_size，退出
        if len(self.memory) < self.batch_size:
            return 0,0
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        '''转为张量
        例如tensor([[-4.5543e-02, -2.3910e-01,  1.8344e-02,  2.3158e-01],...,[-1.8615e-02, -2.3921e-01, -1.1791e-02,  2.3400e-01]])'''
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)

        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        '''torch.gather:对于a=torch.Tensor([[1,2],[3,4]]),那么a.gather(1,torch.Tensor([[0],[1]]))=torch.Tensor([[1],[3]])'''
        q_values = self.policy_net(state_batch).gather(
            dim=1, index=action_batch)  # 等价于self.forward
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states
        next_q_values = self.target_net(next_state_batch).max(
            1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,])
        # 计算 expected_q_value
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + \
            self.gamma * next_q_values * (1-done_batch)
        # self.loss = F.smooth_l1_loss(q_values,expected_q_values.unsqueeze(1)) # 计算 Huber loss
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算 均方误差loss
        #print(expected_q_values.shape)
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        loss.backward()
        # for param in self.policy_net.parameters():  # clip防止梯度爆炸
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型


        average_q_values = q_values.mean().item()
        average_predict_q_values = next_q_values.mean().item()
        #返回预测的q值和决策网络计算的q值，实验用
        return average_q_values, average_predict_q_values


    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


class DDQN:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.temp_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # copy params from policy net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 迭代决策模型的参数所用的优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):
        '''选择动作
        '''
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            action = random.randrange(self.action_dim)
        return action

    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self):

        # 如果记忆不足batch_size，退出
        if len(self.memory) < self.batch_size:
            return 0,0
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        '''转为张量
        例如tensor([[-4.5543e-02, -2.3910e-01,  1.8344e-02,  2.3158e-01],...,[-1.8615e-02, -2.3921e-01, -1.1791e-02,  2.3400e-01]])'''
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)

        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        '''torch.gather:对于a=torch.Tensor([[1,2],[3,4]]),那么a.gather(1,torch.Tensor([[0],[1]]))=torch.Tensor([[1],[3]])'''
        q_values = self.policy_net(state_batch).gather(
            dim=1, index=action_batch)  # 等价于self.forward
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states，列
        #DQN中：
        #next_q_values = self.target_net(next_state_batch).max(
        #    1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,])
        #print(next_q_values)
        #返回最大值列表
        #DDQN中：
        action_values = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            #返回最大值index,列
        next_q_values = self.target_net(next_state_batch).gather(dim=1, index=action_values).squeeze(1)
            #选取使policy_net(next_state_batch)的reward最大的action，并带入target_net，行

        # 计算 expected_q_value
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + \
                            self.gamma * next_q_values * (1 - done_batch)
        # self.loss = F.smooth_l1_loss(q_values,expected_q_values.unsqueeze(1)) # 计算 Huber loss
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算 均方误差loss
        # print(expected_q_values.shape)
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        loss.backward()
        # for param in self.policy_net.parameters():  # clip防止梯度爆炸
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型

        average_q_values = q_values.mean().item()
        average_predict_q_values = next_q_values.mean().item()
        #返回预测的q值和决策网络计算的q值，实验用
        return average_q_values, average_predict_q_values

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
    #交换目标网络和策略王略
    def inverse_policy_target(self):
        self.temp_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.load_state_dict(self.target_net.state_dict())
        self.target_net.load_state_dict(self.temp_net.state_dict())

class REINFORCE:
    def __init__(self,state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device
        self.policy_net = MLP(state_dim,action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=cfg.lr)  # 使用Adam优化器
        self.gamma = cfg.gamma  # 折扣因子


    def choose_action(self, state):
        '''选择动作,为保持和DQN和DDQN结构一致，这里同时保留了两个方法
        '''
        action = self.predict(state)
        return action
    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            act_prob = self.policy_net(state)
            action = safe_multinomial(act_prob,num_samples=1)[0].item()#根据动作概率选取动作

        return action

    #计算每一个step的ut
    def get_ut(self,reward_list,gamma=1.0):
        for i in range(len(reward_list) - 2, -1, -1):
            reward_list[i] += gamma * reward_list[i + 1]
        return np.array(reward_list)

    def update(self,transition_dict):
        state = torch.tensor([transition_dict['states']], device=self.device, dtype=torch.float32)
        action = torch.tensor([transition_dict['actions']], device=self.device, dtype=torch.int64)
        reward = torch.tensor([self.get_ut(transition_dict['rewards'],self.gamma)], device=self.device, dtype=torch.float32)

        action_prob = self.policy_net(state)

        #梯度上升
        #原文档paddle版本求交叉熵
        #log_prob = paddle.sum(-1.0 * paddle.log(act_prob) * paddle.nn.functional.one_hot(act, act_prob.shape[1]),axis=-1)
        log_prob = -torch.gather(torch.log(action_prob), 1, action.unsqueeze(-1)).squeeze(-1)
        loss = log_prob*reward

        loss = nn.MSELoss()(loss,torch.zeros_like(loss))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'checkpoint.pth'))





