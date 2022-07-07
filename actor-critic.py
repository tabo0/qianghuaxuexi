import random

import gym
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn import preprocessing
from torch.distributions import Categorical
from memory_buffer import Memory
from model import Actor, Critic, ActorCriticNetwork

GAMMA = 0.95
# 奖励折扣因子
LR = 0.01
# 学习率

EPISODE = 3000
# 生成多少个episode
STEP = 3000
# 一个episode里面最多多少步
TEST = 10


def readData():
    entry_info = np.load('./data/entry_info.npy',allow_pickle=True).item()
    worker_answer = np.load('./data/worker_answer2.npy',allow_pickle=True)
    project_info = np.load('./data/project_info.npy',allow_pickle=True).item()
    train_df=pd.DataFrame.from_records(worker_answer)
    train_df=train_df[0:1000]
    le = preprocessing.LabelEncoder()

    label_object = {}
    categorical_columns = train_df.columns.to_list()
    categorical_columns=categorical_columns[:-1]
    for col in categorical_columns:
        labelencoder = preprocessing.LabelEncoder()
        labelencoder.fit(train_df[col])
        train_df[col] = labelencoder.fit_transform(train_df[col])
        label_object[col] = labelencoder
    train_df_sort = train_df.sort_values(by=['entry_created_at'], ascending=[True])
    return train_df_sort
def update(model,state,next_state,reward,worker,actor_optimizer,critic_optimizer):
    critic_criterion = nn.MSELoss()
    act_, crit_ = model(state,worker)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        _, crti_2 = model(next_state.detach(),worker)

    REWARD = reward + 0.95 * crti_2

    loss = loss_fn(crit_, REWARD.detach())
    critic_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #losses.append(loss.item())
    critic_optimizer.step()
    act_=act_.float()
    act_.requires_grad=True
    loss1=loss_fn(act_.float(),next_state[1].float())
    actor_optimizer.zero_grad()
    loss1.backward(retain_graph=True)
    actor_optimizer.step()

def main():
    train_df=readData()
    oldProject=np.zeros(2501)
    memory = Memory(50000)
    model=ActorCriticNetwork()
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.01)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=0.01)
    for e in range(10):
        print("e:" ,e)
        cnt=0
        for index in range(100):
            state = 1
            state=torch.tensor([1,state],dtype=torch.long)
            reward=1
            next_state=torch.tensor([1,2],dtype=torch.long)
            update(model,state,next_state,reward,1,actor_optimizer,critic_optimizer)
            memory.push(state, reward, next_state)
        # test
        with torch.no_grad():
            action, crti_2 = model(state.detach(),1)
            print(action,"  ",crti_2)
            if(action==2):
                print("yes")
if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')