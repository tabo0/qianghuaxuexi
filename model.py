import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_dim, state_dim*2)
        self.fc2 = nn.Linear(state_dim*2, action_dim)
        self.worker=nn.Embedding(num_embeddings=1807, embedding_dim=10)
        self.project = nn.Embedding(num_embeddings=2490, embedding_dim=10)
        # self.prod_embed=prod_embed
        # self.worker_embed=worker_embed
    def forward(self, x):
        workerEmb=self.worker(x[0])
        projectEmb=self.project(x[1])
        x=torch.cat([workerEmb,projectEmb], axis=-1)
        x = F.relu(self.fc1(x))
        the_weights = self.fc2(x)
        the_weights = the_weights.view(-1, 1, 10)
        project= [self.project(torch.tensor(i,dtype=torch.long)).reshape(-1,10) for i in range(1,2490)]
        project=torch.cat(project,axis=0)
        product_trans = torch.transpose(project, 0, 1)
        # output the index with maximum weight
        p=torch.matmul(the_weights, product_trans)
        argmax = torch.argmax(p, dim=2)
        return argmax+1
    # PGNetwork的作用是输入某一时刻的state向量，输出是各个action被采纳的概率
    # 和policy gradient中的Policy一样
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim, state_dim*2)
        self.fc2 = nn.Linear(state_dim*2, 1)
        self.worker=nn.Embedding(num_embeddings=1807, embedding_dim=10)
        self.project = nn.Embedding(num_embeddings=2490, embedding_dim=10)

    def forward(self, state,action):
        workerEmb=self.worker(state[0])
        projectEmb=self.project(state[1])
        state=torch.cat([workerEmb,projectEmb], axis=-1)
        workerEmb=self.worker(action[0])
        projectEmb=self.project(action[1])
        action=torch.cat([workerEmb,projectEmb], axis=-1)
        x=torch.cat([state,action], axis=-1)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        '''
        actor_model: pass Actor Network
        critic_model: pass Critic Network
        '''

        self.actor = Actor(20,10)
        self.critic =Critic(40)

    def forward(self, state,worker):
        actor = self.actor(state)
        next_state=torch.tensor([worker,actor],dtype=torch.long)
        critic = self.critic(state, next_state)

        return actor, critic