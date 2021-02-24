# -*- coding: utf-8 -*-

import torch
import numpy as np
from ENV import KYenv, DHenv

def DHenv_random_action(obs):
    return torch.randint(6, (4,))

def KYenv_constant_action(obs):
    return np.array([[3,9],[1,5],[0,3],[2,10]])

def KYenv_random_action(obs):
    #ues = torch.randint(5, (4,))
    #pows = torch.randint(6, (4,))
    #return torch.stack([ues, pows], dim=1)
    return torch.randint(30, (4,))

def env_test(env, n, m, action_func = None):
    assert action_func is not None
    out_l = []
    obs = env.reset()
    print('obs.shape: ', obs.shape)
    l = []
    for _ in range(n): # how many episodes
        r = 0
        obs = env.reset()
        for _ in range(m): # how long for one episode
            action = action_func(obs)
            obs, reward = env.step(action)
            r += 0.02 * (reward - r)
        l.append(r)
    l = torch.tensor(l)
    print('Reward: ', l)
    print('mean: ', l.mean())
    print('var: ', l.var())


env1 = DHenv(
             border=torch.Tensor([40, 40]), 
             enbs = torch.Tensor([[10, 10, 10], 
                                 [10, 30, 10], 
                                 [10, 10, 30],
                                 [10, 30, 30]]), 
             n_ues = 10, 
             noise = 0.1
            )

env2 = KYenv(4, 24, 500)

env_test(env1, 20, 50, action_func=DHenv_random_action)

env_test(env2, 20, 500, action_func=KYenv_random_action)