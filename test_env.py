# -*- coding: utf-8 -*-

import torch
from ENV import KYenv, DHenv

def env_test(env, n, m, test_mode='random action'):
    out_l = []
    obs = env.reset()
    print('obs.shape: ', obs.shape)
    l = []
    for _ in range(n): # how many episodes
        r = 0
        for _ in range(m): # how long for one episode
                # obs, reward = env1.step(torch.tensor([0, 0, 0, 0]))
            if test_mode == 'random action':
                obs, reward = env1.step(torch.randint(6, (5,)))
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

# env2 = 

env_test(env1, 20, 50, test_mode='random action')