# -*- coding: utf-8 -*-

import torch
from env import Env
from controller_vbc import VDN_MAC
from q_learner_vdn_vbc import QLearner
from run import run

env = Env(border=torch.Tensor([40, 40]), 
          enbs = torch.Tensor([[10, 10, 10], 
                               [10, 30, 10], 
                               [10, 10, 30],
                               [10, 30, 30]]), 
          n_ues = 10, 
          noise = 5
          )

controller = VDN_MAC()

learner = QLearner(controller)

epi_length = 0
epi_length_step = 100
batch_size = 8

r_history = []

for episode in range(400): # 1000
    if episode % 400 == 0:
        epi_length += 400
        print(epi_length)
    batch = [run(env, controller, epi_length, test_mode = False) for i in range(batch_size)]
    for key in batch[0].keys():
        batch[0][key] = torch.stack([batch[i][key] for i in range(batch_size)], dim=0)
    batch = batch[0]
    learner.train(batch)
    test = run(env, controller, epi_length, test_mode = True)
    r_history.append(test.detach().item())
    torch.save(r_history, "rhistory")

    if episode % 100 == 0:
        run(env, controller, epi_length, test_mode = True, flag = True)

print("End")