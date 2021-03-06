# -*- coding: utf-8 -*-

import time
import torch
from multiprocessing import Pool

from ENV import KYenv
from controller_vbc import VDN_MAC
from q_learner_vdn_vbc import QLearner
from run import run
from parallel_run_kit import ParallelRun

env = KYenv(4, 24, 500)

filename = 't1.pth'
cuda_flag = True

controller = VDN_MAC(n_actions = 6 * 5, input_shape = 11)
controller.save('state_dicts/' + filename)
learner = QLearner(controller)
if cuda_flag:
    learner.cuda()

test_controller = VDN_MAC(n_actions = 6 * 5, input_shape = 11)
test_controller.load('state_dicts/' + filename)
for par in test_controller.env_blender.parameters():
    par.requires_grad = False
for par in test_controller.agent.parameters():
    par.requires_grad = False

epi_length = 0
epi_length_step = 100
batch_size = 8

r_history = []

t1 = time.time()

for episode in range(100): # 1000
    print(episode)
    if episode % 400 == 0:
        epi_length += 400
        print(epi_length)
        runner = ParallelRun(env, test_controller, epi_length, test_mode = False)

    #batch = [run(env, test_controller, epi_length, test_mode = False) for i in range(batch_size)]
    batch = runner.run(batch_size)
    for key in batch[0].keys():
        batch[0][key] = torch.stack([batch[i][key] for i in range(batch_size)], dim=0)
        if cuda_flag:
            batch[0][key] = batch[0][key].cuda()
    batch = batch[0]

    learner.train(batch)
    
    if episode % 10 == 9:
        learner._update_targets()

    controller.save('state_dicts/' + filename)
    test_controller.load('state_dicts/' + filename)

    test = run(env, test_controller, epi_length, test_mode = True)
    r_history.append(test.detach().item())

    torch.save(r_history, "rhistory")

t2 = time.time()
print('time:', t2 - t1)

print("End")