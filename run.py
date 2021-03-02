# -*- coding: utf-8 -*-

import torch

def run(env, agent, max_length, explore_epsilon=0.2, test_mode = False, flag = False):
    # if test mode is true, communication is limited, choose with fully greedy method, return accumulated reward
    # if test mode is false, full communication, choose with epsilon greedy, return batch

    obs = env.reset()
    agent.init_hidden(1)

    if (not test_mode) or flag:
        batch = {
                    "reward": [], 
                    "actions": [],
                    "obs": []
                }
    
    gamma = 0.98
    r = 0

    for t in range(max_length):
        action = agent.choose_action(obs.unsqueeze(0), test_mode = test_mode or flag)
        if (not test_mode):
            if torch.rand(1) < explore_epsilon:
                action = torch.randint(6, (1,))
        if (not test_mode) or flag:
            batch["obs"].append(obs)
            batch["actions"].append(action.detach())

        obs, reward = env.step(action)
        if (not test_mode) or flag:
            batch["reward"].append(reward)
        r += (1 - gamma) * (reward - r)
    
    #if not test_mode:
        #batch["obs"].append(obs)

    if test_mode:
        if flag:
            batch["reward"] = torch.stack(batch["reward"]) # shape = [t]
            batch["actions"] = torch.stack(batch["actions"]) # shape = [t, 4]
            batch["obs"] = torch.stack(batch["obs"]) # shape = [t, 4, 5, 4]
            torch.save(batch, 'batch')
        return r
    else:
        batch["reward"] = torch.stack(batch["reward"]) # shape = [t]
        batch["actions"] = torch.stack(batch["actions"]) # shape = [t, 4]
        batch["obs"] = torch.stack(batch["obs"]) # shape = [t, 4, 5, 4]
        return batch