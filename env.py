# -*- coding: utf-8 -*-
import torch
import numpy as np

def parabolic_antenna(origin, target, pos):
    '''
        a parabolic antenna at  origin  aims at  target
        return the gain in Db for the position  pos  
    '''
    costheta = ((target - origin) ** 2 + (pos - origin) ** 2 - (target - pos) ** 2).sum() /\
               (2 * np.sqrt(((target - origin) ** 2).sum() * ((pos - origin) ** 2).sum()))
    if costheta > 1:
        costheta = torch.tensor(1.)
    if costheta < -1:
        costheta = torch.tensor(-1.)
    phi = np.arccos(costheta)
    gainDb = -min(20, 12*(phi / np.pi * 6)**2)
    return gainDb

def pathloss(origin, pos):
    '''
        UE to macro BS - UE is inside a house
    '''
    L_ow = 15
    d0 = torch.tensor(10.)

    distance = torch.sqrt(((origin - pos) ** 2).sum())
    if distance < d0:
        distance = d0

    pl_db = 15.3 + 37.6 * torch.log10(distance) + L_ow

    return pl_db

def cal_sig(origin, target, pos, pw):
    '''
        ?
    '''
    db = parabolic_antenna(origin, target, pos) - pathloss(origin, pos)
    linear = 10 ** (0.1 * db)
    return pw * linear

class Env():
    def __init__(self, border, enbs, ues, noise, rw=0.0):
        self.border = border
        self.enbs_pw = enbs[:, 0]
        self.enbs_pos = enbs[:, 1:]
        self.n_ues = ues.shape[0]
        self.origin_ues_pos = ues
        self.n_enbs = enbs.shape[0]
        self.noise = noise
        self.rand_step_length = rw
        self.reset()

        self.patience_decay = 0.9

    def get_stacked_obs(self):
        obs = []
        self.get_global_obs()
        for i in range(self.n_enbs):
            obs.append(self.get_agent_obs(i))
        obs = torch.stack(obs, dim=0)
        return obs

    def reset(self):
        self.time = 0
        self.ues_pos = self.origin_ues_pos.clone() # torch.rand(self.n_ues, 2) * self.border
        self.MA_rate = torch.zeros(self.n_ues)
        self.inds = torch.tensor([[0, 1, 2, 3, 4] for _ in range(self.n_enbs)])

        obs = self.get_stacked_obs()
        return obs
    
    def get_global_obs(self):
        self.global_observation = torch.stack([torch.arange(self.n_ues),
                                               self.ues_pos[:, 0] / self.border[0], 
                                               self.ues_pos[:, 1] / self.border[1], 
                                               self.MA_rate], 
                                               dim=1)
        return self.global_observation[:, 1:]

    def get_agent_obs(self, enb_i):
        dist = ((self.global_observation[:, 1:3] * self.border - self.enbs_pos[enb_i]) ** 2).sum(1)
        inds = torch.argsort(dist)[:5].sort().values
        self.inds[enb_i] = inds
        obs = self.global_observation[inds]
        obs[:, 1:3] -= self.enbs_pos[enb_i] / self.border
        return obs

    def step(self, actions):
        # actions should be a torch tensor, with shape as [n_agents]
        # actions = actions.view(-1)
        self.MA_rate *= self.patience_decay
        plan = [self.inds[i][actions[i]]if actions[i]<5 else -1 for i in range(self.n_enbs)] # the i_th enb shoot at the plan[i]_th ue
        #print(plan)
        signal_list = [[] for _ in range(self.n_ues)]
        for i in range(self.n_enbs):
            total = self.noise
            if plan[i]!= -1:
                for j in range(self.n_enbs):
                    if plan[j] != -1:
                        signal = cal_sig(self.enbs_pos[j], self.ues_pos[plan[j]], self.ues_pos[plan[i]], self.enbs_pw[j])
                        #print(j, "->", i, ": ", signal)
                        total += signal
                        if i == j:
                            main_signal = signal
                signal_list[plan[i]].append((1 - self.patience_decay) * torch.log(1 + main_signal / (total - main_signal)))
                #print(main_signal, total - main_signal, self.noise)
        
        for i in range(self.n_ues):
            if len(signal_list[i]) != 0:
                self.MA_rate[i] += torch.max(torch.tensor(signal_list[i]))

        #reward = torch.log(1 + self.MA_rate).mean()
        reward, _ = torch.stack([self.MA_rate, 3 * torch.ones_like(self.MA_rate)]).min(dim = 0)
        reward = reward.sum()

        self.plan = plan

        # random walk
        self.ues_pos += (torch.rand(self.n_ues, 2) - 0.5) * self.rand_step_length
        flag_mat1 = self.ues_pos > self.border
        flag_mat2 = self.ues_pos < 0
        self.ues_pos = self.ues_pos - 2 * flag_mat2 * self.ues_pos + 2 * flag_mat1 * (self.border - self.ues_pos)

        # rebuild obs
        obs = self.get_stacked_obs()

        self.time += 1
        
        return obs, reward

    def draw(self, size_on=True, serve_on=True):
        # draw UEs position
        plt.figure(dpi=100, figsize=[10, 10])
        plt.xlim(0, self.border[0])
        plt.ylim(0, self.border[1])
        t_ues = self.ues_pos
        if size_on:
            t_size = self.MA_rate
        else:
            t_size = 5
        plt.plot(t_ues[:, 0], t_ues[:, 1], '.', markersize=t_size)
        if serve_on:
            for i in range(self.n_enbs):
                plt.plot()
        plt.show()