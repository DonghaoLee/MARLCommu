# -*- coding: utf-8 -*-
import torch
import numpy as np

def parabolic_antenna(origin, target, pos):
    distance = np.sqrt(((origin - pos) ** 2).sum())
    costheta = ((target - origin) ** 2 + (pos - origin) ** 2 - (target - pos) ** 2).sum() /\
               (2 * np.sqrt(((target - origin) ** 2).sum() * ((pos - origin) ** 2).sum()))
    if costheta > 1:
        costheta = torch.tensor(1.)
    if costheta < -1:
        costheta = torch.tensor(-1.)
    phi = np.arccos(costheta)
    gainDb = -min(20, 12*(phi / np.pi * 6)**2)
    gain = 10. ** (gainDb / 10.)
    power = gain * (10. / distance) ** 2.
    power = min(power, torch.tensor(500.))
    return power

class Env():
    def __init__(self, border, enbs, n_ues, noise):
        self.border = border
        self.enbs_pw = enbs[:, 0]
        self.enbs_pos = enbs[:, 1:]
        self.n_ues = n_ues
        self.n_enbs = enbs.shape[0]
        self.noise = noise
        self.reset()

        self.patience_decay = 0.9
        self.rand_step_length = 0.0 # 2.0

    def reset(self):
        self.time = 0
        self.ues_pos = torch.rand(self.n_ues, 2) * self.border
        self.MA_rate = torch.zeros(self.n_ues)
        self.inds = torch.tensor([[0, 1, 2, 3, 4] for _ in range(self.n_enbs)])

        obs = []
        self.get_global_obs()
        for i in range(self.n_enbs):
            obs.append(self.get_agent_obs(i))
        obs = torch.stack(obs, dim=0)
        return obs
    
    def get_global_obs(self):
        self.global_observation = torch.stack([self.ues_pos[:, 0] / self.border[0], 
                                               self.ues_pos[:, 1] / self.border[1], 
                                               self.MA_rate], dim=1)
        return self.global_observation

    def get_agent_obs(self, enb_i):
        dist = ((self.global_observation[:, 0:2] * self.border - self.enbs_pos[enb_i]) ** 2).sum(1)
        inds = torch.argsort(dist)[:5].sort().values
        self.inds[enb_i] = inds
        obs = self.global_observation[inds]
        return obs

    def step(self, actions):
        # actions should be a torch tensor, with shape as [n_agents]
        # actions = actions.view(-1)
        self.MA_rate *= self.patience_decay
        plan = [self.inds[i][actions[i]]if actions[i]<5 else -1 for i in range(self.n_enbs)] # the i_th enb shoot at the plan[i]_th ue
        for i in range(self.n_enbs):
            total = self.noise
            if plan[i]!= -1:
                for j in range(self.n_enbs):
                    if plan[j] != -1:
                        signal = parabolic_antenna(self.enbs_pos[j], self.ues_pos[plan[j]], self.ues_pos[plan[i]]) * self.enbs_pw[j]
                        total += signal
                        if i == j:
                            main_signal = signal
                self.MA_rate[plan[i]] += (1 - self.patience_decay) * torch.log(1 + main_signal / (total - main_signal))

        #reward = torch.log(1 + self.MA_rate).mean()
        reward, _ = torch.stack([self.MA_rate, 3 * torch.ones_like(self.MA_rate)]).min(dim = 0)
        reward = torch.exp(3. * reward.mean())

        # random walk
        #self.ues_pos += (torch.rand(self.n_ues, 2) - 0.5) * self.rand_step_length
        #flag_mat1 = self.ues_pos > self.border
        #flag_mat2 = self.ues_pos < 0
        #self.ues_pos = self.ues_pos - 2 * flag_mat2 * self.ues_pos + 2 * flag_mat1 * (self.border - self.ues_pos)

        # rebuild obs
        obs = []
        self.get_global_obs()
        for i in range(self.n_enbs):
            obs.append(self.get_agent_obs(i))
        obs = torch.stack(obs, dim=0)
        
        return obs, reward