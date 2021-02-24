import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.comm_utils import *

def parabolic_antenna(origin, target, pos):
    distance = np.sqrt(((origin - pos) ** 2).sum())
    costheta = ((target - origin) ** 2 + (pos - origin) ** 2 - (target - pos) ** 2).sum() / \
               (2 * np.sqrt(((target - origin) ** 2).sum() * ((pos - origin) ** 2).sum()))
    if costheta > 1:
        costheta = torch.tensor(1.)
    if costheta < -1:
        costheta = torch.tensor(-1.)
    phi = np.arccos(costheta)
    gainDb = -min(20, 12*(phi / np.pi * 6)**2)
    gain = 10. ** (gainDb / 10.)
    # power = gain * (10. / distance) ** 2.
    # power = min(power, torch.tensor(500.))
    return gain

class Env():
    def __init__(self,apNum,ueNum,boarder):

        self.time = 0
        self.appos = APgen(boarder,apNum,35)
        self.apNum = apNum
        self.uepos = UEgen(self.appos,boarder,ueNum,7)
        self.boarder = boarder
        self.ueNum = ueNum
        self.dists = getDist(self.appos,self.uepos)
        self.power = np.ones(self.dists.shape) * 0.01
        self.passGain = DSPloss(self.dists,shadowing_std=7)
        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        self.apID = np.argmax(self.passGain,0)
        self.apUE = dict()
        for i in range(apNum):
            self.apUE[i] = np.where(self.apID == i)[0]
        self.achRate = getAchRate(self.passGain + self.fading,self.power,-134)
        self.weight = 1/self.achRate
        self.infPower = getInfpower(self.passGain, self.power)
        self.SINR = getSINR(self.passGain, self.infPower, -134)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = [self.achRate]
        self.alphaI = 0.05
        self.alphaR = 0.01


    def reset(self):

        self.time = 0
        self.uepos = UEgen(self.appos,self.boarder,self.ueNum,7)
        self.dists = getDist(self.appos,self.uepos)
        self.power = np.ones(self.dists.shape) * 0.01
        self.passGain = DSPloss(self.dists,shadowing_std=7)
        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        self.apID = np.argmax(self.passGain,0)
        self.apUE = dict()
        for i in range(self.apID.shape[0]):
            self.apUE[i] = np.where(self.apID == i)[0]
        self.achRate = getAchRate(self.passGain + self.fading,self.power,-134)
        self.weight = 1/self.achRate
        self.infPower = getInfpower(self.passGain+self.fading, self.power)
        self.SINR = getSINR(self.passGain+self.fading, self.infPower, -134)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = [self.achRate]

        return self.struct_obv()


    def global_obv(self):

        return [self.weight, self.SINR]

    def ap_obv(self, apID):

        ue_id = self.apUE[apID]
        if len(ue_id) < 5:
            ue_weights = np.zeros((5,))
            ue_weights[:len(ue_id)] = self.weight[ue_id]
            ue_SINR = np.zeros((5,))
            ue_SINR[:len(ue_id)]  = self.SINR[ue_id]
        else:
            ue_pf = self.pf[ue_id]
            ue_id = np.argsort(ue_pf)[::-1][:5]
            ue_weights = self.weight[ue_id]
            ue_SINR = self.SINR[ue_id]

        return [ue_weights,ue_SINR]

    def struct_obv(self, ):
        struct_ob = []
        for i in range(self.apNum):
            struct_ob.append(torch.tensor(self.ap_obv(i), dtype=torch.float32).view(-1))
        struct_ob = torch.stack(struct_ob)
        return struct_ob

    def step(self, actions):

        ues = []

        # HERE IS A HEURISTIC TRANSFORM
        if len(actions.shape) == 1:
            actions = torch.stack([actions % 5, 0.01 * actions // 5], dim=1)

        for i in range(self.appos.shape[0]):
            ues.append(self.ap_action(i,actions[int(i),0], actions[int(i),1]))


        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        instRate = getAchRate(self.passGain + self.fading,self.power,-134)
        self.achRate = (1 - self.alphaR) * self.achRate + self.alphaR * instRate
        self.weight = 1/self.achRate
        self.infPower = (1 - self.alphaI) * self.infPower + self.alphaI * getInfpower(self.passGain + self.fading, self.power)
        self.SINR = getSINR(self.passGain + self.fading, self.infPower, -134)
        self.pf = self.weight * np.log2(1 + self.SINR)

        reward = 0
        lambda_rew = 0.8

        for i in range(len(ues)):

            if ues[i] != None:
                reward += self.weight[ues[i]] ** lambda_rew * instRate[ues[i]]

        if reward == 0:
            reward = - np.max(self.pf)

        self.accumulated_rate.append(self.achRate)

        return self.struct_obv(), torch.tensor(reward, dtype=torch.float32)

    def ap_action(self, apID, ueID, power):

        ue_id = self.apUE[apID]
        ue_pf = self.pf[ue_id]
        ue_pos = np.argsort(ue_pf)
        ue_id = ue_id[ue_pos]

        if ueID >= len(ue_id):
            ue = None
            self.power[apID,:] = 0
        else:
            ue = ue_id[int(ueID)]
            for i in ue_id:
                self.power[apID,i] = power * parabolic_antenna(self.appos[apID,:],\
                                                               self.uepos[ue,:],\
                                                               self.uepos[i,:])

        return ue

    def get_Rsum(self):
        return np.sum(self.achRate)

    def get_R5per(self):

        pos = round(len(self.achRate) * 0.95)
        sorted_rate = np.sort(self.achRate)[::-1]

        return sorted_rate[pos]

    def plot_scheduling(self):
        # use power for size
        s = self.power[:,0] * 20
        cm = np.linspace(1, self.apNum, self.apNum)

        plt.figure(figsize=(5,5))
        plt.scatter(self.appos[:,0], self.appos[:,1],marker="x", s = s, c = cm)

        plt.scatter(self.uepos[:,0],self.uepos[:,1],marker="o", s = self.achRate * 20, c = self.apID)
        plt.xlim([0,self.boarder])
        plt.ylim([0,self.boarder])
        plt.show()




if __name__ == "__main__":

    env = Env(4,24,500)
    for i in range(1000):
        actions = np.array([[0,0.01],[0,0.01],[0,0.01],[0,0.01]])
        reward = env.step(actions)

    print(env.get_R5per(),env.get_Rsum())
    apUE = env.apUE
    uenum = []
    for i in range(4):
        uenum.append(len(apUE[i]))
    env.reset()
    for i in range(1000):
        act1 = i % uenum[0]
        act2 = i % uenum[1]
        act3 = i % uenum[2]
        act4 = i % uenum[3]

        actions = np.array([[act1,0.01],[act2,0.01],[act3,0.01],[act4,0.01]])

        reward = env.step(actions)
    #
    print(env.get_R5per(),env.get_Rsum())





