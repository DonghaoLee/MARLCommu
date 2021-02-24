import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from utils.comm_utils import *

def parabolic_antenna(origin, target, pos):
    costheta = ((target - origin) ** 2 + (pos - origin) ** 2 - (target - pos) ** 2).sum() / \
               (2 * np.sqrt(((target - origin) ** 2).sum() * ((pos - origin) ** 2).sum()))
    if costheta > 1:
        costheta = torch.tensor(1.)
    if costheta < -1:
        costheta = torch.tensor(-1.)
    phi = np.arccos(costheta)
    gainDb = -min(20, 12*(phi / np.pi * 6)**2)
    gain = 10. ** (gainDb / 10.)
    return gain


def state_normalization(states,state_max,state_min, Q):

    stages = np.array(list(range(Q + 1)))/ Q
    normalized_stages = stages - 1/2
    diff = state_max - state_min
    state_stages = stages * diff + state_min

    normalized_states = copy.deepcopy(states)
    for i in range(len(states)):

        pos = np.where(state_stages <= states[i])[0][-1]
        normalized_states[i] = normalized_stages[pos]


    return normalized_states




class Env():
    def __init__(self,apNum,ueNum,boarder):

        self.apNum = apNum
        self.ueNum = ueNum
        self.boarder = boarder
        self.alphaI = 0.05
        self.alphaR = 0.01


        self.reset()

        flag = True

        while flag:
            flag = False
            for i in range(self.apNum):
                if len(self.apUE[i]) == 0:
                    self.reset()
                    flag = True




    def reset(self):

        self.time = 0
        self.appos = APgen(self.boarder,self.apNum,35)
        self.uepos = UEgen(self.appos,self.boarder,self.ueNum,10)
        self.dists = getDist(self.appos,self.uepos)
        self.power = np.ones(self.dists.shape) * 0.01
        self.passGain = DSPloss(self.dists)
        self.shadowing = log_norm_shadowing(self.apNum,self.ueNum,7)
        self.apID = np.argmax(self.passGain + self.shadowing,0)
        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        self.apUE = dict()
        for i in range(self.apNum):
            self.apUE[i] = np.where(self.apID == i)[0]
        self.achRate = getAchRate(self.passGain + self.fading + self.shadowing,self.power,-174,self.apID)
        self.weight = 1/self.achRate
        self.max_weight = self.weight.max()
        self.min_weight = self.weight.min()
        self.normalized_weight = state_normalization(self.weight,self.max_weight,self.min_weight,20)
        self.infPower = getInfpower(self.passGain + self.fading + self.shadowing, self.power , self.apID)
        self.SINR = getSINR(self.passGain + self.fading + self.shadowing, self.infPower, -174, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
        self.max_SINR = self.SINRdb.max()
        self.min_SINR = self.SINRdb.min()
        self.normalized_SINR = state_normalization(self.SINRdb,self.max_SINR,self.min_SINR,20)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = self.achRate
        self.reward = 0
        self.actions = torch.tensor(np.zeros((self.apNum*2,)),dtype=torch.float32)

        #return self.struct_obv()
        return self.global_obv()


    def global_obv(self):



        global_ob = torch.cat((torch.tensor(self.normalized_weight,dtype=torch.float32).view(-1),
                               torch.tensor(self.normalized_SINR,dtype=torch.float32).view(-1),
                               self.actions,
                               torch.tensor([self.reward],dtype=torch.float32)),0)

        return global_ob

    def ap_obv(self, apID):

        ue_id = self.apUE[apID]
        if len(ue_id) < 5:
            ue_weights = np.zeros((5,))
            ue_weights[:len(ue_id)] = self.weight[ue_id]
            ue_SINR = np.zeros((5,)) - 60
            ue_SINR[:len(ue_id)] = self.SINRdb[ue_id]
        else:
            ue_pf = self.pf[ue_id]
            ue_id = np.argsort(ue_pf)[::-1][:5]
            ue_weights = self.weight[ue_id]
            ue_SINR = self.SINRdb[ue_id]

        ue_weights = state_normalization(ue_weights,self.max_weight,self.min_weight,20)
        ue_SINR = state_normalization(ue_SINR,self.max_SINR,self.min_SINR,20)

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
            actions = torch.stack([actions % 5, actions // 5], dim=1)

        for i in range(self.appos.shape[0]):
            ues.append(self.ap_action(i,actions[int(i),0], actions[int(i),1]))


        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        instRate = getAchRate(self.passGain + self.fading + self.shadowing,self.power,-174, self.apID)
        self.achRate = (1 - self.alphaR) * self.achRate + self.alphaR * instRate
        self.weight = 1/self.achRate
        if self.weight.max() > self.max_weight:
            self.max_weight = self.weight.max()
        if self.weight.min() < self.min_weight:
            self.min_weight = self.weight.min()
        self.normalized_weight = state_normalization(self.weight,self.max_weight,self.min_weight,20)
        self.infPower = (1 - self.alphaI) * self.infPower + \
                        self.alphaI * getInfpower(self.passGain + self.fading + self.shadowing, self.power, self.apID)
        self.SINR = getSINR(self.passGain + self.fading + self.shadowing, self.infPower, -174, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
        if self.SINRdb.max() > self.max_SINR:
            self.max_SINR = self.SINRdb.max()
        if self.SINRdb.min() < self.min_SINR:
            self.min_SINR = self.SINRdb.min()
        self.normalized_SINR = state_normalization(self.SINRdb, self.max_SINR, self.min_SINR, 20)
        self.pf = self.weight * np.log2(1 + self.SINR)

        reward = 0
        lambda_rew = 0.8

        for i in range(len(ues)):

            if ues[i] != None:
                reward += self.weight[ues[i]] ** lambda_rew * instRate[ues[i]]

        if reward == 0:
            reward = - np.max(self.pf)

        self.reward = reward
        self.actions = torch.tensor(np.reshape(actions,(self.apNum*2,)),dtype=torch.float32)

        self.accumulated_rate += instRate
        self.time += 1

        #return self.struct_obv(), torch.tensor(reward, dtype=torch.float32)
        return self.global_obv(), torch.tensor(reward,dtype=torch.float32)

    def ap_action(self, apID, ueID, power):

        ue_id = self.apUE[apID]

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
        return np.sum(self.accumulated_rate)/self.time

    def get_R5per(self):

        pos = round(len(self.accumulated_rate) * 0.95)
        sorted_rate = np.sort(self.accumulated_rate)[::-1]

        return sorted_rate[pos]/self.time

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
    apUE = env.apUE

    for i in range(2000):
        best_ue = []
        for i in range(4):
            ueID = apUE[i]
            uePF = env.pf[ueID]
            sorted_ue = np.argsort(uePF)
            best_ue.append(sorted_ue[0])
        actions = np.array([[best_ue[0],0.01],[best_ue[1],0.01],[best_ue[2],0.01],[best_ue[3],0.01]])
        reward = env.step(actions)

    print(env.get_R5per(),env.get_Rsum())
    apID = env.apID
    env.reset()
    for i in range(2000):

        pos = i % len(apID)
        ap = apID[pos]
        local_id = np.where(apUE[ap] == pos)[0][0]

        actions = np.array([[0,0.0],[0,0.0],[0,0.0],[0,0.0]])
        actions[ap,0] = local_id
        actions[ap,1] = 0.01

        reward = env.step(actions)
    #
    print(env.get_R5per(),env.get_Rsum())





