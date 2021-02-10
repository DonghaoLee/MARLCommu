import numpy as np
import matplotlib.pyplot as plt

from utils.comm_utils import *

class Env():
    def __init__(self,apNum,ueNum,boarder):

        self.time = 0
        self.appos = APgen(boarder,apNum,35)
        self.apnum = apNum
        self.uepos = UEgen(self.appos,boarder,ueNum,7)
        self.boarder = boarder
        self.ueNum = ueNum
        self.dists = getDist(self.appos,self.uepos)
        self.power = np.ones(self.dists.shape) * 10
        self.passGain = DSPloss(self.dists,shadowing_std=3)
        self.apID = np.argmax(self.passGain,0)
        self.apUE = dict()
        for i in range(apNum):
            self.apUE[i] = np.where(self.apID == i)[0]
        self.achRate = getAchRate(self.passGain,self.power,-134)
        self.weight = 1/self.achRate
        self.infPower = getInfpower(self.passGain, self.power)
        self.SINR = getSINR(self.passGain, self.infPower, -134)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = [self.achRate]
        self.alphaI = 0.5
        self.alphaR = 0.7


    def reset(self):

        self.time = 0
        self.uepos = UEgen(self.appos,self.oarder,self.ueNum,10)
        self.dists = getDist(self.appos,self.uepos)
        self.power = np.ones(self.dists.shape) * 10
        self.passGain = DSPloss(self.dists,shadowing_std=7)
        self.apID = np.argmax(self.passGain,0)
        self.apUE = dict()
        for i in range(self.apID.shape[0]):
            self.apUE[i] = np.where(self.apID == i)[0]
        self.achRate = getAchRate(self.passGain,self.power,-134)
        self.weight = 1/self.achRate
        self.infPower = getInfpower(self.passGain, self.power)
        self.SINR = getSINR(self.passGain, self.infPower, -134)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = [self.achRate]


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

    def step(self, actions):

        ues = []
        for i in range(self.appos.shape[0]):
            ues.append(self.ap_action(i,actions[i,0], actions[i,1]))

        self.achRate = (1 - self.alphaR) * self.achRate + self.alphaR * getAchRate(self.passGain,self.power,-134)
        self.weight = 1/self.achRate
        self.infPower = (1 - self.alphaI) * self.infPower + self.alphaI * getInfpower(self.passGain, self.power)
        self.SINR = getSINR(self.passGain, self.infPower, -134)
        self.pf = self.weight * np.log2(1 + self.SINR)

        reward = 0
        lambda_rew = 0.4

        for i in range(len(ues)):

            if ues[i] != None:
                reward += self.weight[ues[i]] ** lambda_rew * self.achRate[ues[i]]


        if reward == 0:
            reward = - np.max(self.pf)

        self.accumulated_rate.append(self.achRate)

        return self.global_obv(), reward





    def ap_action(self, apID, ueID, power):


        ue_id = self.apUE[apID]

        if ueID >= len(ue_id):
            ue = None
            self.power[apID,:] = 0
        else:
            ue = ue_id[ueID]
            self.power[apID,:] = power

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
        cm = np.linspace(1, self.apnum, self.apnum)

        plt.figure(figsize=(5,5))
        plt.scatter(self.appos[:,0], self.appos[:,1],marker="x", s = s, c = cm)

        plt.scatter(self.uepos[:,0],self.uepos[:,1],marker="o", s = self.achRate * 20, c = self.apID)
        plt.xlim([0,self.boarder])
        plt.ylim([0,self.boarder])
        plt.show()




if __name__ == "__main__":

    env = Env(4,24,500)
    print(env.get_R5per()/4,env.get_Rsum()/4)
    actions = np.array([[3,9],[1,5],[0,3],[2,10]])
    env.plot_scheduling()
    reward = env.step(actions)
    actions = np.array([[3,9],[1,5],[0,3],[2,10]])
    print(env.get_R5per()/4,env.get_Rsum()/4)
    env.plot_scheduling()





