#%%

from ddpg import DDPG
from ENV import KYenv
import numpy as np
import matplotlib.pyplot as plt



#%%

env = KYenv(4,24,500)
agent = DDPG(57,8,seed=5,hidden1=400,
             hidden2=300,bsize=64,
             rate=0.0001,prate=0.00001,
             rmsize=int(60000),window_length=int(1),
             tau=0.001,init_w=0.003,
             epsilon=1000,discount=0.99)


#%%

rewards = []
v_losses = []

for i in range(500):

    if np.random.rand() < agent.epsilon:
        act = agent.random_action()
    else:
        act = agent.select_action(env.global_obv().float())


    act = np.abs(act)
    act = np.reshape(act,(4,2))
    act[:,0] = np.round(act[:,0] * 4)
    act[:,1] = act[:,1] * 0.01

    if np.any(np.isnan(act)):
        continue

    s_t1,r = env.step(act)
    agent.observe(r,s_t1,0)
    if agent.memory.nb_entries > 100:
        v_loss, p_loss = agent.update_policy()
        print(f"Step{i-99} | critic loss: {v_loss}| actor loss: {p_loss}| reward: {r}"
              f"| R_sum: {env.get_Rsum()}| R_5per: {env.get_R5per()} \n")

    rewards.append(r)

plt.plot(rewards)
plt.show()