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
             epsilon=480,discount=0.99)


#%%

rewards = []
v_losses = []
R_sum = []
R_5per = []


for i in range(408000):

    if i < 8000:
        act = agent.random_action()
    else:
        act = agent.select_action(env.global_obv().float().cuda())

    act = (act + 1)/2
    act = np.reshape(act,(4,2))
    act[:,0] = np.round(act[:,0] * 5)
    act[:,1] = np.round(act[:,1] * 5) * 0.002

    if np.any(np.isnan(act)):
        continue

    s_t1,r = env.step(act)



    if (i + 1) % 2000 == 0:
        agent.observe(r, s_t1, 1)
        env.reset()
        agent.epsilon = 1
    else:
        agent.observe(r, s_t1, 0)

    if i >= 8000:
        v_loss, p_loss = agent.update_policy()
        print(f"Step{i-99} | critic loss: {v_loss}| actor loss: {p_loss}| reward: {r}"
              f"| R_sum: {env.get_Rsum()}| R_5per: {env.get_R5per()} \n")
        if (i + 1) % 20000 == 0:
            step = int((i + 1 - 8000) / 20000)
            agent.save_model("./models",step=str(step))



    # print(f"Current action : {act}")



    # rewards.append(r)
    # R_sum.append(env.get_Rsum())
    # R_5per.append(env.get_R5per())


# plt.figure()
# plt.plot(R_sum)
# plt.xlabel("episode length")
# plt.ylabel("Rsum (bps/Hz)")
# plt.grid()
#
# plt.figure()
# plt.plot(rewards)
# plt.xlabel("episode length")
# plt.ylabel("Reward")
# plt.grid()
#
#
# plt.figure()
# plt.plot(R_5per)
# plt.xlabel("episode length")
# plt.ylabel("R5% (bps/Hz)")
# plt.grid()
#
#
# env.plot_scheduling()
# plt.show()