import numpy as np
import math
import random
import matplotlib.pyplot as plt

def getDist(appos, uepos):

    apnum = appos.shape[0]
    uenum = uepos.shape[0]

    dists = np.zeros((apnum,uenum))

    for i in range(apnum):

        dists[i,:] = np.sqrt(np.power(appos[i,0] - uepos[:,0],2) + np.power(appos[i,1] - uepos[:,1],2))

    return dists

def getAchRate(losses,power,noise):

    if type(power) == np.ndarray:
        flag = True
    else:
        flag = False


    power_real = 10 ** ((power - 30)/10)

    losses_sqr = 10 ** (losses / 10)

    noise = 10 ** (noise/10)

    if flag:
            power_rec = losses_sqr * power_real
    else:
            power_rec = losses_sqr * power_real

    apID = np.argmax(losses,0)

    acc_rate = np.zeros((losses.shape[1],))

    for i in range(losses.shape[1]):

        power_sig = power_rec[apID[i],i]
        power_inf_noise = np.sum(power_rec[:,i], 0) - power_sig + noise
        acc_rate[i] = np.log2(1 + power_sig * 1/power_inf_noise)

    return acc_rate

def getInfpower(losses,power):

    apID = np.argmax(losses,0)

    power_real = 10 ** ((power - 30)/10)

    losses_sqr = 10 ** (losses / 10)

    power_rec = losses_sqr * power_real

    infpower = np.zeros(apID.shape)

    for i in range(losses.shape[1]):
        power_sig = power_rec[apID[i],i]
        infpower[i] = np.sum(power_rec[:,i], 0) - power_sig

    return  infpower

def getSINR(losses,infpower,noise):

    SINR = np.zeros(infpower.shape)

    power_real = 10 ** (-20/10)

    losses_sqr = 10 ** (losses / 10)

    apID = np.argmax(losses,0)

    noise = 10 ** (noise/10)

    for i in range(infpower.shape[0]):
        SINR[i] = power_real * losses_sqr[apID[i],i] /(infpower[i] + noise)


    return SINR





def APgen(area_range, apnum, min_dist, paint = False):

    pos = np.zeros((apnum,2))

    for i in range(apnum):

        while True:
            xpos = random.random() * area_range
            ypos = random.random() * area_range

            if np.all(np.sqrt(np.power(pos[:,0] - xpos,2) + np.power(pos[:,1] - ypos,2)) > min_dist):
                pos[i,0] = xpos
                pos[i,1] = ypos
                break

    if paint:
        plt.scatter(pos[:,0],pos[:,1])
        plt.xlim([0,area_range])
        plt.ylim([0,area_range])
        plt.show()

    return pos

def UEgen(appos,area_range,uenum,min_dist,paint = False ):

    uepos = np.zeros((uenum,2))

    for i in range(uenum):

        while True:

            xpos = random.random() * area_range
            ypos = random.random() * area_range

            if np.all(np.sqrt(np.power(appos[:,0] - xpos,2) + np.power(appos[:,1] - ypos,2)) > min_dist):

                uepos[i,0] = xpos
                uepos[i,1] = ypos
                break

    if paint:
        cm = np.linspace(0,appos.shape[0] - 1, appos.shape[0])
        dists = getDist(appos,uepos)
        apID = np.argmin(dists,0)

        plt.figure()
        plt.set_cmap("RdBu_r")
        plt.scatter(appos[:,0], appos[:,1],s=100,marker="x",c=cm)
        plt.scatter(uepos[:,0],uepos[:,1],marker="o",c=apID)
        plt.xlim([0,area_range])
        plt.ylim([0,area_range])
        plt.show()

    return uepos

def DSPloss(dists, conf=[5.8*10e8,3,2,2,4], shadowing_std = 7 , **kwargs):
    """

    :param dists: array of distances
    :param conf: configuration of dsp loss, including fc; ht; hr; alpha0 and alpha1
    :param shadowing_std: The std of log-normal shadowing
    :param kwargs: mode, pre-defined configs that can be used directly
    :return:
    """

    if len(kwargs) > 0:
        try:
            conf = MODELS[kwargs['mode']]

        except:
            print("Unknown key word or mode not supported!")

    fc = conf[0]
    lambdac = 3e9 / fc
    ht = conf[1]
    hr = conf[2]
    alpha0 = conf[3]
    alpha1 = conf[4]

    Rc = 4*ht*hr / lambdac
    # antenna gain is set to 10 dBi each
    K0 = 20 * math.log10(lambdac/(4*math.pi)) + 20

    mask = dists > Rc
    xdim,ydim = dists.shape
    loss = np.zeros((xdim,ydim))

    loss[~mask] = -10 * alpha0 * np.log10(dists[~mask])
    loss[mask] = 10 * (alpha1 - alpha0) * np.log10(Rc) - 10 * alpha1 * np.log10(dists[mask])

    shadowing = np.random.randn(xdim,ydim) * shadowing_std

    return loss + shadowing + K0

def rayleigh_fading():
    pass




MODELS = {
    "Macro":[8.6*10e8,60,2,2,4],
    "802.11b":[2.4*10e9,3,2,2,4],
    "802.11a":[5.8*10e9,3,2,2,4],
    "LTE":[7*10e8,5,2,2,4],
    "mmWave":[6*10e10,2,2,2,4],
    "eg":[10e9,10,2,2,4],
}


if __name__ == "__main__":

    appos = APgen(500,4,35)
    uepos = UEgen(appos,500,24,19)
    dists = getDist(appos,uepos)

    loss = DSPloss(dists,mode = "eg")
    acc_rate = getAchRate(loss, 10, -134)
    acc_rate.sort()
    print(acc_rate,sum(acc_rate))