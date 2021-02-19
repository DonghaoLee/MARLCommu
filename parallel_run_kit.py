import time
import torch
import numpy as np
from multiprocessing import Pool

from run import run

class ParallelRun():
    def __init__(self, env, agent, max_length, test_mode=False, flag=False):
        self.env = env
        self.agent = agent
        self.max_length = max_length
        self.test_mode = test_mode
        self.flag = flag 
    
    def run(self, batch_size):
        p = Pool(batch_size)
        base_seed = time.time()
        seeds = [base_seed + i for i in range(batch_size)]

        batch = p.map(self.randrun, seeds)
        p.close()
        p.join()
        return batch

    def randrun(self, seed):
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        b = run(self.env, self.agent, self.max_length, test_mode = self.test_mode)
        return b