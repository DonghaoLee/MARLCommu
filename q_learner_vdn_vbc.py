import copy
import torch as th
import torch.nn as nn
import numpy as np
from torch.optim import RMSprop, SGD


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs):
        return th.sum(agent_qs, dim=2, keepdim=True)


class QLearner:
    def __init__(self, mac):
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        
        self.params = list(mac.parameters())

        self.mixer = VDNMixer()
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.params += list(self.mac.env_blender.parameters())
        #self.optimiser = RMSprop(params=self.params, lr=0.1, alpha=0.99, eps=0.00001)
        self.optimiser = SGD(params=self.params, lr=0.01, momentum = 0.9)

        self.gamma = 0.0 #0.98
        self.normalization_const = 10.
        self.grad_norm_clip = 500.

    def set_sgd(self, lr):
        self.optimiser = SGD(params=self.params, lr=lr, momentum = 0.)

    def train(self, batch):
        # batch["obs"].shape=[batch, t, obs]

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1].unsqueeze(-1)
        batch = batch["obs"]
        batch_size = batch.shape[0]
    
        # Calculate estimated Q-Values
        mac_out = []
        difference_out = []
        difference_out1 = []         
        self.mac.init_hidden(batch_size)
        for t in range(batch.shape[1]):
            agent_local_outputs, hidden_states = self.mac.forward(batch[:, t])
            dummy0 = self.mac.env_blender(hidden_states[:,0,:].view(batch_size,-1))
            dummy1 = self.mac.env_blender(hidden_states[:,1,:].view(batch_size,-1))
            dummy2 = self.mac.env_blender(hidden_states[:,2,:].view(batch_size,-1))
            dummy3 = self.mac.env_blender(hidden_states[:,3,:].view(batch_size,-1)) 
            
            agent0 = (dummy1 + dummy2 + dummy3)/3.0
            agent1 = (dummy0 + dummy2 + dummy3)/3.0
            agent2 = (dummy0 + dummy1 + dummy3)/3.0
            agent3 = (dummy0 + dummy1 + dummy2)/3.0
    
            agent_global_outputs = th.cat((agent0.view((batch_size,1,6)),agent1.view((batch_size,1,6)),agent2.view((batch_size,1,6)),agent3.view((batch_size,1,6))),1)
            agent_outs = agent_local_outputs #+ agent_global_outputs
            difference = agent_global_outputs 
            mac_out.append(agent_outs)
            difference_out.append(difference)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        difference_out = th.stack(difference_out, dim=1)  # Concat over time

        difference_out = th.std(difference_out,dim = 3).sum()
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        avg_difference = difference_out.mean()/((agent_outs.shape[0]*agent_outs.shape[1]*agent_outs.shape[2]*batch.shape[1]))

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch_size)
        for t in range(batch.shape[1]):
            target_agent_local_outputs, target_hidden_states = self.target_mac.forward(batch[:, t])
    
            dummy0 = self.target_mac.env_blender(target_hidden_states[:,0,:].view(batch_size,-1))
            dummy1 = self.target_mac.env_blender(target_hidden_states[:,1,:].view(batch_size,-1))
            dummy2 = self.target_mac.env_blender(target_hidden_states[:,2,:].view(batch_size,-1))
            dummy3 = self.target_mac.env_blender(target_hidden_states[:,3,:].view(batch_size,-1))

            target_agent0 = (dummy1 + dummy2 + dummy3)/3.0
            target_agent1 = (dummy0 + dummy2 + dummy3)/3.0
            target_agent2 = (dummy0 + dummy1 + dummy3)/3.0
            target_agent3 = (dummy0 + dummy1 + dummy2)/3.0

            target_agent_global_outputs = th.cat((target_agent0.view((batch_size,1,6)),
                                                  target_agent1.view((batch_size,1,6)),
                                                  target_agent2.view((batch_size,1,6)),
                                                  target_agent3.view((batch_size,1,6))),1)
            target_agent_outs = target_agent_local_outputs #+ target_agent_global_outputs
            target_mac_out.append(target_agent_outs)
          
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Max over target Q-Values
        
        # Get actions that maximise live Q (for double q-learning)
        t_mac_out = mac_out.clone()
        cur_max_actions = t_mac_out[:, 1:].max(dim=3, keepdim=True)[1]
        target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        #target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals).squeeze(-1)
        target_max_qvals = self.target_mixer(target_max_qvals).squeeze(-1)
        #target_max_qvals[]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * target_max_qvals

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).mean()# + self.normalization_const * avg_difference

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        print(loss)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        print(grad_norm)
        self.optimiser.step()
        return loss, grad_norm

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()