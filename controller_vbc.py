import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from rnn_agent import RNNAgent

# Definition of Message Encoder
class Env_blender(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape):
        super(Env_blender, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape, hidden_shape)
        self.fc3 = nn.Linear(hidden_shape, output_shape)

    def forward(self, input_mac):
        hidden = F.elu(self.fc1(input_mac))
        hidden = F.elu(self.fc2(hidden))
        output_mac = self.fc3(hidden)
        return output_mac
    
# Agent Network
class VDN_MAC:
    def __init__(self, n_agents = 4, n_actions = 6, input_shape = 21, delta = th.tensor([10.0, 10.0])):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.input_shape = input_shape
        self._build_agents(input_shape)
        #self.agent_output_type = args.agent_output_type

        #self.action_selector = action_REGISTRY[args.action_selector](args)
        self.env_blender = Env_blender(64, self.n_actions, 196) #.cuda()
        self.delta1 = delta[0]
        self.delta2 = delta[1]
        self.epsilon_greedy = 0.2
        self.hidden_states = None

        self.cuda_flag = False
        
    def choose_action(self, obs, test_mode=False):
        ori_q_value, hiddens = self.forward(obs) # hiddens.shape = 1, 4, 64
        dummys = th.stack([self.env_blender(hiddens[:, i, :].view(1, -1)).detach() for i in range(self.n_agents)], dim=1) # dummys.shape = 1, 4, 6
        sum_dummy = dummys.sum(1) # 1, 1, 6
        dummys = (sum_dummy - dummys) / (self.n_agents - 1)
        if test_mode:
            std = th.std(dummys, dim = 2)
            dummys = dummys * (std > self.delta2).unsqueeze(2)
        q_value = ori_q_value.detach() #+ dummys
        actions = q_value.argmax(dim=-1)
        if not test_mode and np.random.rand() < self.epsilon_greedy:
            actions = th.randint(6, (4,))
        return actions.view(-1)

    def forward(self, ep_batch, test_mode=False):
        # ep_batch.shape == [batch, n_agents, limited_n_ues, 4(No., x, y, patience)]
        batch = ep_batch.shape[0]
        agent_inputs = ep_batch.view(batch, self.n_agents, -1)
        agent_number = th.arange(self.n_agents, dtype=th.float32).unsqueeze(0).expand(batch, -1).view(batch, self.n_agents, 1)
        if self.cuda_flag:
            agent_number = agent_number.cuda()
        agent_inputs = th.cat([agent_inputs, agent_number], dim=-1).view(batch * self.n_agents, -1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)  
        
        return agent_outs.view(batch, self.n_agents, -1), self.hidden_states.view(batch, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        if self.cuda_flag:
            self.hidden_states = self.hidden_states.cuda()

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()
        self.env_blender.cuda()
        self.cuda_flag = True

    def _build_agents(self, input_shape):
        self.agent = RNNAgent(input_shape, 64, self.n_actions)

    def save(self, filename):
        outfile = {'agent':self.agent.state_dict(), 'blender':self.env_blender.state_dict()}
        th.save(outfile, filename)

    def load(self, filename):
        infile = th.load(filename)
        self.agent.load_state_dict(infile['agent'])
        self.env_blender.load_state_dict(infile['blender'])