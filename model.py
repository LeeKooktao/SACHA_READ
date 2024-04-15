import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# config
import yaml
config = yaml.safe_load(open("./configs/config_base.yaml", 'r'))
policy_input_shape = (config['K_obs'], 3, config['fov_size'][0], config['fov_size'][1])#3x3x9x9
policy_output_shape = config['action_dim']
action_dim = config['action_dim']
hidden_channels = config['hidden_channels']#128
hidden_dim = config['hidden_dim']#256
fov_size = tuple(config['fov_size'])#（9，9）
obs_r = (int(np.floor(fov_size[0]/2)), int(np.floor(fov_size[1]/2)))#（4,4）
num_heads = config['num_heads']#2
K_obs = config['K_obs']#3


class MHABlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)
        self.mha = nn.MultiheadAttention(output_dim * num_heads, num_heads, batch_first=True)

    def forward(self, x):
        output, _ = self.mha(self.W_Q(x), self.W_K(x), self.W_V(x))
        output = self.W_O(output)
        return output


# https://github.com/tkipf/pygcn
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
#x：节点特征张量，adj：邻接矩阵
    def forward(self, x, adj):
        #矩阵乘法：输入与权重矩阵相乘
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        #与邻接矩阵相乘
        return torch.sparse.mm(adj, x)

#基于图卷积的交流模块，采用两层图卷积实现
class CommBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5, use_bias=True):
        super().__init__()
        self.gcn_1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def initialize_weights(self):
        self.gcn_1.initialize_weights()
        self.gcn_2.initialize_weights()

    def forward(self, x, adj):
        x = F.relu(self.gcn_1(x, adj))
        x = self.dropout(x)
        x = self.gcn_2(x, adj)
        return x

#残差块，两层的卷积层实现
class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        output = x
        output = self.conv1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output += x
        output = F.relu(output)
        return output


class AttentionPolicy(nn.Module):
    #com、3x3x9x9、5、128、256、2
    def __init__(self, communication, input_shape=policy_input_shape, output_shape=policy_output_shape,
                 hidden_channels=hidden_channels, hidden_dim=hidden_dim, num_heads=num_heads):
        super().__init__()
        self.communication = communication
        self.input_shape = policy_input_shape
        self.output_shape = policy_output_shape
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.obs_encoder = nn.Sequential(
            #（128,9,9）
            nn.Conv2d(in_channels=3, out_channels=self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            ResBlock(self.hidden_channels),
            ResBlock(self.hidden_channels),
            ResBlock(self.hidden_channels),
            #（16，9，9）
            nn.Conv2d(in_channels=hidden_channels, out_channels=16, kernel_size=1),
            nn.ReLU(True),
            #1296
            nn.Flatten(0),
        )
        self.memory_encoder = nn.GRUCell(16 * self.input_shape[2] * self.input_shape[3], self.hidden_dim)
        self.mha_block = MHABlock(self.hidden_dim, self.hidden_dim, self.num_heads)
        self.comm_block = CommBlock(self.hidden_dim)
        self.value_decoder = nn.Linear(self.hidden_dim, 1)
        self.advantage_decoder = nn.Linear(self.hidden_dim, self.output_shape)
        # 为模型中的线性层和二维卷积层初始化权重（使用 xavier_uniform_ 初始化方法）和偏置（如果存在的话，使用常数初始化方法）
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    #得到邻接矩阵，若两个智能体都在4x4的范围内，则邻接矩阵为1，否则为0
    def _get_adj_from_states(self, state):
        num_agents = len(state)
        adj = np.zeros((num_agents, num_agents), dtype=float)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                x_i, y_i = state[i][0], state[i][1]
                x_j, y_j = state[j][0], state[j][1]
                if abs(x_i - x_j) <= obs_r[0] and abs(y_i - y_j) <= obs_r[1]:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0
        return adj

    def forward(self, obs, hidden, state):
        num_agents = len(state)
        next_hidden, latent = [], []#latent用于存放节点特征
        obs, hidden = obs.float(), hidden.float()
        for i in range(num_agents):
            o_i = [self.obs_encoder(obs[i, k, :]) for k in range(K_obs)]  # 使用obs_encoder对三通道的观测数据进行编码
            o_i = [self.memory_encoder(o, h) for o, h in zip(o_i, hidden)]
            #此时刻的输出和隐状态堆叠得到信息
            o_i = torch.stack(o_i)
            o_i = self.mha_block(o_i)
            next_hidden.append(torch.sum(o_i, 0))
            latent.append(torch.sum(o_i, 0))
        if self.communication:
            adj = torch.from_numpy(self._get_adj_from_states(state)).to(torch.float32)
            latent = torch.stack(latent)
            adj = adj.cuda(latent.get_device()) if latent.is_cuda else adj
            latent = self.comm_block(latent, adj)
        V_n = [self.value_decoder(x) for x in latent]
        A_n = [self.advantage_decoder(x) for x in latent]
        # V和A的维度不同，将V拓展后与A相加
        Q_n = [V + A - A.mean(0, keepdim=True) for V, A in zip(V_n, A_n)]
        # log(pi)策略梯度使用
        log_pi = [F.log_softmax(Q, dim=0) for Q in Q_n]
        action = [torch.argmax(Q, dim=0) for Q in Q_n]
        return action, torch.stack(next_hidden), log_pi


class AttentionCritic(nn.Module):
    def __init__(self, action_dim=policy_output_shape, hidden_dim=hidden_dim,
                 hidden_channels=hidden_channels, num_heads=num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.fov_encoder = nn.Sequential(
            # 128x9x9
            nn.Conv2d(in_channels=3, out_channels=self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            ResBlock(self.hidden_channels),
            ResBlock(self.hidden_channels),
            ResBlock(self.hidden_channels),
            # 16x9x9
            nn.Conv2d(in_channels=hidden_channels, out_channels=16, kernel_size=1),
            nn.ReLU(True),
            # 1296
            nn.Flatten(0),
        )
        self.obs_encoder = nn.Linear(16 * fov_size[0] * fov_size[1], 16)
        self.action_encoder = nn.Linear(action_dim, 16)
        self.mha_block = MHABlock(32, 32, num_heads)
        self.value_decoder = nn.Linear(32, 1)
        self.advantage_decoder = nn.Linear(32, action_dim)

    def _get_observed_agents_from_states(self, state):
        num_agents = len(state)
        observed_agents = [[] for _ in range(num_agents)]
        for i in range(num_agents):
            observed_agents[i].append(i)
            for j in range(i + 1, num_agents):
                x_i, y_i = state[i][0], state[i][1]
                x_j, y_j = state[j][0], state[j][1]
                if abs(x_i - x_j) <= obs_r[0] and abs(y_i - y_j) <= obs_r[1]:
                    observed_agents[i].append(j)
        return observed_agents

    def forward(self, obs, action, state):
        num_agents = len(state)
        observed_agents = self._get_observed_agents_from_states(state)
        latent = []
        for i in range(num_agents):
            c_i = []
            for j in observed_agents[i]:
                o_j = [self.obs_encoder(self.fov_encoder(obs[j, k, :])) for k in range(K_obs)]
                o_j = torch.mean(torch.stack(o_j), 0)
                a_j = F.one_hot(action[j].clone().to(torch.int64), num_classes=action_dim).float()
                a_j = self.action_encoder(a_j)
                c_i.append(torch.concat((o_j, a_j)))
            c_i = torch.stack(c_i)
            c_i = self.mha_block(c_i)
            latent.append(torch.sum(c_i, 0))
        V_n = [self.value_decoder(x) for x in latent]
        A_n = [self.advantage_decoder(x) for x in latent]
        Q_n = [V + A - A.mean(0, keepdim=True) for V, A in zip(V_n, A_n)]
        return Q_n
    #计算policy的优势函数的baseline，参考原文b的计算：不采用原始动作，采用其他的所有动作得到的Q值的和
    def get_coma_baseline(self, obs, action, state):
        b = []
        for i in range(len(state)):
            p = 0.0
            for j in range(self.action_dim):
                temp_action = copy.deepcopy(action)
                temp_action[i] = j
                p += self.forward(obs, action, state)[i] / self.action_dim
            b.append(p)
        return b