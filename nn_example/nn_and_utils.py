#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class NeuralNet(nn.Module):
    def __init__(self, in_size, out_size, num_layers, hidden_size, lr):
        super(NeuralNet, self).__init__()

        self.dropl = nn.Dropout(p=0.5)
        self.elu = nn.ELU()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        llayers = []
        next_input_l_size = in_size
        for i in range(num_layers):
            ll = nn.Linear(next_input_l_size, hidden_size)
            self._initialize_layer(ll)
            llayers.append(ll)
            llayers.append(self.elu)
            llayers.append(nn.BatchNorm1d(hidden_size))
            llayers.append(self.dropl)
            next_input_l_size = hidden_size

        llast = nn.Linear(next_input_l_size, out_size)
        self._initialize_layer(llast)
        llayers.append(llast)

        self.layers = nn.Sequential(*llayers)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
            )
        self.msel = nn.MSELoss(reduction='none')
        self.hardtanh = nn.Hardtanh()

    def criterion(self, output, target):
        loss = self.msel(output, target)
        #loss = self.hardtanh(loss)
        return loss.mean()

    def forward(self, x):
        x = self.layers(x)
        return x

    def _initialize_layer(self, layer):
        nn.init.constant_(layer.bias, 0)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(layer.weight, gain=gain)

def decide(best_action):
    action = np.random.choice(5, p=[.05, .05, .05, .05, .8])
    if action == 4:
        return best_action
    return action

class ReplayMemory():
    def __init__(self, capacity, batch_size, dim):
        self.capacity = capacity
        self.batch_size = batch_size
        self.dim = dim

        self.memory = torch.empty((capacity, dim)).cuda()
        self.insert_pos = 0
        self.ready_to_get = False

    def insert_sample(self, new):
        self.memory[self.insert_pos] = torch.tensor(new)
        self.insert_pos += 1
        if self.insert_pos >= self.capacity:
            self.insert_pos = 0
            self.ready_to_get = True

        return self

    def get_batch(self):
        if not self.ready_to_get:
            raise Exception("Must load memory first")
        choice = np.random.choice(self.capacity, self.batch_size)
        batch = self.memory[choice]
        #batch = batch.contiguous()

        return batch

def formatter(f):
    f = np.round(f, 2)
    fabs = np.abs(f)
    r = str(f)

    if fabs == np.round(fabs, 1):
        r += "0"
    elif fabs == np.round(fabs, 0):
        r += "00"

    if f > 0:
        r = "+" + r
    if fabs < 10:
        r = " " + r
    if f == 0:
        r = "  --- "


    r += " "

    return r

def print_statistics(neural_net):
    action_value = np.zeros((16, 5, 4))
    for time in range(1):
        states = np.empty((16, 2))
        states[:, 0] = np.arange(16)
        states[:, 1] = time
        states = torch.as_tensor(states, dtype=torch.float32)

        with torch.no_grad():
            neural_net.eval()
            action_value_t = neural_net(dummify(states.cuda())).cpu().numpy()
            action_value[:, time, :] = action_value_t

    np.set_printoptions(formatter=dict(float=formatter))
    for time in range(1):
        action_value_f = action_value[:, time]
        for i in range(0, 16, 4):
            for j in range(i, (i+4)):
                print("    ", end="")
                print(action_value_f[j, [0]], end="     ")
            print()

            for j in range(i, (i+4)):
                print(action_value_f[j, [3,1]], end=" ")
            print()

            for j in range(i, (i+4)):
                print("    ", end="")
                print(action_value_f[j, [2]], end="     ")
            print()

            print("--------------------------------------------------")

        policy = np.argmax(action_value_f, 1).reshape((4,4))
        policy = np.array(policy, dtype=str)
        policy[policy == '0'] = " up  "
        policy[policy == '1'] = "right"
        policy[policy == '2'] = "down "
        policy[policy == '3'] = "left "
        print(policy)

def dummify(x):
    x = x.cpu().numpy()
    y = x[:, 0]
    y = np.hstack((np.arange(16), y))
    y = pd.get_dummies(y)[16:]
    x = np.hstack((y, x[:, [1]]))
    x = torch.as_tensor(x).cuda()
    return x
