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
from game import Game
from nn_and_utils import NeuralNet, decide, ReplayMemory, formatter
from nn_and_utils import print_statistics, dummify
import torch

nrefroze = 40
nepisodes = 5_000_000
replay_memory_capacity = 5_000
batch_size = 100
lr = 0.05

# Construct and load memory
replay_memory = ReplayMemory(replay_memory_capacity, batch_size, 7)
game = Game()
for _ in range(replay_memory_capacity):
    cur_state = game.state
    action = np.random.choice(4)
    new_state, reward, ended = game.act(action)
    sample = [*cur_state, *new_state, action, reward, ended]
    replay_memory.insert_sample(sample)
    if ended:
        game = Game()

neural_net = NeuralNet(17, 4, 5, 1000, lr).cuda()

#example = torch.as_tensor(cur_state,dtype=torch.float32).cuda()[None]
#example = torch.cat((example,example))
#neural_net.forward = torch.jit.trace(neural_net, example).forward

neural_net_frozen = NeuralNet(17, 4, 5, 1000, lr).cuda()
neural_net_frozen.load_state_dict(neural_net.state_dict())
neural_net_frozen.eval()

refroze = 0
for episode in range(nepisodes):
    if not episode % 100:
        print(episode + 1, "out of", nepisodes)
        print_statistics(neural_net)

    ended = 0
    game = Game()

    while not ended:
        #print(neural_net.state_dict())
        cur_state = game.state

        with torch.no_grad():
            neural_net.eval()
            best_action = torch.as_tensor(cur_state,dtype=torch.float32)
            best_action = neural_net(dummify(best_action.cuda()[None]))
            best_action = np.argmax(best_action.cpu().numpy()).item()

        action = decide(best_action)
        #action = np.random.choice(4)
        new_state, reward, ended = game.act(action)
        sample = [*cur_state, *new_state, action, reward, ended]
        replay_memory.insert_sample(sample)
        #if np.random.random() <= 0.95:
        #    continue

        neural_net.train()
        neural_net.optimizer.zero_grad()

        batch = replay_memory.get_batch()
        batch_cur_state = batch[:, [0,1]]
        batch_new_state = batch[:, [2,3]]
        batch_action = torch.as_tensor(batch[:, 4], dtype=torch.int64)
        batch_reward = batch[:, 5]
        batch_ended = torch.as_tensor(batch[:, 6], dtype=torch.int64)

        output = neural_net(dummify(batch_cur_state))
        idx = tuple(range(len(batch_action)))
        idx = (idx, tuple(batch_action.numpy()))
        output = output[idx]

        with torch.no_grad():
            target = neural_net_frozen(dummify(batch_new_state))
            target = torch.max(target, 1)[0]
            target = 0.8 * target

            target = target.cpu().numpy()
            target[batch_ended.numpy()==1] = 0.
            target = torch.as_tensor(target)
            target = target.cuda()

            target = target + batch_reward

        loss = neural_net.criterion(output, target)
        loss.backward()
        np_loss = loss.data.item()
        if np.isnan(np_loss):
            raise RuntimeError("Loss is NaN")
        neural_net.optimizer.step()
        neural_net.scheduler.step()

        if refroze >= nrefroze:
            neural_net_frozen.load_state_dict(neural_net.state_dict())
            refroze = 0
        else:
            refroze += 1
