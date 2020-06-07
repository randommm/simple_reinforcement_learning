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

try:
    from scipy.special import softmax
except ImportError:
    def softmax(x):
        y = x - np.max(x)
        w = np.exp(y)
        w = w / w.sum()
        return w

def progress_bar(current, total, barsize=65):
    current = current + 1
    base = "\r[{:{barsize}}] {:.2f}%"
    f1 = "="*(current*barsize//total)
    f2 = 100/total*current
    print(base.format(f1, f2, barsize=barsize), end='')
    if current == total:
        print()

# Actions:
# up = 0
# right = 1
# down = 2
# left = 3

# Map:
#  0   1   2   3
#  4  5B   6   7W
#  8  9B  10L  11
# 12  13  14   15
# 5 and 9: barriers
# 10: lose
# 7: win

class Game():
    def __init__(self):
        # which states of the grid will block movement for each action
        self.state_barriers_3 = [0, 4, 8, 12, 6]
        self.state_barriers_2 = [12, 13, 14, 15, 1]
        self.state_barriers_1 = [3, 11, 15, 4, 8]
        self.state_barriers_0 = [0, 1, 2, 3, 13]

        # which states t+1 of the grid will cause lose/win
        self.state_lose = 10
        self.state_win = 7
        self.barriers = [5, 9]

        self.reward_noend = -0.1
        self.reward_win = 10
        self.reward_lose = -10

        self.state = self.state_lose
        while (
            self.state in self.barriers or
            self.state in [self.state_lose, self.state_win]
            ):
            self.state = np.random.choice(16)


        self.ended = 0

    def act(self, action):
        assert self.ended == 0  # ensure no playing after game is over
        assert self.state not in self.barriers
        assert self.state not in [self.state_win, self.state_lose]

        real_action = self._find_real_action(action)
        reward, self.ended = self._update_state(real_action)

        return self.state, reward, self.ended

    # Given an action, find true action that will be performed
    def _find_real_action(self, action):
        # 45% prob -> action itself (0 degrees)
        # 25% prob -> 90 or -90 degrees
        # 5% -> inverse action (180 degrees)
        rng_action = np.random.random()
        if rng_action >= 0.55:
            real_action = action
        elif rng_action >= 0.3:
            real_action = action - 3
        elif rng_action >= 0.05:
            real_action = action - 1
        else:
            real_action = action - 2

        if real_action == -3:
            real_action = 1
        elif real_action == -2:
            real_action = 2
        elif real_action == -1:
            real_action = 3

        return real_action

    def _update_state(self, real_action):
        no_end_reward = self.reward_noend, 0

        if real_action == 0:
            if self.state in self.state_barriers_0:
                return no_end_reward
            to_sum = -4
        elif real_action == 1:
            if self.state in self.state_barriers_1:
                return no_end_reward
            to_sum = 1
        elif real_action == 2:
            if self.state in self.state_barriers_2:
                return no_end_reward
            to_sum = 4
        elif real_action == 3:
            if self.state in self.state_barriers_3:
                return no_end_reward
            to_sum = -1
        else:
            raise ValueError("Invalid real_action", real_action)

        self.state += to_sum
        if self.state == self.state_lose:
            return self.reward_lose, 1
        elif self.state == self.state_win:
            return self.reward_win, 1

        return no_end_reward

def decide(state, action_value):
    action = np.random.choice(5, p=[.1, .1, .1, .1, .6])
    if action <= 3:
        return action

    actions_log_weights = action_value[state]
    actions_weights = softmax(actions_log_weights)
    action = np.random.choice(4, p=actions_weights)
    return action

action_value = np.zeros((16, 4))

nepisodes = 20_0
lrs = np.geomspace(2, 1, nepisodes)-1
for episode in range(nepisodes):
    progress_bar(episode, nepisodes)
    ended = 0
    game = Game()
    lr = lrs[episode]
    while not ended:
        cur_state = game.state
        # action = np.random.choice(4)
        action = decide(cur_state, action_value)
        new_state, reward, ended = game.act(action)
        p1 = (1 - lr) * action_value[cur_state, action]
        p2 = action_value[new_state, np.argmax(action_value[new_state])]
        p2 = lr * (reward + .75 * p2)
        action_value[cur_state, action] = p1 + p2

def formatter(f):
    f = np.round(f, 2)
    r = str(f)

    if f == np.round(f, 1):
        r += "0"
    elif f == np.round(f, 0):
        r += "00"

    if f > 0:
        r = " " + r
    elif f == 0:
        r = " 0.00"

    r += " "

    return r

np.set_printoptions(formatter=dict(float=formatter))
for i in range(0, 16, 4):
    for j in range(i, (i+4)):
        print("   ", end="")
        print(action_value[j, [0]], end="     ")
    print()

    for j in range(i, (i+4)):
        print(action_value[j, [3,1]], end=" ")
    print()

    for j in range(i, (i+4)):
        print("   ", end="")
        print(action_value[j, [2]], end="     ")
    print()

    print("-"*64)

policy = np.argmax(action_value, 1).reshape((4,4))
policy = np.array(policy, dtype=str)
policy[policy == '0'] = " up  "
policy[policy == '1'] = "right"
policy[policy == '2'] = "down "
policy[policy == '3'] = "left "
print(policy)
