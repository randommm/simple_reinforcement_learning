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


        self.loss_spendbat = 0.4 # loss when no win nor lose
        self.loss_lose = 20.0 # loss when lose

        self.reward_win = 30.0 # reward when win

        self.battery_left = 10.0 # total battery energy
        self.battery_spend = 1.0 # battery energy loss per move

        self.state = self.state_lose
        while (
            self.state in self.barriers or
            self.state in [self.state_lose, self.state_win]):
            self.state = np.random.choice(16)

        self.state = [self.state, 0]

        self.ended = 0

    def act(self, action):
        assert self.ended == 0 # ensure no playing after game is over
        assert self.state[0] not in self.barriers
        assert self.state[0] not in [self.state_win, self.state_lose]

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
        if real_action == 0:
            if self.state[0] in self.state_barriers_0:
                to_sum = 0
            else:
                to_sum = -4
        elif real_action == 1:
            if self.state[0] in self.state_barriers_1:
                to_sum = 0
            else:
                to_sum = 1
        elif real_action == 2:
            if self.state[0] in self.state_barriers_2:
                to_sum = 0
            else:
                to_sum = 4
        elif real_action == 3:
            if self.state[0] in self.state_barriers_3:
                to_sum = 0
            else:
                to_sum = -1
        else:
            raise ValueError("Invalid real_action", real_action)

        self.state = self.state.copy()
        self.state[0] += to_sum
        self.state[1] += 1

        if self.state[0] == self.state_lose:
            return - self.loss_lose - self.loss_spendbat, 1
        elif self.state[0] == self.state_win:
            return self.reward_win - self.loss_spendbat, 1
        elif self.battery_left <= self.battery_spend:
            return - self.loss_lose - self.loss_spendbat, 1

        self.battery_left -= self.battery_spend

        return - self.loss_spendbat, 0

action_value = np.zeros((16, 13, 4))
action_value_frozen = action_value.copy()
def decide(state, action_value):
    actions_log_weights = action_value[state[0], state[1]]
    actions_weights = softmax(actions_log_weights)
    actions_weights += .4
    actions_weights /= 2.6
    action = np.random.choice(4, p=actions_weights)
    return action

nepisodes = 20000
lrs = np.geomspace(2, 1.1, nepisodes)-1
refroze = 0
for episode in range(nepisodes):
    ended = 0
    game = Game()
    lr = lrs[episode]

    while not ended:
        cur_state = game.state
        #action = decide(cur_state, action_value)
        action = np.random.choice(4)
        new_state, reward, ended = game.act(action)

        action_value[cur_state[0], cur_state[1], action] = (
            (1 - lr) *
            action_value[cur_state[0], cur_state[1], action] +
            lr * (reward + 0.99 *
                np.max(action_value_frozen[new_state[0], new_state[1]]))
        )
        if refroze >= 20:
            action_value_frozen = action_value.copy()
            refroze = 0
        else:
            refroze += 1

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

np.set_printoptions(formatter=dict(float=formatter))

for time in range(2):
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

        print("-----------------------------------------------------------")

    policy = np.argmax(action_value_f, 1).reshape((4,4))
    policy = np.array(policy, dtype=str)
    policy[policy == '0'] = " up  "
    policy[policy == '1'] = "right"
    policy[policy == '2'] = "down "
    policy[policy == '3'] = "left "
    print(policy)
