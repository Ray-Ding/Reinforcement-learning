import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay       #奖励衰减
        self.epsilon = e_greedy         #随机行为概率
        self.q_table = pd.DataFrame(columns = self.actions,dtype=np.float64)     #建立Q表

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform()<self.epsilon:           #小于则取最q表最大值行为,没有最大值就随机选择
            state_action = self.q_table.loc[observation,:]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)      #检查下一步的状态是否存在
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_,:].max()
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)       #更新为target-predict的值

