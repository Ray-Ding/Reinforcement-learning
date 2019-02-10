from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(150):
        obersvation = env.reset()   #环境观测

        while True:
            env.render()    #更新环境
            action = RL.choose_action(str(obersvation))     #选择行为
            obersvation_,reward,done = env.step(action)     #选择行为后的反馈
            RL.learn(str(obersvation),action,reward,str(obersvation_))      #得分进行RL学习
            obersvation = obersvation_

            if done:
                break

    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()