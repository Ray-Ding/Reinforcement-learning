from Maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(10):
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
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()