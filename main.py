import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

#Constants

learning_rate = 0.1
right_reward = 1
wrong_reward = 1
discount_factor = 0.95
epsilon = 0.5
epsilon_decay = 0.9998
number_of_episodes = 200000
number_of_actions = 3
time_steps= 15



# There are three actions : 1 ,2 and 3
# In each time step, the agent should take the correct action
# Number of correct action is (time_step % 3)
# In t=1 > A=1, t=3(0) > A=3, t=5 > A=2, t=8 > A=2 . . .

class  agent:
    def __init__(self):

        self.selected_action = []
        self.action_list = []

    def action(self, action_number):

        if action_number == 0:
            self.selected_action = 1
            self.action_list.append(self.selected_action)
        if action_number == 1:
            self.selected_action = 2
            self.action_list.append(self.selected_action)
        if action_number == 2:
            self.selected_action = 3
            self.action_list.append(self.selected_action)
        return (self.selected_action)


q_table = np.zeros((time_steps, number_of_actions))
for i in range(0, time_steps):
    for j in range (0 , number_of_actions):
        q_table[i][j] = np.random.uniform(1, 5)


total_reward = []
n = 0
avg = 0
new_moving_avg = []
for episode in trange(number_of_episodes):

    episode_reward = 0
    player = agent()

    #exploration and exploitation
    for i in range(time_steps):
        ts = i
        if np.random.random() > epsilon:
            action = np.argmax(q_table[ts])
        else:
            action = np.random.randint(0, 3)
        #rewards
        player.action(action)
        if player.selected_action % 3  == ts % 3:
            episode_reward += right_reward
        else:
            episode_reward -= wrong_reward
         #q_table update
        if ts == time_steps-1:
            max_future_q = 0
        else:
             new_ts = (ts+1)
             max_future_q = np.max(q_table[new_ts])
             current_q = q_table[ts][action]

        new_q = (1 - learning_rate) * current_q + learning_rate * (episode_reward+ discount_factor * max_future_q)
        q_table[ts][action] = new_q

    total_reward.append(episode_reward)
    epsilon *= epsilon_decay

    n += 1
    avg = avg + (episode_reward-avg)/n
    new_moving_avg.append(avg)

print (q_table)
print(player.action_list)

SHOW_EVERY=100
#moving_avg = np.convolve(total_reward, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
#plt.plot([i for i in range(len(moving_avg))], moving_avg)
#plt.ylabel(f"Reward {SHOW_EVERY}ma")
#plt.xlabel("episode #")
#plt.show()



moving_avg = np.convolve(total_reward, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot([kk for kk in range(len(new_moving_avg))], new_moving_avg)
plt.plot([k for k in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Average Reward every {SHOW_EVERY} tiem step")
plt.xlabel("episode #")
plt.show()

