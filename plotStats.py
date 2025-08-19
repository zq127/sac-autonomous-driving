import os
import numpy as np
import matplotlib.pyplot as plt

file_name = ('SAC6_training_stats_e200__1754581622.npy')
distance = True
file_path = os.path.join('sac_logs', file_name)
training_stats = np.load(file_path)
reward_avg = [stat[0] for stat in training_stats]
reward_min = [stat[1] for stat in training_stats]
reward_max = [stat[2] for stat in training_stats]
avg_ep_length = [stat[3] for stat in training_stats]
collision_pct = [stat[4] for stat in training_stats]
top_speed = [stat[5] for stat in training_stats]
avg_speed = [stat[6] for stat in training_stats]
if distance:
    avg_distance = [stat[7] for stat in training_stats]
else:
    avg_distance = np.zeros(len(training_stats))
# 假设 training_stats 存储了所有训练统计数据
#critic_1_loss = [stat[8] for stat in training_stats]  # critic_1_loss 保存在 training_stats 中的第 8 列
#critic_2_loss = [stat[9] for stat in training_stats]  # critic_2_loss 保存在 training_stats 中的第 9 列
#actor_loss = [stat[10] for stat in training_stats]    # actor_loss 保存在第 10 列，如果存储了的话

x = np.arange(len(training_stats) * 10, step=10)

plt.plot(x, reward_avg, 'b', label='average')
plt.plot(x, reward_min, 'g', label='min')
plt.plot(x, reward_max, 'r', label='max')
plt.title('reward over episodes')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend()
plt.show()

plt.plot(x, avg_ep_length, label='avg episode length')
plt.legend()
plt.show()

plt.plot(x, collision_pct, label='collision percent')
plt.legend()
plt.title('collision percentage over episodes')
plt.xlabel('episodes')
plt.ylabel('percentage of episodes ending in collision')
plt.show()

plt.plot(x, top_speed, 'r', label='top speed')
plt.plot(x, avg_speed, 'b', label='average speed')
plt.title('speed over episodes')
plt.xlabel('episodes')
plt.ylabel('speed in kph')
plt.legend()
plt.show()







print(f'Minimum reward: {reward_min[:100]}')
print(f'Average reward: {reward_avg[:100]}')
print(f'Max reward: {reward_max[:]}')
print(f'Top speed: {top_speed[:100]}')
print(f'Average speed: {avg_speed[:100]}')
print(f'Average distance: {avg_distance[:100]}')

