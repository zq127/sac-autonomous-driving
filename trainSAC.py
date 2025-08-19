import os
import time
import random
import numpy as np
import tensorflow as tf
from threading import Thread
from tqdm import tqdm

from settings import *
from sacCarEnv import CarEnv
from sacagent import SACAgent
#from modifiedTB import ModifiedTensorBoard
from modifiedTB import ModifiedTensorBoard


if __name__ == '__main__':
    ep_rewards = []
    ep_length = []
    ep_distance = []
    collision_hist = []
    max_speed = []
    avg_speed = []
    critic_1_losses = []  # List to store critic 1 losses
    critic_2_losses = []  # List to store critic 2 losses
    actor_losses = []  # List to store actor losses

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)  # ✅ 兼容 TF 2.10

    # 创建训练所需目录
    os.makedirs('sac_models', exist_ok=True)
    os.makedirs('sac_logs', exist_ok=True)

    print("[INFO] Initializing environment and agent...")
    env = CarEnv()
    tb = ModifiedTensorBoard()
    agent = SACAgent(new=(STARTING_EPISODE == 1), tensorboard=tb)
    #tensorboard_callback = ModifiedTensorBoard(log_dir='./logs')

    # 后台训练线程（异步）
    #trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    #trainer_thread.start()

    #while not agent.training_initialized:
    #    time.sleep(0.01)

    # === warm‑up：先随机采几百步经验，让经验池里有“车会动”的数据 ===
    print("[INFO] Collecting warm‑up experience ...")
    state = env.reset()
    for _ in range(200):
        rand_action = np.array([
            np.random.uniform(-0.2, 1.0),      # throttle_raw  →  0.4~1.0
            np.random.normal(0.0, 0.3),        # steer
            -1.0                               # brake_raw
        ])
        next_state, reward, done, kph, _ = env.step(rand_action)
        agent.update_replay_memory((state, rand_action, reward, next_state, done))
        state = next_state if not done else env.reset()
    print("[INFO] Warm‑up done!")

    # 热启动
    #agent.get_qs(agent.critic_1, np.ones((env.im_height, env.im_width, 3)), np.ones(3))
    dummy_state = np.ones((1, env.im_height, env.im_width, 3), dtype=np.float32)  # (1, 64, 64, 3)
    dummy_action = np.ones((1, 3), dtype=np.float32)  # (1, 3)
    agent.get_qs(agent.critic_1, dummy_state, dummy_action)

    for episode in tqdm(range(STARTING_EPISODE, STARTING_EPISODE + EPISODES), ascii=True, unit='episodes'):
        print(f"[INFO] Episode {episode} started.")

        env.collision_hist = []
        episode_reward = 0
        episode_speed = []
        episode_distance = []
        step = 1
        current_state = env.reset()
        #print("[DEBUG][reset] 返回:", np.shape(current_state))
        #if current_state.shape == (1, 64, 64, 3):
        #    current_state = current_state[0]
        #elif current_state.shape != (64, 64, 3):
        #   raise ValueError(f"[reset] 返回 shape 不对: {current_state.shape}")
        done = False
        episode_start = time.time()

        while not done:
            #print("[DEBUG] current_state.shape:", np.shape(current_state))
            action= agent.get_action(current_state)
            #print("[DEBUG] action.shape:", np.shape(action))
            new_state, reward, done, kph, distance = env.step(action)
            #print("[DEBUG][loop] step 返回 new_state.shape:", np.shape(new_state))
            #if new_state.shape == (1, 64, 64, 3):
            #    new_state = new_state[0]
            #elif new_state.shape != (64, 64, 3):
            #    raise ValueError(f"[step] 返回 shape 不对: {new_state.shape}")

            #critic_1_loss, critic_2_loss, actor_loss = agent.train_step()
            #critic_1_losses.append(critic_1_loss.item())
            #critic_2_losses.append(critic_2_loss.item())
            #actor_losses.append(actor_loss.item())
            #print(f"critic_1_loss shape: {np.array(critic_1_loss).shape}")
            #print(f"critic_2_loss shape: {np.array(critic_2_loss).shape}")
            #print(f"actor_loss shape: {np.array(actor_loss).shape}")

            episode_reward += reward
            episode_speed.append(kph)
            episode_distance.append(distance)

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            if len(agent.buffer.buffer) >= MIN_REPLAY_SIZE and step % 5 == 0:  # 经验池够大才开始学
                #for _ in range(4):
                agent.train_step()

            current_state = new_state
            step += 1

        # 清理 Actor
        for actor in env.actor_list:
            actor.destroy()

        # 记录日志
        episode_end = time.time()
        ep_rewards.append(episode_reward)
        ep_length.append(episode_end - episode_start)
        ep_distance.append(episode_distance[-1])
        max_speed.append(max(episode_speed))
        avg_speed.append(sum(episode_speed) / len(episode_speed))
        collision_hist.append(1 if len(env.collision_hist) > 0 else 0)

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            

            avg_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = np.min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = np.max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            avg_len = np.mean(ep_length[-AGGREGATE_STATS_EVERY:])
            collision_pct = np.mean(collision_hist[-AGGREGATE_STATS_EVERY:])
            top_speed = np.max(max_speed[-AGGREGATE_STATS_EVERY:])
            avg_kph = np.mean(avg_speed[-AGGREGATE_STATS_EVERY:])
            avg_dist = np.mean(ep_distance[-AGGREGATE_STATS_EVERY:])
            #avg_critic_1_loss = np.mean(critic_1_losses[-AGGREGATE_STATS_EVERY:])
            #avg_critic_2_loss = np.mean(critic_2_losses[-AGGREGATE_STATS_EVERY:])
            #avg_actor_loss = np.mean(actor_losses[-AGGREGATE_STATS_EVERY:])
            print(f"[STAT] Ep{episode}: reward={avg_reward:.2f}, max_speed={top_speed:.1f}kph, "
                  f"len={avg_len:.1f}s, collision_rate={collision_pct:.2%}")

            #agent.training_stats = np.vstack((agent.training_stats, [
            #    [avg_reward, min_reward, max_reward, avg_len, collision_pct, top_speed, avg_kph, avg_dist,avg_critic_1_loss, avg_critic_2_loss, avg_actor_loss]
            #]))

            agent.training_stats = np.vstack((agent.training_stats, [
                [avg_reward, min_reward, max_reward, avg_len, collision_pct, top_speed, avg_kph, avg_dist]
            ]))

        # 模型保存
        if not episode % SAVE_EVERY:
            timestamp = int(time.time())
            agent.actor.save(f'sac_models/{MODEL_NAME}_actor_e{episode}__{timestamp}.h5')
            agent.critic_1.save(f'sac_models/{MODEL_NAME}_critic_1_e{episode}__{timestamp}.h5')
            agent.critic_2.save(f'sac_models/{MODEL_NAME}_critic_2_e{episode}__{timestamp}.h5')
            np.save(f'sac_logs/{MODEL_NAME}_training_stats_e{episode}__{timestamp}', agent.training_stats)

    # 最终保存
    agent.terminate = True
    #trainer_thread.join()
    final_episode = STARTING_EPISODE + EPISODES - 1
    agent.actor.save(f'sac_models/{MODEL_NAME}_actor_{final_episode}.h5')
    agent.critic_1.save(f'sac_models/{MODEL_NAME}_critic_1_{final_episode}.h5')
    agent.critic_2.save(f'sac_models/{MODEL_NAME}_critic_2_{final_episode}.h5')
    np.save(f'sac_logs/{MODEL_NAME}_training_stats_e{final_episode}', agent.training_stats)
