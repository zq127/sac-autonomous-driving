import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] TensorFlow GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
from settings import *
# sacagent.py 顶部
#print("[DEBUG] sacagent module path:", __file__)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [np.array(x) for x in zip(*[self.buffer[i] for i in indices])]

class SACAgent:
    
    def training_initialized(self):
        return self.actor is not None and self.critic_1 is not None and self.critic_2 is not None

    def get_qs(self, critic_model, state, action):
        # 兼容 (64,64,3) 或 (1,64,64,3) 等
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        if len(state.shape) == 3:  # (64,64,3)
            state = np.expand_dims(state, 0)
        if len(action.shape) == 1:  # (3,)
            action = np.expand_dims(action, 0)
        q_value = critic_model([state, action])
        return q_value.numpy()[0]


    def train_in_loop(self):
        self.terminate = False
        self.training_initialized = True
        print("[INFO] Training loop started, waiting for enough experiences...")
        while not self.terminate:
            if len(self.buffer.buffer) < TRAINING_BATCH_SIZE:
                time.sleep(0.01)
                continue
            self.train_step()

    def update_replay_memory(self, transition):
        self.buffer.add(transition)

    def __init__(self, new=True, tensorboard=None):
        self.state_shape = (IM_HEIGHT, IM_WIDTH, 3)
        self.action_shape = 3
        self.buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = ALPHA
        self.training_stats = np.empty((0, 8))  # 假设你每次统计 8 个指标
        self.tensorboard = tensorboard


        self.actor = self.build_actor()
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.target_critic_1 = self.build_critic()
        self.target_critic_2 = self.build_critic()

        self.actor_optimizer = Adam(ACTOR_LR)
        self.critic_1_optimizer = Adam(CRITIC_LR)
        self.critic_2_optimizer = Adam(CRITIC_LR)

        if not new:
            self.load_models()

        self.update_target(self.target_critic_1, self.critic_1)
        self.update_target(self.target_critic_2, self.critic_2)
    

    def train_step(self):
    # 1. 经验采样
        states, actions, rewards, next_states, dones = self.buffer.sample(TRAINING_BATCH_SIZE)
        # 归一化输入
        states = states.astype(np.float32) / 255.0
        next_states = next_states.astype(np.float32) / 255.0
        actions = actions.astype(np.float32)
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)

    # 2. 更新 critic 网络
        with tf.GradientTape(persistent=True) as tape:
            # 下一个状态的动作
            next_mean, next_log_std = self.actor(next_states)
            next_log_std = tf.clip_by_value(next_log_std, -20, 2)
            next_std = tf.exp(next_log_std)
            next_normal = tfp.distributions.Normal(next_mean, next_std)
            next_action = tf.tanh(next_normal.sample())
            next_log_prob = next_normal.log_prob(next_action)
            next_log_prob = tf.reduce_sum(next_log_prob, axis=1, keepdims=True)
            next_log_prob -= tf.reduce_sum(tf.math.log(1 - tf.square(next_action) + 1e-6), axis=1, keepdims=True)
            
            target_q1 = self.target_critic_1([next_states, next_action])
            target_q2 = self.target_critic_2([next_states, next_action])
            target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * self.gamma * target_q

            # 当前 Q
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])

            # Critic 损失
            critic_1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic_2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        del tape

    # 3. 更新 actor 网络
        with tf.GradientTape() as tape2:
            mean, log_std = self.actor(states)
            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)
            normal = tfp.distributions.Normal(mean, std)
            x_t = normal.sample()
            y_t = tf.tanh(x_t)
            log_prob = normal.log_prob(x_t)
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            log_prob -= tf.reduce_sum(tf.math.log(1 - tf.square(y_t) + 1e-6), axis=1, keepdims=True)
            q1 = self.critic_1([states, y_t])
            q2 = self.critic_2([states, y_t])
            q = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(self.alpha * log_prob - q)

        actor_grads = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        if self.tensorboard:
            self.tensorboard.update_stats(
                actor_loss=float(actor_loss.numpy()),
                critic_1_loss=float(critic_1_loss.numpy()),
                critic_2_loss=float(critic_2_loss.numpy()),
            )

        # 4. 软更新目标网络
        self.update_target(self.target_critic_1, self.critic_1)
        self.update_target(self.target_critic_2, self.critic_2)

        #return critic_1_loss.numpy(), critic_2_loss.numpy(), actor_loss.numpy()
    # 5. 可选：输出调试信息
    # print(f"[TRAIN] actor_loss={actor_loss.numpy():.4f}, critic1_loss={critic_1_loss.numpy():.4f}, critic2_loss={critic_2_loss.numpy():.4f}")


    def build_actor(self):
        inputs = Input(shape=self.state_shape)
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        mean = Dense(self.action_shape)(x)
        log_std = Dense(self.action_shape)(x)
        return Model(inputs, [mean, log_std])

    def build_critic(self):
        state_input = Input(shape=self.state_shape)
        action_input = Input(shape=(self.action_shape,))
        x1 = tf.keras.layers.Rescaling(1./255)(state_input)
        x1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x = Concatenate()([x1, action_input])
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        q_value = Dense(1)(x)
        return Model([state_input, action_input], q_value)

    def update_target(self, target_model, source_model):
        for target_weights, source_weights in zip(target_model.weights, source_model.weights):
            target_weights.assign(self.tau * source_weights + (1 - self.tau) * target_weights)

    def get_action(self, state):
        """
        兼容 (64,64,3) 或 (1,64,64,3) 的输入。如果是 (batch,64,64,3)，只取 batch[0]。
        """
        state = np.array(state, dtype=np.float32)
        # 如果输入是单帧
        if state.shape == (64, 64, 3):
            state_b = np.expand_dims(state, 0)  # (1,64,64,3)
        # 如果输入是 batch=1
        elif state.shape == (1, 64, 64, 3):
            state_b = state
        # 如果是更大 batch（如用于调试），只取第一帧
        elif len(state.shape) == 4 and state.shape[0] > 1:
            state_b = state[:1]
        else:
            raise ValueError(f"[get_action] Unexpected state shape: {state.shape}")
        mean, log_std = self.actor(state_b)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)
        normal_dist = tfp.distributions.Normal(mean, std)
        sample = normal_dist.sample()
        y_t = tf.tanh(sample)
        action = y_t.numpy()[0].flatten()  # (3,)
        return action










    def save_models(self):
        self.actor.save(ACTOR_TO_LOAD)
        self.critic_1.save(CRITIC1_TO_LOAD)
        self.critic_2.save(CRITIC2_TO_LOAD)

    def load_models(self):
        if os.path.exists(ACTOR_TO_LOAD):
            self.actor = tf.keras.models.load_model(ACTOR_TO_LOAD)
        if os.path.exists(CRITIC1_TO_LOAD):
            self.critic_1 = tf.keras.models.load_model(CRITIC1_TO_LOAD)
        if os.path.exists(CRITIC2_TO_LOAD):
            self.critic_2 = tf.keras.models.load_model(CRITIC2_TO_LOAD)
