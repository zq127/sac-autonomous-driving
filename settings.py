# Settings for training
IM_HEIGHT = 64
IM_WIDTH = 64
SECONDS_PER_EPISODE = 25
REPLAY_MEMORY_SIZE = 20000
MIN_REPLAY_SIZE = 200
MINIBATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
MODEL_NAME = 'SAC6'
SPEED_MAX_REWARD = 1
SPEED_MIN_REWARD = -1
DIST_REWARD = 4
FPS = 30

# === 奖励权重（让速度/进度成为主要动力）===
W_LANE     = 0.8              # (在 sacCarEnv.py 里用到)
W_HEADING  = 0.4
W_SPEED    = 6.0              # 1.0 → 3.0
W_SMOOTH   = 0.5
W_PROGRESS = 6.0              # 2.0 → 4.0

# 折扣因子：未来奖励权重
GAMMA = 0.99  
# 目标网络软更新系数：越小越稳定
TAU = 0.005   
# 熵系数（鼓励探索）
ALPHA = 0.2   

# Actor 网络学习率（策略网络）
ACTOR_LR = 1e-4

# Critic 网络学习率（两个Q网络）
CRITIC_LR = 1e-4

MEMORY_FRACTION = 0.8

STARTING_EPISODE = 1
EPISODES = 200
LEARNING_RATE = 0.001
DISCOUNT = 0.99
ALPHA = 0.2
TAU = 0.005
EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.999
AGGREGATE_STATS_EVERY = 5
SAVE_EVERY = 100

UPDATE_TARGET_EVERY = 50

SHOW_PREVIEW = False
EVAL_SAC = False
#ACTOR_TO_LOAD = r'sac_models\SAC1_actor_e50_1753283849.h5'
#CRITIC1_TO_LOAD = r'sac_models\SAC1_critic_1_e50_1753283849.h5'
#CRITIC2_TO_LOAD = r'sac_models\SAC1_critic_2_e50_1753283849.h5'


ACTOR_TO_LOAD =None
CRITIC1_TO_LOAD = None
CRITIC2_TO_LOAD =None
