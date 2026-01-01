#!/usr/bin/env python
# coding: utf-8

# # Snake 10x10

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gymnasium as gym
import gym_snakegame
from scipy.signal import step
import pygame
import sys

# seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch device:", device)


# ## Test the gymnasium env

# In[2]:


PLAY_MANUAL = True  # set to False if u wanna see random moves
env = gym.make(
    "gym_snakegame/SnakeGame-v0",
    board_size=5,
    n_channel=1,
    n_target=1,
    render_mode="human"
)

obs, info = env.reset()

KEY_MAP = {
    pygame.K_UP: 2,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 0,
    pygame.K_LEFT: 3
}

total_reward = 0
steps = 0

print("hit arrows to play! or close window to quit")
if not PLAY_MANUAL:
    print("auto mode: doing 250 random steps...")

try:
    while True:
        action = None

        if PLAY_MANUAL:
            # wait for key press
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                if event.type == pygame.KEYDOWN:
                    if event.key in KEY_MAP:
                        action = KEY_MAP[event.key]
            if action is None:
                continue  # wait till key pressed
        else:
            if steps >= 250:
                break
            action = env.action_space.sample()
            steps += 1

        # step
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward

        print(f"step {steps}/100, score: {total_reward} ({reward} | trunc:{truncated} | trmt:{terminated})")

        if terminated or truncated:
            print(f"GameOver!  total score: {total_reward}")
            # reset everything
            obs, info = env.reset()
            total_reward = 0
            steps = 0

except KeyboardInterrupt:
    pass
finally:
    env.close()


# ## Model

# In[3]:


class SnakeV0(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(SnakeV0, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ## Replay buffer

# In[4]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ## Agent

# In[5]:


class DQNAgent:
    def __init__(self, modelclass, state_size, action_size, lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=64, target_update=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Q-network and target network
        self.q_net = modelclass(state_size, action_size).to(device)
        self.target_net = modelclass(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Sync target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayBuffer(buffer_size)
        self.step_count = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # current q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # next q-values frm target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ## Training loop

# In[8]:


# TODO: make a common reward func

def train_dqn(modelclass, board_size=10, env_name="gym_snakegame/SnakeGame-v0", episodes=2000, max_steps=100):
    env = gym.make(env_name,
                   board_size=board_size,
                   n_channel=1,
                   n_target=1,
                   render_mode=None)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    agent = DQNAgent(modelclass=modelclass, state_size=state_size, action_size=action_size)

    scores = deque(maxlen=100)  # for moving average
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    epsilon = epsilon_start

    print("Starting training...")
    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0

        for t in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # rewards calculations
            if done and reward == 0:      # died without eating
                reward = -1.0
            # reward = reward - 0.01  # optional shaping to encourage speed

            next_state = next_state.flatten()
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Avg Reward (last 100): {avg_score:.3f}, Epsilon: {epsilon:.3f}")

    env.close()
    return agent, env


def evaluate_agent(agent, env, episodes=10, max_steps=100):
    success = 0
    for _ in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        for _ in range(max_steps):
            action = agent.act(state, epsilon=0.0)  # greedy
            state, reward, terminated, truncated, _ = env.step(action)
            state = state.flatten()
            done = terminated or truncated

            # rewards calculations
            if terminated: # died
                reward = -10.0
            if done and reward == 0:      # died without eating
                reward = -1.0
            # reward = reward - 0.01  # optional shaping to encourage speed

            if done:
                success += reward
                break
    print(f"\nSuccess rate over {episodes} episodes: {success}/{episodes} ({100 * success / episodes:.1f}%)")


# In[9]:


agent, env = train_dqn(modelclass=SnakeV0, episodes=20_000)


# In[41]:


eval_env = gym.make("gym_snakegame/SnakeGame-v0",
                    board_size=10,
                    n_channel=1,
                    n_target=1,
                    render_mode="human")
evaluate_agent(agent, eval_env, episodes=200)
eval_env.close()


# In[ ]:





# In[ ]:


# .astype(np.float32)



# ```
# # TODO: make a common reward func
# 
# def train_dqn(modelclass, board_size=10, env_name="gym_snakegame/SnakeGame-v0", episodes=2000, max_steps=100):
#     env = gym.make(env_name,
#                    board_size=board_size,
#                    n_channel=1,
#                    n_target=1,
#                    render_mode=None)
#     state_size = np.prod(env.observation_space.shape)
#     action_size = env.action_space.n
# 
#     agent = DQNAgent(modelclass=modelclass, state_size=state_size, action_size=action_size)
# 
#     scores = deque(maxlen=100)  # for moving average
#     epsilon_start = 1.0
#     epsilon_end = 0.01
#     epsilon_decay = 0.995
# 
#     epsilon = epsilon_start
# 
#     print("Starting training...")
#     for episode in range(episodes):
#         state, _ = env.reset()
#         state = state.flatten()
#         total_reward = 0
# 
#         for t in range(max_steps):
#             action = agent.act(state, epsilon)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
# 
#             # rewards calculations
#             if done and reward == 0:      # died without eating
#                 reward = -1.0
#             # reward = reward - 0.01  # optional shaping to encourage speed
# 
#             next_state = next_state.flatten()
#             agent.remember(state, action, reward, next_state, done)
#             agent.replay()
# 
#             state = next_state
#             total_reward += reward
# 
#             if done:
#                 break
# 
#         scores.append(total_reward)
#         epsilon = max(epsilon_end, epsilon_decay * epsilon)
# 
#         if episode % 100 == 0:
#             avg_score = np.mean(scores)
#             print(f"Episode {episode}, Avg Reward (last 100): {avg_score:.3f}, Epsilon: {epsilon:.3f}")
# 
#     env.close()
#     return agent, env
# 
# 
# def evaluate_agent(agent, env, episodes=10, max_steps=100):
#     success = 0
#     for _ in range(episodes):
#         state, _ = env.reset()
#         state = state.flatten()
#         for _ in range(max_steps):
#             action = agent.act(state, epsilon=0.0)  # greedy
#             state, reward, terminated, truncated, _ = env.step(action)
#             state = state.flatten()
#             done = terminated or truncated
# 
#             # rewards calculations
#             if terminated: # died
#                 reward = -10.0
#             if done and reward == 0:      # died without eating
#                 reward = -1.0
#             # reward = reward - 0.01  # optional shaping to encourage speed
# 
#             if done:
#                 success += reward
#                 break
#     print(f"\nSuccess rate over {episodes} episodes: {success}/{episodes} ({100 * success / episodes:.1f}%)")
# ```
# 
# make a common reward function asnd use it in both places
# make the reward function such that
# - when the snake body filles the whole board -> reward very high (as it filled the whole board it must have eaten all items so i dont think theres need to check score)
# - when snake hits wall -> very heavy punishment
# - when movign slow -> punishment
# - when in loop and does nothing -> mid punishment
# - when eating item -> (low/mid/high suggest which ones best) reward
# -

# In[4]:





# In[5]:


results

