import numpy as np
import gymnasium as gym
import os
import time
import pygame


# --- ЧАТ-БОТ ---
def predict_bot_response(message):
    return "Система готова к анализу новых данных!"


# --- CARTPOLE ---
def train_cartpole_minimal(episodes=50):
    env = gym.make("CartPole-v1", render_mode="human")
    bins = [20, 20, 20, 20]
    obs_low = np.array([-4.8, -3.5, -0.42, -3.5])
    obs_high = np.array([4.8, 3.5, 0.42, 3.5])
    path = 'data/q_table_cp.npy'
    if not os.path.exists('data'): os.makedirs('data')
    q_table = np.load(path) if os.path.exists(path) else np.zeros(bins + [env.action_space.n])
    max_reward = 0
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            pygame.event.pump()
            env.render()
            state_adj = (np.clip(state, obs_low, obs_high) - obs_low) / (obs_high - obs_low) * (np.array(bins) - 1)
            discrete_state = tuple(state_adj.astype(int))
            action = np.argmax(q_table[discrete_state])
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc
        if ep_reward > max_reward:
            max_reward = ep_reward
    np.save(path, q_table)
    env.close()
    return int(max_reward)
# --- MAZE ---
def train_maze_minimal(episodes=50):
    SIZE = 5
    CELL = 80
    pygame.init()
    screen = pygame.display.set_mode((SIZE * CELL, SIZE * CELL))
    walls = [[1, 1], [2, 2], [3, 3], [0, 4]]
    path = 'data/q_table_maze.npy'
    if not os.path.exists('data'): os.makedirs('data')
    if os.path.exists(path):
        q_table = np.load(path)
        if q_table.shape[0] != SIZE: q_table = np.zeros((SIZE, SIZE, 4))
    else:
        q_table = np.zeros((SIZE, SIZE, 4))
    max_reward = -999
    for ep in range(episodes):
        state = [0, 0]
        total_r = 0
        for _ in range(40):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return 0
            old_s = tuple(state)
            action = np.argmax(q_table[old_s]) if np.random.random() > 0.05 else np.random.randint(4)
            new_pos = list(state)
            if action == 0 and state[0] > 0:
                new_pos[0] -= 1
            elif action == 1 and state[0] < SIZE - 1:
                new_pos[0] += 1
            elif action == 2 and state[1] > 0:
                new_pos[1] -= 1
            elif action == 3 and state[1] < SIZE - 1:
                new_pos[1] += 1
            reward = -5 if new_pos in walls else (100 if new_pos == [SIZE - 1, SIZE - 1] else -1)
            if new_pos not in walls: state = new_pos
            screen.fill((20, 20, 25))
            for r in range(SIZE):
                for c in range(SIZE):
                    color = (220, 60, 60) if [r, c] in walls else (50, 50, 50)
                    pygame.draw.rect(screen, color, (c * CELL, r * CELL, CELL, CELL), 1 if [r, c] not in walls else 0)
            pygame.draw.rect(screen, (0, 255, 100), ((SIZE - 1) * CELL, (SIZE - 1) * CELL, CELL, CELL))
            pygame.draw.circle(screen, (0, 180, 255), (state[1] * CELL + CELL // 2, state[0] * CELL + CELL // 2), 20)
            pygame.display.flip()
            time.sleep(0.01)
            new_s = tuple(state)
            q_table[old_s][action] += 0.2 * (reward + 0.9 * np.max(q_table[new_s]) - q_table[old_s][action])
            total_r += reward
            if state == [SIZE - 1, SIZE - 1]: break
        if total_r > max_reward:
            max_reward = total_r
    pygame.quit()
    np.save(path, q_table)
    return int(max_reward)