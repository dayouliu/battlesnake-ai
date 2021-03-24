from collections import deque
import random
import time
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

class Cell(object):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    SNAKE = 3
    SNAKEHEAD = 4

class Action(object):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    HORIZONTAL = [LEFT, RIGHT]
    VERTICAL = [UP, DOWN]

class Reward(object):
    ALIVE = -0.1
    FOOD = 5
    DEAD = -100
    WON = 100

class Snake:
    def __init__(self, sid, head):
        self.sid = sid
        self.body = deque()
        self.add(head)
        self.head = head
        self.prev_action = Action.UP
        self.grow = True
    
    def add(self, cell):
        self.body.appendleft(cell)
        self.head = cell

    def remove(self):
        return self.body.pop()

class Game(object):
    def __init__(self, rows, cols, spawns):
        self.rows = rows
        self.cols = cols
        self.grid = [[Cell.EMPTY for j in range(cols)] for i in range(rows)]
        
        self.snakes = {}
        self.snake_count = len(spawns)
        for i in range(self.snake_count):
            self.snake_add(i+1, spawns[i])

        self.gen_food(self.get_empty())
        self.gen_food(self.get_empty())
        self.gen_food(self.get_empty())
        self.gen_food(self.get_empty())
        self.gen_food(self.get_empty())
        self.gen_food(self.get_empty())
        self.gen_food(self.get_empty())

        self.score = 0

    def get_cell(self, cell):
        return self.grid[cell[0]][cell[1]]
    
    def set_cell(self, cell, val):
        self.grid[cell[0]][cell[1]] = val
    
    def validate_inbounds(self, cell):
        return 0 <= cell[0] and cell[0] < self.rows and 0 <= cell[1] and cell[1] < self.cols
    
    def validate_action(self, snake, action):
        if len(snake.body) == 1: return True
        if action == snake.prev_action: return True
        valid_vertical = action in Action.VERTICAL and snake.prev_action in Action.HORIZONTAL
        valid_horizontal = action in Action.HORIZONTAL and snake.prev_action in Action.VERTICAL
        return valid_vertical or valid_horizontal

    def snake_add(self, sid, head):
        snake = Snake(sid, head)
        self.snakes[sid] = snake
        self.set_cell(head, Cell.SNAKEHEAD)

    # returns false if snake eats itself
    # otherwise return true
    def snake_move(self, snake, head):
        self.set_cell(snake.head, Cell.SNAKE)
        if not snake.grow:
            tail = snake.remove()
            self.set_cell(tail, Cell.EMPTY)
        else:
            snake.grow = False
        
        if self.get_cell(head) == Cell.SNAKE:
            return False
        
        if self.get_cell(head) == Cell.FOOD:
            snake.grow = True
        
        snake.add(head)
        self.set_cell(head, Cell.SNAKEHEAD)
        return True

    def snake_head_next(self, snake, action):
        r, c = snake.head
        if action == Action.LEFT:
            return (r - 1, c)
        elif action == Action.RIGHT:
            return (r + 1, c)
        elif action == Action.UP:
            return (r, c + 1)
        elif action == Action.DOWN:
            return (r, c - 1)
    
    def get_empty(self):
        empty = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                if self.get_cell(cell) == Cell.EMPTY:
                    empty.append((r, c))
        return empty
    
    def gen_food(self, empty):
        cell = random.sample(empty, 1)[0]
        self.set_cell(cell, Cell.FOOD)

    def step(self, action):
        snake = self.snakes[1]

        # invalid move reverts to prev move
        if not self.validate_action(snake, action): 
            action = snake.prev_action
        snake.prev_action = action

        snake_head_next = self.snake_head_next(snake, action)

        # snake out of bounds
        if not self.validate_inbounds(snake_head_next):
            return Reward.DEAD

        snake_head_next_cell = self.get_cell(snake_head_next)
        
        # snake hits wall
        if snake_head_next_cell in [Cell.WALL]:
            return Reward.DEAD
        
        snake_move_valid = self.snake_move(snake, snake_head_next)
        
        # snake ate itself
        if not snake_move_valid:
            return Reward.DEAD

        cells_empty = self.get_empty()

        # snake won
        if len(cells_empty) == 0:
            return Reward.WON

        # snake eat food
        if snake_head_next_cell == Cell.FOOD:
            self.gen_food(cells_empty)
            self.score += 1
            return Reward.FOOD
        
        # existence is pain
        return Reward.ALIVE


class BattleSnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # TODO: define observation_space
    def __init__(self):
        self.rows = 11
        self.cols = 11
        self.spawns = [(self.rows//2, self.cols//2)]

        self.action_space = spaces.Discrete(3)
        self.observation_space = self.gen_observation_space()

        self.game = Game(self.rows, self.cols, self.spawns)
        self.viewer = None

    # TODO: define observation, info
    def step(self, action):
        reward = self.game.step(action)
        score = self.game.score
        done = reward in [Reward.DEAD, Reward.WON]
        obs = self.get_state()
        info = {}

        return obs, reward, score, done, info

    # TODO: define observation
    def reset(self):
        self.game = Game(self.rows, self.cols ,self.spawns)
        self.observation_space = self.gen_observation_space()
        return self.get_state()
    
    def get_state(self):
        return self.game.grid.copy()

    def render(self, mode='human', close=False):
        width = 600
        height = 600
        len = width / self.game.rows

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        for r in range(self.game.rows):
            for c in range(self.game.cols):
                left = c * len
                right = (c+1) * len
                top = height - (r+1) * len
                bot = height - r * len
                square = rendering.FilledPolygon([(left,bot), (left,top), (right,top), (right,bot)])
                if self.game.get_cell((r, c)) != Cell.EMPTY:
                    if self.game.get_cell((r, c)) == Cell.SNAKE:
                        square.set_color(0, 0, 0)
                    elif self.game.get_cell((r, c)) == Cell.SNAKEHEAD:
                        square.set_color(0, 1, 0)
                    elif self.game.get_cell((r, c)) == Cell.FOOD:
                        square.set_color(1, 0, 0)
                    self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def seed(self):
        pass
    
    def gen_observation_space(self):
        return spaces.Box( 
                low=0, 
                high=5,
                shape=(self.rows,self.cols),
                dtype=np.uint8)

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer(object):
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Q
        pred = self.model(state)
        
        # Q_new = R + gamma * max(Q)
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[idx] + self.gamma * torch.max(self.mode(next_state[idx]))
            
            target[i][torch.argmax(action).item()] = Q_new  


class Agent(object):
    def __init__(self):
        self.games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11*11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, env):
        grid = env.game.grid.copy()
        state = []
        for row in grid: 
            for i in row: 
                state.append(i)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        states, actions,  rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = 0
    total_score = 0
    record = 0
    agent = Agent()
    env = BattleSnakeEnv()

    while True:
        state_old = agent.get_state(evn)
        action = agent.get_action(state_old)

        state_new, reward, done, info = env.step(action)
        state_new = agent.get_state(state_new)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            env.reset()
            agent.games += 1
            agent.train_short_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print("game: record: {} score: {}".format(record, score))

env = BattleSnakeEnv()
agent = Agent()
print(agent.get_state(env))

'''
env = BattleSnakeEnv()
for i in range(1):
    env.reset()
    for t in range(100):
        time.sleep(0.5) 
        env.render()

        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        for i in range(len(state)):
            print(" ".join([str(x) for x in state[i]]))
        print()
        if done:
            print('episode: {} | timesteps: {}'.format(i, t))
            break

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = BattleSnakeEnv()

model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=200000)

# Enjoy trained agent
for i in range(100):
    obs = env.reset()
    env.render()
    for t in range(1000):
        time.sleep(0.5) 
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print(reward)
        if done:
            print('episode: {} | timesteps: {}'.format(i, t))
            break
'''
