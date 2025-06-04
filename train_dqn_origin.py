import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.flappy_bird import FlappyBird
from src.utils import pre_processing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(512, 2))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return self.fc2(output)

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=20000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000)
    parser.add_argument("--log_path", type=str, default="tensorboard_dqn")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    return parser.parse_args()

def train(opt):
    torch.manual_seed(123)

    model = DeepQNetwork().to(device)
    target_q_net = DeepQNetwork().to(device)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    game_state = FlappyBird("dpn")
    state, reward, terminal = game_state.step(0)

    replay_memory = []
    iter = 0
    max_reward = 0

    while iter < opt.num_iters:
        prediction = model(state)
        epsilon = opt.final_epsilon + ((opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            action = randint(0, 1)
        else:
            action = prediction.argmax().item()

        next_state, reward, terminal = game_state.step(action)
        replay_memory.append([state, action, reward, next_state, terminal])
        state = next_state

        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]

        loss = 0
        if len(replay_memory) > opt.batch_size:
            batch = sample(replay_memory, opt.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

            states = torch.cat(state_batch, dim=0).to(device)
            actions = torch.tensor(action_batch).view(-1, 1).to(device)
            rewards = torch.tensor(reward_batch).view(-1, 1).to(device)
            dones = torch.tensor(terminal_batch).view(-1, 1).int().to(device)
            next_states = torch.cat(next_state_batch, dim=0).to(device)

            q_values = model(states).gather(1, actions)
            max_next_q_values = target_q_net(next_states).gather(1, actions).max(1)[0].view(-1, 1)
            q_targets = rewards + opt.gamma * max_next_q_values * (1 - dones)

            optimizer.zero_grad()
            loss = criterion(q_values, q_targets)
            loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                target_q_net.load_state_dict(model.state_dict())

        if reward > max_reward:
            max_reward = reward
            print(f"max_reward Iteration: {iter + 1}/{opt.num_iters}, Action: {action}, Loss: {loss}, Epsilon: {epsilon:.4f}, Reward: {reward}, Q-value: {torch.max(prediction).item():.4f}")

        if iter % 1000 == 0:
            print(f"Iteration: {iter + 1}/{opt.num_iters}, Action: {action}, Loss: {loss}, Epsilon: {epsilon:.4f}, Reward: {reward}, Q-value: {torch.max(prediction).item():.4f}")

        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction).item(), iter)

        if (iter + 1) % 5000 == 0:
            torch.save(model.state_dict(), f"{opt.saved_path}/flappy_bird_origin_dqn_{iter+1}.pth")

        iter += 1

    torch.save(model.state_dict(), f"{opt.saved_path}/flappy_bird_origin_dqn_final.pth")

if __name__ == "__main__":
    opt = get_args()
    train(opt)
