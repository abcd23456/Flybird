import argparse
import os
from torch.utils.data import BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.flappy_bird import FlappyBird

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser("Implementation of PPO to play Flappy Bird")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--log_path", type=str, default="tensorboard_ppo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    return parser.parse_args()

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.Tanh())
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(nn.Linear(512, 2))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flat(output)
        output = self.drop(output)
        output = self.fc1(output)
        return nn.functional.softmax(self.fc3(output), dim=1)

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        return self.net(input)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in reversed(td_delta):
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float).to(device)

def train(opt):
    torch.manual_seed(123)

    actor = PolicyNet().to(device)
    critic = ValueNet().to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=opt.lr)

    # 加载模型（断点续训）
    actor_path = os.path.join(opt.saved_path, "flappy_bird_actor")
    if os.path.exists(actor_path):
        checkpoint = torch.load(actor_path, map_location=device)
        actor.load_state_dict(checkpoint['net'])
        actor_optimizer.load_state_dict(checkpoint['optimizer'])
        print("Actor model loaded successfully.")

    critic_path = os.path.join(opt.saved_path, "flappy_bird_critic")
    if os.path.exists(critic_path):
        checkpoint = torch.load(critic_path, map_location=device)
        critic.load_state_dict(checkpoint['net'])
        critic_optimizer.load_state_dict(checkpoint['optimizer'])
        print("Critic model loaded successfully.")

    writer = SummaryWriter(opt.log_path)

    game_state = FlappyBird("ppo")
    state, reward, terminal = game_state.step(0)
    max_reward = 0
    iter = 0
    replay_memory = []
    evaluate_num = 0
    evaluate_rewards = []

    while iter < opt.num_iters:
        terminal = False
        episode_return = 0.0

        while not terminal:
            prediction = actor(state)
            action_dist = torch.distributions.Categorical(prediction)
            action = action_dist.sample().item()
            next_state, reward, terminal = game_state.step(action)
            replay_memory.append([state, action, reward, next_state, terminal])
            state = next_state
            episode_return += reward

        if episode_return > max_reward:
            max_reward = episode_return
            print(f"max_reward Iteration: {iter + 1}/{opt.num_iters}, Reward: {episode_return}")

        if len(replay_memory) > opt.batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
            states = torch.cat(state_batch).to(device)
            actions = torch.tensor(action_batch).view(-1, 1).to(device)
            rewards = torch.tensor(reward_batch).view(-1, 1).to(device)
            dones = torch.tensor(terminal_batch).view(-1, 1).int().to(device)
            next_states = torch.cat(next_state_batch).to(device)

            with torch.no_grad():
                td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                td_delta = td_target - critic(states)
                advantage = compute_advantage(opt.gamma, opt.lmbda, td_delta)
                old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

            for _ in range(opt.epochs):
                for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                    log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                    ratio = torch.exp(log_probs - old_log_probs[index])
                    surr1 = ratio * advantage[index]
                    surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]
                    actor_loss = -torch.mean(torch.min(surr1, surr2))
                    critic_loss = nn.functional.mse_loss(critic(states[index]), td_target[index].detach())

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()
            replay_memory = []

        iter += 1

        if (iter + 1) % 10 == 0:
            evaluate_num += 1
            evaluate_rewards.append(episode_return)
            print(f"evaluate_num: {evaluate_num} \t episode_return: {episode_return}")
            writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step=iter)

        if (iter + 1) % 1000 == 0:
            os.makedirs(opt.saved_path, exist_ok=True)
            actor_dict = {"net": actor.state_dict(), "optimizer": actor_optimizer.state_dict()}
            critic_dict = {"net": critic.state_dict(), "optimizer": critic_optimizer.state_dict()}
            torch.save(actor_dict, os.path.join(opt.saved_path, "flappy_bird_actor"))
            torch.save(critic_dict, os.path.join(opt.saved_path, "flappy_bird_critic"))
            print(f"Model saved at iteration {iter + 1}")

if __name__ == "__main__":
    opt = get_args()
    train(opt)
