import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

state_dimm = 7

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)


# print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# class OUNoise:
#
#     def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, max_sigma=0.1, min_sigma=0, decay_period=500):
#         self.action_dimension = action_dimension
#         self.scale = scal
#         self.mu = mu
#         self.theta = theta
#         self.sigma = max_sigma  #  OU噪声的参数
#
#         self.max_sigma = max_sigma
#
#         self.min_sigma = min_sigma
#         self.decay_period = decay_period
#         self.state = np.ones(self.action_dimension) * self.mu
#         self.n_actions = action_dimension.shape[0]
#         self.low = action_dimension.low[0]
#
#         self.high = action_dimension.high[0]
#         self.reset()
#
#     def reset(self):
#         self.state = np.ones(self.action_dimension) * self.mu
#
#     def state_noise(self):
#
#         x = self.state
#
#         dx = self.theta * (self.mu-x)+self.sigma * np.random.randn(self.n_actions)
#
#         self.state = x + dx
#
#         return self.state
#
#     def noise(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
#         self.state = x + dx
#         return self.state * self.scale

class Actor(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_dim, max_action, is_recurrent=False
    ):
        super(Actor, self).__init__()
        self.recurrent = is_recurrent

        if self.recurrent:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
            self.l2 = nn.Linear(hidden_dim + state_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, action_dim)

            self.max_action = max_action

        else:
            self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, action_dim)

            self.max_action = max_action

    def forward(self, state, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            state = state.view(state.size(0), -1, state_dimm)
            # print(hidden[0].shape,state.shape)
            # print(state.shape)
            for i in range(state.shape[1] - 1, 0, -1):
                ss = state[:, i, :]

                ss = ss.unsqueeze(1)
                a, hidden = self.l1(ss, hidden)
                # print(ss.shape, hidden[0].shape)
                # print(a.shape)
            # print(hidden)
            a = torch.cat([a, state[:, 0, :].unsqueeze(1)], -1)
            h = hidden
            a = F.relu(self.l2(a))
            a = torch.tanh(self.l3(a))
        else:
            self.l1.flatten_parameters()
            state = state.view(state.size(0), -1, state_dimm)
            # print(hidden[0].shape,state.shape)
            # print(state.shape)
            for i in range(state.shape[1] - 1, 0, -1):
                ss = state[:, i, :]

                ss = ss.unsqueeze(1)
                a, hidden = self.l1(ss, hidden)
                # print(ss.shape, hidden[0].shape)
                # print(a.shape)
            # print(hidden)
            h = hidden
            a = F.relu(self.l2(a))
            a = torch.tanh(self.l3(a))

        return self.max_action * a, h


class Critic(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_dim, is_recurrent=False
    ):
        super(Critic, self).__init__()
        self.recurrent = is_recurrent

        if self.recurrent:
            self.l1 = nn.LSTM(
                state_dimm + action_dim, hidden_dim, batch_first=True)

            self.l4 = nn.LSTM(
                state_dimm + action_dim, hidden_dim, batch_first=True)
            self.l2 = nn.Linear(hidden_dim + state_dimm, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, 1)

            # Q2 architecture
            self.l5 = nn.Linear(hidden_dim + state_dimm, hidden_dim)
            self.l6 = nn.Linear(hidden_dim, 1)

        else:
            self.l1 = nn.LSTM(
                state_dimm + action_dim, hidden_dim, batch_first=True)

            self.l4 = nn.LSTM(
                state_dimm + action_dim, hidden_dim, batch_first=True)
            self.l2 = nn.Linear(hidden_dim , hidden_dim)
            self.l3 = nn.Linear(hidden_dim, 1)

            # Q2 architecture
            self.l5 = nn.Linear(hidden_dim , hidden_dim)
            self.l6 = nn.Linear(hidden_dim, 1)

        # Q1 architecture

    def forward(self, state, action, hidden1, hidden2):
        sa = torch.cat([state, action], -1)
        if self.recurrent:
            self.l1.flatten_parameters()
            self.l4.flatten_parameters()
            state = state.view(state.size(0), -1, state_dimm)
            for i in range(state.shape[1] - 1, 0, -1):
                ss = state[:, i, :]

                ss = ss.unsqueeze(1)
                ss = torch.cat([ss, action], dim=2)
                q1, hidden1 = self.l1(ss, hidden1)
                q2, hidden2 = self.l4(ss, hidden2)
            q1 = torch.cat([q1, state[:, 0, :].unsqueeze(1)], -1)
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            q2 = torch.cat([q2, state[:, 0, :].unsqueeze(1)], -1)

            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)

            # print(a.shape)
            # q1, hidden1 = self.l1(sa, hidden1)
            # q2, hidden2 = self.l4(sa, hidden2)
        else:
            self.l1.flatten_parameters()
            self.l4.flatten_parameters()
            state = state.view(state.size(0), -1, state_dimm)
            for i in range(state.shape[1] - 1, 0, -1):
                ss = state[:, i, :]

                ss = ss.unsqueeze(1)
                ss = torch.cat([ss, action], dim=2)
                q1, hidden1 = self.l1(ss, hidden1)
                q2, hidden2 = self.l4(ss, hidden2)
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action, hidden1):
        sa = torch.cat([state, action], -1)

        if self.recurrent:
            self.l1.flatten_parameters()
            state = state.view(state.size(0), -1, state_dimm)
            for i in range(state.shape[1] - 1, 0, -1):
                ss = state[:, i, :]

                ss = ss.unsqueeze(1)
                ss = torch.cat([ss, action], dim=2)
                q1, hidden1 = self.l1(ss, hidden1)
            q1 = torch.cat([q1, state[:, 0, :].unsqueeze(1)], -1)

            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
        else:
            self.l1.flatten_parameters()
            state = state.view(state.size(0), -1, state_dimm)
            for i in range(state.shape[1] - 1, 0, -1):
                ss = state[:, i, :]

                ss = ss.unsqueeze(1)
                ss = torch.cat([ss, action], dim=2)
                q1, hidden1 = self.l1(ss, hidden1)

            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
        return q1


class RTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_dim,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=3e-4,
            recurrent_actor=False,
            recurrent_critic=False,
    ):
        self.on_policy = False
        self.recurrent = recurrent_actor
        self.actor = Actor(
            state_dim, action_dim, hidden_dim, max_action,
            is_recurrent=recurrent_actor
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)

        self.critic = Critic(
            state_dim, action_dim, hidden_dim,
            is_recurrent=recurrent_critic
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def get_initial_states(self, batch_size=1):
        h_0, c_0 = None, None
        if self.actor.recurrent:
            h_0 = torch.zeros((
                self.actor.l1.num_layers,
                batch_size,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=device)

            c_0 = torch.zeros((
                self.actor.l1.num_layers,
                batch_size,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=device)
        return (h_0, c_0)

    def select_action(self, state, hidden, test=True):
        if self.recurrent:
            state = torch.FloatTensor(
                state.reshape(1, -1)).to(device)[:, None, :]
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # print(state, hidden)

        action, hidden = self.actor(state, hidden)
        return action.cpu().data.numpy().flatten(), hidden

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, hidden, next_hidden, ind, weights = \
            replay_buffer.sample(batch_size)
        # print((hidden[1].shape))
        # print(next_hidden)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state, next_hidden)[0] + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(
                next_state, next_action, next_hidden, next_hidden)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            # print("target_Q",target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, hidden, hidden)
        # print('current_Q1',current_Q1)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)
        # print(critic_loss.item())
        loss_prio = critic_loss.cpu().detach().numpy() * weights
        loss_prio = abs(loss_prio)  # 可能需要对损失值取绝对值
        loss_prio = [loss_prio]  # 将浮点数放入列表中

        loss_prio = torch.Tensor(loss_prio)  # 将其转换为PyTorch张量
        loss_prio_numpy = loss_prio.detach().cpu().numpy()  # 移动到CPU并转换为NumPy数组

        # 确保 loss_prio_numpy 现在是一个 NumPy 数组
        replay_buffer.update_priorities(ind, loss_prio_numpy)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(
                state, self.actor(state, hidden)[0], hidden).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
    def ceshi(self, state, hidden,action):
        current_Q1, current_Q2 = self.critic(state, action, hidden, hidden)
        return current_Q1, current_Q2

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
