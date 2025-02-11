import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

state_dimm = 7

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class lstm_parallel(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_parallel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.f1 = nn.Linear(self.hidden_size, 128)
        self.f2 = nn.Linear(self.hidden_size, 128)

    def forward(self, x_input, hidden_state, long=10):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        self.lstm.flatten_parameters()
        batch_size = x_input.size(0)
        state = x_input.view(batch_size, -1, self.input_size)
        # print('1',hidden_state[0].shape,state.shape)
        n_target = state.size(1)
        hidden_init, cell_init = torch.zeros(n_target, batch_size, self.hidden_size), torch.zeros(n_target, batch_size, self.hidden_size)
        hidden, cell = torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)
        hidden_init = hidden_init.to(device)
        cell_init = cell_init.to(device)
        hidden = hidden.to(device)
        cell = cell.to(device)

        for i in range(1, n_target):
            ss = state[:, i, :]
            ss = ss.unsqueeze(1)
            _, (hidden_init[i, :, :], cell_init[i, :,:]) = self.lstm(ss, hidden_state)
        # print(hidden_init.shape)
        for i in range(n_target):
            # print((self.f1(hidden_init[i, :, :])*state[:, i, 0]/long).shape)
            # print(self.f1(hidden_init[i, :, :]).shape,(state[:, i, 0]/long).shape)

            # print((self.f1(hidden_init[i, :, :])*state[:, i, 0]/long).shape,(self.f2(hidden_init[i, :, :])*state[:, i, 1]).shape)

            # print(torch.cat([self.f1(hidden_init[i, :, :])*state[:, i, 0]/long, self.f2(hidden_init[i, :, :])*state[:, i, 1]], dim=-1)[0].shape)
            hidden[0, :, :] += torch.cat([self.f1(hidden_init[i, :, :])*state[:, i, 0].unsqueeze(1)/long, self.f2(hidden_init[i, :, :])*state[:, i, 1].unsqueeze(1)], dim=-1)
            cell[0, :, :] += torch.cat([self.f1(cell_init[i, :, :])*state[:, i, 0].unsqueeze(1)/long, self.f2(cell_init[i, :, :])*state[:, i, 1].unsqueeze(1)], dim=-1)
        # print('hidden',hidden.shape)
        hiddenn = (hidden, cell)
        # lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        # print(hidden.shape)
        return hiddenn

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_serial(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_serial, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, hidden_state):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        self.lstm.flatten_parameters()
        batch_size = x_input.size(0)
        # print(x_input.shape)
        state = x_input.view(batch_size, -1, self.input_size)
        n_target = state.size(1)
        # hidden_init, cell_init = torch.zeros(n_target, batch_size, self.hidden_size), torch.zeros(n_target, batch_size,
        #                                                                                           self.hidden_size)
        # hidden, cell = torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)
        # hidden_init = hidden_init.to(device)
        # cell_init = cell_init.to(device)
        # hidden = hidden.to(device)
        # cell = cell.to(device)
        for i in range(n_target - 1, 0, -1):
            ss = state[:, i, :]
            ss = ss.unsqueeze(1)
            # print(ss.shape)
            a, hidden_state = self.lstm(ss, hidden_state)
            # print(a.shape)
            # print(hidden_state[0].shape)
        #shape:a torch.Size([batch_size=1, 1, output_size])
        #hidden_state,size tuple(hidden,cell)  ([1,batch_size,hidden_size=256])



        return a, hidden_state

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
class Actor(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_dim, max_action, is_recurrent=False
    ):
        super(Actor, self).__init__()
        self.recurrent = is_recurrent

        if self.recurrent:
            # self.l1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
            self.serial = lstm_serial(state_dimm, hidden_dim)
            self.parallel = lstm_parallel(state_dimm, hidden_dim)
            self.LSTM = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)

            self.l2 = nn.Linear(hidden_dim + state_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, action_dim)

            self.max_action = max_action

        else:
            self.serial = lstm_serial(state_dimm, hidden_dim)
            self.parallel = lstm_parallel(state_dimm, hidden_dim)
            self.LSTM = nn.LSTM(input_size=2 * hidden_dim, hidden_size=hidden_dim,
                                num_layers=1, batch_first=True)

            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, action_dim)

            self.max_action = max_action

    def forward(self, state, hidden):
        if self.recurrent:
            state = state.view(state.size(0), -1, state_dimm)
            # print(hidden[0].shape,state.shape)
            # print(state.shape)
            # print('hidden',hidden[0].shape)
            a, serial = self.serial(state, hidden)
            # print(serial[0].shape)
            # print('state',serial[0].shape)
            parallel = self.parallel(state, hidden)
            # print(parallel[0].shape)

            state_embedded = torch.cat([serial[0],parallel[0]], -1)
            state_embedded = state_embedded.view(state_embedded.shape[1],state_embedded.shape[0],-1)
            a, hidden = self.LSTM(state_embedded, hidden)

            # print(a.shape,hidden[0].shape)#torch.Size([2, 1, 256]) torch.Size([1, 2, 256])
            a = torch.cat([a, state[:, 0, :].unsqueeze(1)], -1)
            h = hidden
            a = F.relu(self.l2(a))
            a = torch.tanh(self.l3(a))
        else:
            state = state.view(state.size(0), -1, state_dimm)
            # print(hidden[0].shape,state.shape)
            # print(state.shape)
            # print('hidden',hidden[0].shape)
            a, serial = self.serial(state, hidden)
            # print(serial[0].shape)
            # print('state',serial[0].shape)
            parallel = self.parallel(state, hidden)
            # print(parallel[0].shape)

            state_embedded = torch.cat([serial[0], parallel[0]], -1)
            state_embedded = state_embedded.view(state_embedded.shape[1], state_embedded.shape[0], -1)
            a, hidden = self.LSTM(state_embedded, hidden)

            # print(a.shape,hidden[0].shape)#torch.Size([2, 1, 256]) torch.Size([1, 2, 256])
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
            # self.l1 = nn.LSTM(
            #     state_dimm + action_dim, hidden_dim, batch_first=True)
            self.l11 = lstm_serial(state_dimm + action_dim, hidden_dim)
            self.l22 = lstm_serial(state_dimm + action_dim, hidden_dim)
            self.l33 = lstm_parallel(state_dimm + action_dim, hidden_dim)
            self.l44 = lstm_parallel(state_dimm + action_dim, hidden_dim)
            self.LSTM = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)


            # self.l4 = nn.LSTM(
            #     state_dimm + action_dim, hidden_dim, batch_first=True)
            self.l2 = nn.Linear(hidden_dim + state_dimm, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, 1)

            # Q2 architecture
            self.l5 = nn.Linear(hidden_dim + state_dimm, hidden_dim)
            self.l6 = nn.Linear(hidden_dim, 1)

        else:
            self.l11 = lstm_serial(state_dimm + action_dim, hidden_dim)
            self.l22 = lstm_serial(state_dimm + action_dim, hidden_dim)
            self.l33 = lstm_parallel(state_dimm + action_dim, hidden_dim)
            self.l44 = lstm_parallel(state_dimm + action_dim, hidden_dim)
            self.LSTM = nn.LSTM(input_size=2 * hidden_dim, hidden_size=hidden_dim,
                                num_layers=1, batch_first=True)

            # self.l4 = nn.LSTM(
            #     state_dimm + action_dim, hidden_dim, batch_first=True)
            self.l2 = nn.Linear(hidden_dim , hidden_dim)
            self.l3 = nn.Linear(hidden_dim, 1)

            # Q2 architecture
            self.l5 = nn.Linear(hidden_dim , hidden_dim)
            self.l6 = nn.Linear(hidden_dim, 1)

        # Q1 architecture

    def forward(self, state, action, hidden1, hidden2):
        sa = torch.cat([state, action], -1)
        # print(sa.shape,state.shape,action.shape)
        if self.recurrent:

            # self.l1.flatten_parameters()
            # self.l4.flatten_parameters()
            # state = torch.cat([state, action], -1)
            # print(action.shape)
            state = state.view(state.size(0), -1, state_dimm)
            new_action = action.repeat(1, state.shape[1], 1)
            new_state = torch.cat([state, new_action], -1)

            a11, serial11 = self.l11(new_state, hidden1)
            a33, serial22 = self.l22(new_state, hidden2)
            parallel33 = self.l33(new_state, hidden1)
            parallel44 = self.l44(new_state, hidden2)
            # print(serial11[0].shape,parallel33[0].shape)

            state_embedded1 = torch.cat([serial11[0],parallel33[0]], -1)
            state_embedded1 = state_embedded1.view(state_embedded1.shape[1],state_embedded1.shape[0],-1)
            q1, hidden1 = self.LSTM(state_embedded1, hidden1)
            state_embedded2 = torch.cat([serial22[0], parallel44[0]], -1)
            state_embedded2 = state_embedded2.view(state_embedded2.shape[1], state_embedded2.shape[0], -1)
            q2, hidden2 = self.LSTM(state_embedded2, hidden2)
            # print(a.shape,hidden[0].shape)#torch.Size([2, 1, 256]) torch.Size([1, 2, 256])

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
            # self.l1.flatten_parameters()
            # self.l4.flatten_parameters()
            # state = torch.cat([state, action], -1)
            # print(action.shape)
            state = state.view(state.size(0), -1, state_dimm)
            new_action = action.repeat(1, state.shape[1], 1)
            new_state = torch.cat([state, new_action], -1)

            a11, serial11 = self.l11(new_state, hidden1)
            a33, serial22 = self.l22(new_state, hidden2)
            parallel33 = self.l33(new_state, hidden1)
            parallel44 = self.l44(new_state, hidden2)
            # print(serial11[0].shape,parallel33[0].shape)

            state_embedded1 = torch.cat([serial11[0], parallel33[0]], -1)
            state_embedded1 = state_embedded1.view(state_embedded1.shape[1], state_embedded1.shape[0], -1)
            q1, hidden1 = self.LSTM(state_embedded1, hidden1)
            state_embedded2 = torch.cat([serial22[0], parallel44[0]], -1)
            state_embedded2 = state_embedded2.view(state_embedded2.shape[1], state_embedded2.shape[0], -1)
            q2, hidden2 = self.LSTM(state_embedded2, hidden2)
            # print(a.shape,hidden[0].shape)#torch.Size([2, 1, 256]) torch.Size([1, 2, 256])

            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)

            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)

            # print(a.shape)
            # q1, hidden1 = self.l1(sa, hidden1)
            # q2, hidden2 = self.l4(sa, hidden2)

        return q1, q2

    def Q1(self, state, action, hidden1):
        sa = torch.cat([state, action], -1)

        if self.recurrent:

            state = state.view(state.size(0), -1, state_dimm)
            new_action = action.repeat(1, state.shape[1], 1)
            new_state = torch.cat([state, new_action], -1)

            a11, serial11 = self.l11(new_state, hidden1)
            parallel33 = self.l33(new_state, hidden1)
            # print(serial11[0].shape,parallel33[0].shape)

            state_embedded1 = torch.cat([serial11[0], parallel33[0]], -1)
            state_embedded1 = state_embedded1.view(state_embedded1.shape[1], state_embedded1.shape[0], -1)
            q1, hidden1 = self.LSTM(state_embedded1, hidden1)
            # self.l1.flatten_parameters()

            # q2, hidden2 = self.l4(new_state, hidden2)

            # state = state.view(state.size(0), -1, state_dimm+2)
            # q1, hidden1 = self.l1(state, hidden1)

            # for i in range(state.shape[1] - 1, 0, -1):
            #     ss = state[:, i, :]
            #
            #     ss = ss.unsqueeze(1)
            #     ss = torch.cat([ss, action], dim=2)
            #     q1, hidden1 = self.l1(ss, hidden1)
            q1 = torch.cat([q1, state[:, 0, :].unsqueeze(1)], -1)

            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
        else:
            state = state.view(state.size(0), -1, state_dimm)
            new_action = action.repeat(1, state.shape[1], 1)
            new_state = torch.cat([state, new_action], -1)

            a11, serial11 = self.l11(new_state, hidden1)
            parallel33 = self.l33(new_state, hidden1)
            # print(serial11[0].shape,parallel33[0].shape)

            state_embedded1 = torch.cat([serial11[0], parallel33[0]], -1)
            state_embedded1 = state_embedded1.view(state_embedded1.shape[1], state_embedded1.shape[0], -1)
            q1, hidden1 = self.LSTM(state_embedded1, hidden1)
            # self.l1.flatten_parameters()

            # q2, hidden2 = self.l4(new_state, hidden2)

            # state = state.view(state.size(0), -1, state_dimm+2)
            # q1, hidden1 = self.l1(state, hidden1)

            # for i in range(state.shape[1] - 1, 0, -1):
            #     ss = state[:, i, :]
            #
            #     ss = ss.unsqueeze(1)
            #     ss = torch.cat([ss, action], dim=2)
            #     q1, hidden1 = self.l1(ss, hidden1)

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
        # self.lstm_parallel = lstm_parallel(state_dimm, hidden_dim).to(device)
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
        # print(hidden[0].shape,self.lstm_parallel(state, hidden)[0].shape)
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
        # loss_prio = torch.Tensor(loss_prio)  # 将列表转换为PyTorch张量

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
