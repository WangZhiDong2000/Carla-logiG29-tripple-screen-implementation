from collections import deque

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
            self, state_dim, action_dim, hidden_size,
            max_size=int(5e3), recurrent=False, gamma=0.8, n_step=1,alpha=0.6, PrioritizedReplay=False, beta_start=0.4, beta_frames=100000
    ):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.PrioritizedReplay = PrioritizedReplay
        self.max_size = int(max_size)
        self.frame = 1 #for beta calculation

        self.ptr = 0
        self.size = 0
        self.recurrent = recurrent

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        self.num_done = 0
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=1000)
        self.Return = deque(maxlen=1000)
        self.Return1 = deque(maxlen=1000)
        self.priorities = np.zeros((self.max_size,), dtype=np.float32)

        self.iter_ = 0
        self.n_step = n_step

        if self.recurrent:
            self.h = np.zeros((self.max_size, hidden_size))
            self.nh = np.zeros((self.max_size, hidden_size))

            self.c = np.zeros((self.max_size, hidden_size))
            self.nc = np.zeros((self.max_size, hidden_size))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(
            self, state, action, next_state, reward, done, hiddens, next_hiddens
    ):
        max_prio = self.priorities.max() if self.ptr > 0 else 1.0
        self.n_step_buffer.append(reward)
        self.Return.append(reward)
        self.Return1.append(reward)

        self.iter_ += 1

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.priorities[self.ptr] = max_prio

        if self.recurrent:
            h, c = hiddens
            nh, nc = next_hiddens

            # Detach the hidden state so that BPTT only goes through 1 timestep
            self.h[self.ptr] = h.detach().cpu()
            self.c[self.ptr] = c.detach().cpu()
            self.nh[self.ptr] = nh.detach().cpu()
            self.nc[self.ptr] = nc.detach().cpu()
        if done == 1.0:
            for i in range(self.n_step + 5):
                self.Return.append(0.0)
                self.Return1.append(0.0)

            for i in range(len(self.n_step_buffer) - 1, -1, -1):
                for inx in range(self.n_step):
                    self.n_step_buffer[i] += self.gamma ** (inx + 1) * self.Return[inx + 1 + i]
                    self.Return1[i] += self.gamma ** (inx + 1) * self.Return[inx + 1 + i]

                self.Return[i + self.n_step - 1] = self.Return1[i + self.n_step - 1]

            self.iter_ = 0
            for i in range(len(self.n_step_buffer)):
                self.reward[(self.num_done + i) % self.max_size] = self.n_step_buffer[i]
                print(self.reward[(self.num_done + i) % self.max_size])
            self.num_done = (self.ptr + 1) % self.max_size
            self.n_step_buffer.clear()
            self.Return1.clear()
            self.Return.clear()

        self.ptr = (self.ptr + 1) % self.max_size

        self.size = min(self.size + 1, self.max_size)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    def sample(self, batch_size=100):
        # TODO: Clean this up. There's probably a cleaner way to seperate
        # on-policy and off-policy sampling. Clean up extra-dimension indexing
        # also
        N = len(self.state)
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        P = probs/probs.sum()
        # print(P,N,prios)
        if self.PrioritizedReplay:
            ind = np.random.choice(N, int(batch_size), p=P)
            beta = self.beta_by_frame(self.frame)
            self.frame += 1

            # Compute importance-sampling weight
            weights = (N * P[ind]) ** (-beta)
            # normalize weights
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)
        else:
            ind = np.random.randint(0, self.size, size=int(batch_size))
            weights = 0

        # TODO: Clean up indexing. RNNs needs batch shape of
        # Batch size * Timesteps * Input size
        if not self.recurrent:
            return self._ff_sampling(ind)

        h = torch.tensor(self.h[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        c = torch.tensor(self.c[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        nh = torch.tensor(self.nh[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)
        nc = torch.tensor(self.nc[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)

        # TODO: Return hidden states or not, or only return the
        # first hidden state (although it's already been detached,
        # so returning nothing might be better)
        hidden = (h, c)
        next_hidden = (nh, nc)

        s = torch.FloatTensor(
            self.state[ind][:, None, :]).to(self.device)
        a = torch.FloatTensor(
            self.action[ind][:, None, :]).to(self.device)
        ns = torch.FloatTensor(
            self.next_state[ind][:, None, :]).to(self.device)
        r = torch.FloatTensor(
            self.reward[ind][:, None, :]).to(self.device)
        d = torch.FloatTensor(
            self.not_done[ind][:, None, :]).to(self.device)


        return s, a, ns, r, d, hidden, next_hidden,ind, weights

    def on_policy_sample(self):
        ind = np.arange(0, self.size)

        # TODO: Clean up indexing. RNNs needs batch shape of
        # Batch size * Timesteps * Input size
        if not self.recurrent:
            return self._ff_sampling(ind)

        h = torch.tensor(self.h[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        c = torch.tensor(self.c[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float).to(self.device)
        nh = torch.tensor(self.nh[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)
        nc = torch.tensor(self.nc[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float).to(self.device)

        # TODO: Return hidden states or not, or only return the
        # first hidden state (although it's already been detached,
        # so returning nothing might be better)
        hidden = (h, c)
        next_hidden = (nh, nc)

        s = torch.FloatTensor(
            self.state[ind][:, None, :]).to(self.device)
        a = torch.FloatTensor(
            self.action[ind][:, None, :]).to(self.device)
        ns = torch.FloatTensor(
            self.next_state[ind][:, None, :]).to(self.device)

        # reward and dones don't need to be "batched"
        r = torch.FloatTensor(
            self.reward[ind]).to(self.device)
        d = torch.FloatTensor(
            self.not_done[ind]).to(self.device)

        return s, a, ns, r, d, hidden, next_hidden

    def _ff_sampling(self, ind):
        # FF only need Batch size * Input size, on_policy or not
        hidden = None
        next_hidden = None

        s = torch.FloatTensor(self.state[ind]).to(self.device)
        a = torch.FloatTensor(self.action[ind]).to(self.device)
        ns = \
            torch.FloatTensor(self.next_state[ind]).to(self.device)
        r = torch.FloatTensor(self.reward[ind]).to(self.device)
        d = torch.FloatTensor(self.not_done[ind]).to(self.device)

        return s, a, ns, r, d, hidden, next_hidden

    def clear_memory(self):
        self.ptr = 0
        self.size = 0
