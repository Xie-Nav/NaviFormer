# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size):
        # num_steps = 20; num_processes = bs; obs_shape = (2+2+4+16, 240, 240)
        # action_space = Discrete(4); rec_state_size = 1
        
        if action_space.__class__.__name__ == 'Discrete':
            # g_rollouts.n_actions = 1
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        # g_rollouts.obs torch.Size([20+1, bs, 2+2+4+16, 240, 240])
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        # g_rollouts.rec_states torch.Size([20+1, bs, 1])
        self.rec_states = torch.zeros(num_steps + 1, num_processes, rec_state_size)
        # g_rollouts.rewards torch.Size([20, bs])
        self.rewards = torch.zeros(num_steps, num_processes)
        # g_rollouts.value_preds torch.Size([20+1, bs])
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        # g_rollouts.returns torch.Size([20+1, bs])
        self.returns = torch.zeros(num_steps + 1, num_processes)
        # g_rollouts.action_log_probs torch.Size([20, bs])
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        # g_rollouts.actions torch.Size([20, bs, 1])
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions),
                                   dtype=action_type)
        # g_rollouts.masks 初始化的时候是全1的数组 torch.Size([20, bs])
        self.masks = torch.ones(num_steps + 1, num_processes)
        # g_rollouts.num_steps = 20
        self.num_steps = num_steps
        # g_rollouts.step = 0
        self.step = 0
        # g_rollouts.has_extras = False
        self.has_extras = False
        # g_rollouts.extras_size = None
        self.extras_size = None

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        # self.invalid_action_masks = self.invalid_action_masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        # 
        self.obs[self.step + 1].copy_(obs)
        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        
        # 
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        # self.invalid_action_masks[0].copy_(self.invalid_action_masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        # next_value torch.Size([bs]) 
        # use_gae = False; gamma = 0.99; tau = 0.95
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            # g_rollouts.returns torch.Size([20+1, bs])
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                # g_rollouts.returns torch.Size([20+1, bs])
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        # advantages torch.Size([20, bs]); num_mini_batch = 2
        # g_rollouts.rewards torch.Size([20, bs])
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        # 这里应该是一次采集10个样本
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'rec_states': self.rec_states[:-1].view(
                    -1, self.rec_states.size(-1))[indices],
                # g_rollouts.actions torch.Size([20, bs, 1])
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                # g_rollouts.action_log_probs torch.Size([20, bs])
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                # advantages torch.Size([20, bs]);
                'adv_targ': advantages.view(-1)[indices],
                # g_rollouts.extras torch.Size([20+1, bs, 2])
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                if self.has_extras else None,
                # 'invalid_action_masks': self.invalid_action_masks[:-1].view(-1, 4)[indices],
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size, extras_size):
        # num_steps = 20; num_processes = bs; obs_shape = (24, 240, 240)
        # action_space = Discrete(4); rec_state_size = 1
        # extras_size = 2
        super(GlobalRolloutStorage, self).__init__(
            num_steps, num_processes, obs_shape, action_space, rec_state_size)
        
        # g_rollouts.extras torch.Size([20+1, bs, 2 ])
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        # g_rollouts.has_extras = True
        self.has_extras = True
        # g_rollouts.extras_size = 2
        self.extras_size = extras_size
        # # g_rollouts.invalid_action_masks  torch.Size([20+1, bs, 4])
        # self.invalid_action_masks = torch.zeros(num_steps+1, num_processes, 4)

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras):
        self.extras[self.step + 1].copy_(extras)
        # self.invalid_action_masks[self.step+1].copy_(invalid_action_masks)
        super(GlobalRolloutStorage, self).insert(
            obs, rec_states, actions,
            action_log_probs, value_preds, rewards, masks)
