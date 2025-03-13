# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim


class PPO():

    def __init__(
            self,
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=None,
            eps=None,
            max_grad_norm=None,
            use_clipped_value_loss=True):

        # actor_critic 是我们刚刚创建好的网络
        # clip_param = 0.2; ppo_epoch = 4; num_mini_batch = 2; value_loss_coef = 0.5
        # entropy_coef = 0.001; lr = 2.5e-05; eps = 1e-05; max_grad_norm = 0.5
        # use_clipped_value_loss=True
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        # g_agent.use_clipped_value_loss = True
        self.use_clipped_value_loss = use_clipped_value_loss

        # 这是我们梯度下降的优化器
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                           actor_critic.parameters()),
                                    lr=lr, eps=eps)

    def update(self, rollouts):
        # rollouts.returns和rollouts.value_preds的最后一项是不算的所以不用考虑
        # advantages torch.Size([20, bs])
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                # advantages torch.Size([20, bs]) g_agent.num_mini_batch = 2
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_preds = sample['value_preds']
                returns = sample['returns']
                adv_targ = sample['adv_targ']

                # Reshape to do in a single forward pass for all steps
                # 再循环的过程中模型参数是在不算更新的, 而sample中的奖励是没有发生变化的,
                # 所以这样的PPO训练方式就是为了让新的模型参数和奖励标签越来越接近
                values, action_log_probs, dist_entropy, _ = \
                    self.actor_critic.evaluate_actions(
                        sample['obs'], sample['rec_states'],
                        sample['masks'], sample['actions'],
                        extras=sample['extras']
                        # ,invalid_action_masks = sample['invalid_action_masks']
                        )

                ratio = torch.exp(action_log_probs -
                                  sample['old_action_log_probs'])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(
                            -self.clip_param, self.clip_param)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped
                                            - returns).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
