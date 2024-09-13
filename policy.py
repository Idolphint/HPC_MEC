import torch
import torch.nn as nn
from torch.distributions import Categorical


class GridPolicy(nn.Module):
    def __init__(self, mec_module_num):
        super().__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(mec_module_num, 16, 5),
        #     # nn.LeakyReLU(),
        #     nn.Conv2d(16, 16, 3),
        #     # nn.LeakyReLU(),
        #     # nn.AvgPool2d(kernel_size=5),  # 期望到这一步变为16,1
        #     # nn.LeakyReLU(),
        # )
        self.conv1 = nn.Sequential(
            nn.Linear(100, 32),
            nn.LeakyReLU(),
            # nn.Linear(256,32),
            # nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(7*32, 32),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(32, 4),
            # nn.LeakyReLU(),
            # nn.Linear(16, 4),
            nn.Softmax(),
        )
        self.critic = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, grid_vector):
        # 接受目标和当前的grid code的差产生策略,直接用attention产生action，
        # print("input", grid_vector.shape)
        conv_x = self.conv1(grid_vector)
        # print(conv_x.shape)
        conv_x = self.fc2(conv_x.reshape(1, -1))
        # print(conv_x.shape, conv_x)
        action_probs = self.actor(conv_x).squeeze()
        # print(action_probs.shape)
        values = self.critic(conv_x).squeeze()
        # print("action prob", action_probs)
        return action_probs, values

    def get_actions(self, grid_vector, train=True):
        action_prob, value = self(grid_vector)
        cat = Categorical(action_prob)
        if train:
            action = cat.sample()
        else:
            action = torch.argmax(action_prob)
        return action, cat.log_prob(action), cat.entropy().mean(), value


class ReplayBuffer:
    def __init__(self, buffer_size, mec_module=7, mec_x_num=10, mec_y_num=10, writer=None, device='cpu'):
        self.buffer_size = buffer_size
        self.mec_module = mec_module
        self.mec_x = mec_x_num
        self.mec_y = mec_y_num
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.gloabl_step = 0

        self.device = device
        self.states = torch.zeros((buffer_size, mec_module, mec_x_num* mec_y_num)).to(device)
        self.dones = torch.zeros((buffer_size,)).to(device)
        self.log_probs = torch.zeros((buffer_size, )).to(device)
        self.values = torch.zeros((buffer_size, )).to(device)
        self.rewards = torch.zeros(buffer_size).to(device)
        self.advantages = torch.zeros(self.buffer_size).to(device)
        self.actions = torch.zeros(self.buffer_size).to(device)
        self.index = 0

        self.writer = writer

    def reset_state(self):
        self.states = torch.zeros((self.buffer_size, self.mec_module, self.mec_x* self.mec_y)).to(self.device)
        self.dones = torch.zeros((self.buffer_size,)).to(self.device)
        self.log_probs = torch.zeros((self.buffer_size, )).to(self.device)
        self.values = torch.zeros((self.buffer_size,)).to(self.device)
        self.rewards = torch.zeros(self.buffer_size).to(self.device)
        self.advantages = torch.zeros(self.buffer_size).to(self.device)
        self.actions = torch.zeros(self.buffer_size).to(self.device)
        self.index = 0

    def add_state(self, state, done, logp, value, r, a):
        if self.index < self.buffer_size:
            self.states[self.index].copy_(state)
            self.dones[self.index] = done
            self.log_probs[self.index].copy_(logp)
            self.values[self.index].copy_(value)
            self.rewards[self.index] = r
            self.actions[self.index] = a

            self.index += 1

    def _discount_rewards(self, final_value, discount=0.99, gae_discount=0.95):
        r_discounted = torch.zeros_like(self.rewards).to(self.device)
        last_value = final_value
        # last_adv = torch.zeros_like(final_value)
        for i in reversed(range(self.index)):
            mask = 1.0 - self.dones[i]
            last_value = last_value * mask

            last_value = self.rewards[i] + discount * last_value
            r_discounted[i] = last_value

        return r_discounted

    def a2c_loss(self, final_value, entropy):
        loss_mask = torch.zeros(self.buffer_size).to(self.device).bool()
        loss_mask[:self.index] = 1
        print("check mask", loss_mask.sum())
        # loss_mask = torch.BoolTensor(loss_mask).to(self.device)
        entropy /= self.index
        # calculate advantage
        # i.e. how good was the estimate of the value of the current state
        rewards = self._discount_rewards(final_value)
        advantage = rewards - self.values
        advantage = advantage[loss_mask]
        # weight the deviation of the predicted value (of the state) from the
        # actual reward (=advantage) with the negative log probability of the action taken
        policy_loss = (-self.log_probs[loss_mask] * advantage.detach()).mean()

        # the value loss weights the squared difference between the actual
        # and predicted rewards
        value_loss = advantage.pow(2).mean()

        # return the a2c loss
        # which is the sum of the actor (policy) and critic (advantage) losses
        # due to the fact that batches can be shorter (e.g. if an env is finished already)
        # MEAN is used instead of SUM
        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        if self.writer is not None:
            self.writer.add_scalar("a2c_loss", loss.item(), self.gloabl_step)
            self.writer.add_scalar("policy_loss", policy_loss.item(), self.gloabl_step)
            self.writer.add_scalar("value_loss", value_loss.item(), self.gloabl_step)
            self.writer.add_histogram("advantage", advantage.detach(), self.gloabl_step)
            self.writer.add_histogram("rewards", rewards.detach(), self.gloabl_step)
            self.writer.add_histogram("action_prob", self.log_probs.detach(), self.gloabl_step)
            self.gloabl_step += 1

        print("loss ", loss.item(), "policy loss", policy_loss.item(), "value", value_loss.item(),
              "adv", advantage.mean().item(), "entropy", entropy)

        return loss

if __name__ == '__main__':
    policy = GridPolicy(mec_module_num=7)
    grid_input = torch.randn((1,7, 100))
    act, logp, entropy, value = policy.get_actions(grid_input, train=True)