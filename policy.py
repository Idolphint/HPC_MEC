import random

import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import numpy as np

# Define a custom activation function using sine
class SineLayer(nn.Module):
    # https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=znA_YU6-B8yc
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    # 需要注意paper是通过坐标预测坐标内容的，该layer可替代linear+relu使用
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class GridPolicy(nn.Module):
    def __init__(self, device, action_std_init=0.01, mec_module_num=7):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(mec_module_num*2, 32, 3), #, padding=0, padding_mode='circular'),
            # SinActivation(),
            nn.MaxPool2d(3,1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3),  #ding=1, padding_mode='circular'),
            # SinActivation(),
            # nn.AvgPool2d(kernel_size=4),
            nn.LeakyReLU(),
            # nn.Sigmoid(),
            # nn.MaxPool2d(3, 1),
            # nn.AvgPool2d(kernel_size=5),  # 期望到这一步变为16,1
            # nn.LeakyReLU(),
        )
        self.device = device
        self.action_continue = False
        self.action_std = action_std_init
        if self.action_continue:
            self.action_dim = 2
            self.action_var = torch.full((2,), action_std_init * action_std_init).to(device)

        # self.conv1 = nn.Sequential(
        #     nn.Linear(100, 32),
        #     nn.LeakyReLU(),
        #     # nn.Linear(16,2),
        #     # nn.Sigmoid(),
        # )
        self.fc2 = nn.Sequential(
            nn.Linear(32*4*4, 32),
            nn.LeakyReLU(),
            # SinActivation(),
        )
        # self.fc2 = SineLayer(32, 32, is_first=True)

        self.actor = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 4),
            nn.Softmax(),
        )
        self.critic = nn.Sequential(
            nn.Linear(32, 1)
        )
        self.loc_decoder = nn.Linear(32,2)
    def set_action_std(self, new_action_std):
        if self.action_continue:
            self.action_std = new_action_std
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
    def get_loc(self, grid_vector):
        x = self.conv1(grid_vector)
        x = x.reshape(x.shape[0], -1)
        x = self.fc2(x)
        y = self.loc_decoder(x)
        return y

    def forward(self, grid_vector):
        # 7,100 or n,7,100
        # 接受目标和当前的grid code的差产生策略,直接用attention产生action，
        # print("input", grid_vector.shape)
        conv_x = self.conv1(grid_vector)
        # print("step1", conv_x.shape)
        conv_x = conv_x.reshape(conv_x.shape[0], -1)
        conv_x = self.fc2(conv_x)
        if np.random.rand() < 0.004:
            print("step2", conv_x)
        act_vec = self.actor(conv_x).squeeze()
        values = self.critic(conv_x).squeeze()
        return act_vec, values

    def get_actions(self, grid_vector, train=True):
        action_prob, value = self(grid_vector)
        if not train and np.random.rand() < 0.1:
            print("check action probs", action_prob)
        if self.action_continue:
            action_mean = action_prob
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            cat = MultivariateNormal(action_mean, cov_mat)
        else:
            cat = Categorical(action_prob)
        if train:
            action = cat.sample()
        else:  # TODO 连续的可以用max吗？
            if self.action_continue:
                action = action_prob
                print("check ori act", action)
            else:  # 离散动作必须sample才有效
                action = cat.sample()
        # print("check ori action", action_prob, action)
        return action, cat.log_prob(action).squeeze(0), cat.entropy().mean(), value

    def evaluate(self, old_state, old_action):
        action_mean, value = self(old_state)
        # aux loss
        pred_diff = self.get_loc(old_state)
        if self.action_continue:
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            # print("check action var", action_var.shape, cov_mat.shape, old_action.shape)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            dist = Categorical(action_mean)
        # for single action continuous environments
        if self.action_continue and self.action_dim == 1:
            old_action = old_action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(old_action)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy, pred_diff


class ReplayBuffer:
    def __init__(self, buffer_size, mec_module=7, mec_x_num=10, mec_y_num=10, writer=None, device='cpu'):
        self.buffer_size = buffer_size
        self.mec_module = mec_module
        self.mec_x = mec_x_num
        self.mec_y = mec_y_num
        self.value_coeff = 0.5
        self.entropy_coeff = 0.03
        self.gloabl_step = 0

        self.device = device
        self.states = torch.zeros((buffer_size, mec_module*2, mec_x_num, mec_y_num)).to(device)
        self.dones = torch.zeros((buffer_size,)).to(device)
        self.log_probs = torch.zeros((buffer_size, )).to(device)
        self.values = torch.zeros((buffer_size, )).to(device)
        self.rewards = torch.zeros(buffer_size).to(device)
        self.advantages = torch.zeros(self.buffer_size).to(device)
        self.actions = torch.zeros(self.buffer_size).to(device)
        self.loc_diff = torch.zeros((self.buffer_size,2)).to(device)
        self.index = 0

        self.writer = writer

    def reset_state(self):
        self.states = torch.zeros((self.buffer_size, self.mec_module*2, self.mec_x, self.mec_y)).to(self.device)
        self.dones = torch.zeros((self.buffer_size,)).to(self.device)
        self.log_probs = torch.zeros((self.buffer_size, )).to(self.device)
        self.values = torch.zeros((self.buffer_size,)).to(self.device)
        self.rewards = torch.zeros(self.buffer_size).to(self.device)
        self.advantages = torch.zeros(self.buffer_size).to(self.device)
        self.actions = torch.zeros(self.buffer_size).to(self.device)
        self.loc_diff = torch.zeros((self.buffer_size, 2)).to(self.device)
        self.index = 0

    def add_state(self, state, done, logp, value, r, a, loc_diff):
        if self.index < self.buffer_size:
            self.states[self.index].copy_(state)
            self.dones[self.index] = done
            self.log_probs[self.index].copy_(logp)
            self.values[self.index].copy_(value)
            self.rewards[self.index] = r
            self.actions[self.index] = a
            self.loc_diff[self.index] = loc_diff

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

    def get_buffer(self):
        loss_mask = torch.zeros(self.buffer_size).to(self.device).bool()
        loss_mask[:self.index] = 1
        old_states = self.states[loss_mask]
        old_actions = self.actions[loss_mask]
        old_logp = self.log_probs[loss_mask]
        old_values = self.values[loss_mask]
        diff = self.loc_diff[loss_mask]
        return old_states, old_actions, old_logp, old_values, diff

    def a2c_loss(self, final_value, entropy):
        loss_mask = torch.zeros(self.buffer_size).to(self.device).bool()
        loss_mask[:self.index] = 1
        # print("check mask", loss_mask.sum())
        # loss_mask = torch.BoolTensor(loss_mask).to(self.device)
        entropy /= self.index
        # calculate advantage
        # i.e. how good was the estimate of the value of the current state
        rewards = self._discount_rewards(final_value, discount=0.9)
        print(rewards)
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
        if self.gloabl_step%100==0:
            self.entropy_coeff = max(0.01, self.entropy_coeff-0.02)

        if self.writer is not None:
            self.writer.add_scalar("entropy", entropy, self.gloabl_step)
            self.writer.add_scalar("policy_loss", policy_loss.item(), self.gloabl_step)
            self.writer.add_scalar("value_loss", value_loss.item(), self.gloabl_step)
            self.writer.add_histogram("advantage", advantage.detach(), self.gloabl_step)
            self.writer.add_histogram("rewards", rewards.detach(), self.gloabl_step)
            self.writer.add_histogram("action_prob", self.log_probs.detach(), self.gloabl_step)
            self.gloabl_step += 1

        print("loss ", loss.item(), "policy loss", policy_loss.item(), "value", value_loss.item(),
              "adv", advantage.mean().item(), "entropy", entropy)

        return loss

def simple_train_loc(model, cache_code, device):  # TODO 测试完记得把conv in_c * 2
    # device = torch.device('cuda:1')
    # cache_code = np.load("./policy_ckpt/mec_data.npy", allow_pickle=True).item()
    # u_mecs = cache_code["u_mecs"]
    # locs = cache_code["locs"]
    # model = GridPolicy(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    Mse = torch.nn.MSELoss()
    # def get_code(pos):
    #     if pos.shape[0]>2:
    #         loc_sim = np.linalg.norm(locs[:, np.newaxis, :] - pos[np.newaxis, ...], axis=-1) # 1000*n
    #         g_code = u_mecs[np.argmin(loc_sim, axis=0)]
    #     else:
    #         loc_sim = np.linalg.norm(locs - pos, axis=1)
    #         g_code = u_mecs[np.argmin(loc_sim)]
    #     return g_code
    for i in range(3000):
        p = np.random.random((20, 2))
        code = cache_code.get_grid_code(p).copy()

        code = torch.Tensor(code).to(device).transpose(2,1).reshape(p.shape[0], 7, 10, 10)
        p_ = p.reshape((10,2,2))
        code_ = code.reshape(10, 7*2,10,10)
        p_diff = p_[:,0,:]-p_[:,1,:]
        p_y = model.get_loc(code_)
        p_diff = torch.Tensor(p_diff).to(device)
        p_y = torch.Tensor(p_y).to(device)
        loss = Mse(p_diff, p_y)
        if i % 10 == 0:
            print("step ", i, "loss", loss.mean().item(), p_diff[0], p_y[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    # policy = GridPolicy(device=torch.device('cuda:0'), mec_module_num=7)
    # grid_input = torch.randn((1, 7, 100))
    # act, logp, entropy, value = policy.get_actions(grid_input, train=True)
    simple_train_loc()