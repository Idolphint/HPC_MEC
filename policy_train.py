import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from Coupled_model import make_coupled_net
from env import Env, ConfigParam
from util import init_model, draw_nav_traj
import brainpy as bp
import brainpy.math as bm
from policy import GridPolicy, ReplayBuffer, simple_train_loc
import torch.optim as optim
from Train import testing_func
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0")
title = "ppo+cnn+sinL"
os.makedirs("./log/", exist_ok=True)
writer = SummaryWriter(log_dir="./log/policy_"+datetime.datetime.now().strftime("%Y-%m-%d-%H")+title)

x_grid = np.linspace(0,10,10)
y_grid = np.linspace(0,10,10)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

class CachedGridCode:
    def __init__(self, model, env, v_abs=0.02):
        u_mecs, locs = testing_func(model, env, v_abs/2, init_get_loc=0, get_view=1, reset_stre=[0.,5.,5.],
                                    test_traj=False, test_module_name="only_mec")
        u_mecs = (u_mecs-u_mecs.min()) / (u_mecs.max()-u_mecs.min())
        data = {"u_mecs": bm.as_numpy(u_mecs).copy(), "locs": bm.as_numpy(locs).copy()}
        np.save(file="./policy_ckpt/mec_data.npy", arr=data, allow_pickle=True)
        self.u_mecs = bm.as_numpy(u_mecs).copy()
        self.locs = bm.as_numpy(locs).copy()

    def get_grid_code(self, pos):
        loc_sim = np.linalg.norm(self.locs - pos, axis=1)
        g_code = self.u_mecs[np.argmin(loc_sim)]
        return g_code

def ready_for_new_env(model, env):
    loc, fea, r, d, _ = env.reset()
    init_model(model, loc, fea)
    locs, loc_feas, velocitys, total_time = env.get_line_traj(start=env.start_pos, end=env.goal_pos, v_abs=0.01)
    velocity = bm.array(velocitys)
    locs = bm.array(locs)
    loc_feas = bm.array(loc_feas)
    indices = bm.arange(total_time)

    def run_net_policy(i, v, loc, loc_fea, get_view=1):
        model.step_run(i, velocity=v, loc=loc, loc_fea=loc_fea,
                       get_view=get_view, get_loc=0, train=0)
        r_mec = model.u_mec_module
        return r_mec
    r_mecs = bm.for_loop(run_net_policy, (indices, velocity, locs, loc_feas), progress_bar=True)
    final_grid_code = r_mecs[-1]
    print("check final grid code", final_grid_code[:, 0].T)

    # 重新初始化回到原点
    init_model(model, loc, fea)
    return final_grid_code

def train(policy_ckpt=None):
    # 初始化建图模型
    map_model = make_coupled_net(config)
    state_dict = bp.checkpoints.load_pytree(map_model_ckpt_path)  # load the state dict
    bp.load_state(map_model, state_dict)
    v_abs = 0.02  # TODO 如果和grid code设置不同可能有误差
    grid_code_cache = CachedGridCode(map_model, env)

    # 初始化策略模型
    policy = GridPolicy(device, 0.90, 7).to(device)  #config.num_mec_module).to(device)
    if policy_ckpt is not None:
        policy.load_state_dict(torch.load(policy_ckpt))

    # 使用坐标预训练
    simple_train_loc(policy, grid_code_cache, device)
    num_episode = 20000
    max_step = 500
    buffer = ReplayBuffer(buffer_size=max_step, mec_module=config.num_mec_module,
                          mec_x_num=config.num_mec, mec_y_num=config.num_mec, device=device, writer=writer)
    optimizer = optim.Adam([
        {"params":policy.conv1.parameters(), "lr": 0},
        {"params":policy.fc2.parameters(), "lr": 0},
    ], lr=1e-3)
    # optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    Mse = torch.nn.MSELoss()
    ckpt_path = "./policy_ckpt/"+datetime.datetime.now().strftime("%Y-%m-%d-%H")+title+'/'
    os.makedirs(ckpt_path, exist_ok=True)
    last_save_r = 0

    # def run_net_policy(i, v, loc, loc_fea, get_view=1):
    #     map_model.step_run(i, velocity=v, loc=loc, loc_fea=loc_fea,
    #                    get_view=get_view, get_loc=0, train=0)
    #     r_mec = map_model.u_mec_module
    #     return r_mec
    loss_step = 0
    def update():
        rewards = buffer._discount_rewards(final_value=0, discount=0.99)[:buffer.index]
        old_states, old_actions, old_logp, old_values, gt_diffs = buffer.get_buffer()
        print(rewards.shape, old_values.shape)
        adv = rewards.detach() - old_values.detach()
        for k in range(10):
            logp, value, dist_entropy, pred_diff = policy.evaluate(old_states, old_actions)
            loc_loss = Mse(pred_diff, gt_diffs)

            value = torch.squeeze(value)
            ratios = torch.exp(logp - old_logp.detach())

            surr1 = ratios*adv
            surr2 = torch.clamp(ratios, 1-0.2,1+0.2)*adv
            policy_loss = -torch.min(surr1, surr2)
            value_loss = Mse(value, rewards)
            loss = policy_loss + 0.5 * value_loss - 0.01*dist_entropy + 10*loc_loss*(k==0)
            # loss = loc_loss
            writer.add_scalar("policy_loss", policy_loss.mean().item(), loss_step)
            writer.add_scalar("value_loss", value_loss.mean().item(), loss_step)
            writer.add_scalar("entropy", dist_entropy.mean().item(), loss_step)
            writer.add_scalar("loc diff", loc_loss.mean().item(), loss_step)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()


    for i in range(num_episode):
        loc, fea, r, d, _ = env.reset()
        final_grid_code = grid_code_cache.get_grid_code(env.goal_pos)
        grid_code_now = grid_code_cache.get_grid_code(env.agent_pos)
        gt_diff = torch.Tensor(env.goal_pos - env.agent_pos).to(device)
        # final_grid_code = ready_for_new_env(map_model, env)
        # grid_code_now = map_model.u_mec_module.value
        episode_entropy = 0
        epi_r = 0
        for j in range(max_step):
            t0 = time.time()
            state = bm.as_numpy(bm.cat([final_grid_code,grid_code_now], axis=1)).copy()
            # if j % 100 == 0:
            #     plt.scatter(x_mesh, y_mesh, c=state[:,np.random.randint(7)], s=28)
            #     plt.savefig(f"./tmp/input_state_{i}_{j}.jpg")
            #     print("save", i, j)
            state = torch.Tensor(state).to(device)
            # state = state.reshape(1, 2, config.num_mec, config.num_mec)
            # 200,7 ->7,200 -> 1, 7, 10, 10
            state = state.transpose(1,0).reshape(1, config.num_mec_module*2, config.num_mec, config.num_mec)
            # print("check state shape", state.shape)
            if j % 50==0:
                print("pos and tar", env.goal_pos, env.agent_pos)
            action, logp, entropy, value = policy.get_actions(state)
            episode_entropy += entropy
            # print("model get action using", time.time()-t0)
            t0 = time.time()
            act = action.detach().cpu().numpy().flatten() if False else action.item()
            loc, fea, reward, done, v_vec = env.step(act, v_abs=v_abs)
            buffer.add_state(state.squeeze(0), done, logp, value, reward, action.detach(), gt_diff)
            gt_diff = torch.Tensor(env.goal_pos - env.agent_pos).to(device)
            # map_model.update(bm.array(v_vec), bm.array(loc), bm.array(fea), get_loc=0, get_view=1, train=0)
            # grid_code_now = map_model.u_mec_module
            # grid_code_now = run_net_policy(1, bm.array(v_vec), bm.array(loc), bm.array(fea), get_view=1)
            grid_code_now = grid_code_cache.get_grid_code(loc)
            # print("env update using", time.time()-t0)
            if done:
                epi_r = reward
                break
        writer.add_scalar("train_epi_r", epi_r, global_step=i)
        print("finish one episode", i, epi_r)
        with torch.no_grad():
            # state = final_grid_code - grid_code_now
            # state = torch.Tensor(bm.as_numpy(state).copy()).to(device)
            # state = state.transpose(1, 0).reshape(1, config.num_mec_module, config.num_mec, config.num_mec)
            # _, _, _, final_value = policy.get_actions(state)
            final_value = torch.zeros(1).to(device)

        # optimizer.zero_grad()
        # loss = buffer.a2c_loss(final_value, episode_entropy)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm(policy.parameters(), max_norm=0.5)
        # optimizer.step()
        update()
        loss_step+=1
        buffer.reset_state()

        # 定时更新action_std, 3000次训练后保持std=0.1
        if i % 30 == 0:
            policy.set_action_std(max(0.1, 1-i//30*0.01))

        # 定时测试并保存
        if i % 30 == 0:
            avg_r = test(grid_code_cache, policy, test_step_num=500)
            print("test avg r", avg_r)
            if avg_r > last_save_r:
                last_save_r = avg_r
                torch.save(policy.state_dict(), ckpt_path+"best_policy.pt")

def test(grid_code_cache=None, policy=None, test_step_num=10000, draw=False):
    # 初始化建图模型
    v_abs = 0.02
    # 初始化policy
    if policy is None:
        policy = GridPolicy(device, 0.1, config.num_mec_module).to(device)
        policy.load_state_dict(torch.load(policy_ckpt))

    # 初始化cache
    if grid_code_cache is None:
        map_model = make_coupled_net()
        state_dict = bp.checkpoints.load_pytree(map_model_ckpt_path)  # load the state dict
        bp.load_state(map_model, state_dict)
        grid_code_cache = CachedGridCode(map_model, env)

    # data use for visual
    test_traj = []
    done_idx = []
    done = True
    final_grid_code = None
    grid_code_now = None
    total_reward = 0
    for i in range(test_step_num):
        if done:
            print("when done", i, env.goal_pos, env.start_pos)
            loc, fea, r, d, _ = env.reset()
            final_grid_code = grid_code_cache.get_grid_code(env.goal_pos)
            grid_code_now = grid_code_cache.get_grid_code(env.agent_pos)

            done_idx.append(i)
            test_traj.append(loc.copy())
            # final_grid_code = ready_for_new_env(map_model, env)
            # grid_code_now = map_model.u_mec_module
        # state = final_grid_code - grid_code_now
        state = bm.as_numpy(bm.cat([final_grid_code,grid_code_now],axis=1)).copy()
        state = torch.Tensor(state).to(device)
        # state = torch.Tensor(state[:, 0]).to(device)
        # state = state.reshape(1, 2, config.num_mec, config.num_mec)
        # 1, 7, 10, 10
        # 200,7 ->7,200 -> 1, 7, 10, 10
        # state = state.transpose(1, 0).reshape(1, config.num_mec_module * 2, config.num_mec, config.num_mec)
        state = state.transpose(1, 0).reshape(1, config.num_mec_module*2, config.num_mec, config.num_mec)
        action, logp, entropy, value = policy.get_actions(state, train=False)
        act = action.detach().cpu().numpy().flatten() if False else action.item()
        loc, fea, reward, done, v_vec = env.step(act, v_abs=v_abs)
        # print("step ", i, "loc=", env.agent_pos, grid_code_now[:5, 0])
        test_traj.append(loc.copy())
        total_reward += reward
        grid_code_now = grid_code_cache.get_grid_code(env.agent_pos)
        # grid_code_now = run_net_policy(1, bm.array(v_vec), bm.array(loc), bm.array(fea), get_view=1)
    print("total reward", total_reward)

    # 开始绘图
    if total_reward >= 1 and draw:
        test_traj = np.array(test_traj)

        print("begin draw", test_traj)
        for di in range(1, len(done_idx)):
            part_traj = test_traj[done_idx[di-1]: done_idx[di]]
            grid_code_traj = []
            for p in part_traj:
                code_p = grid_code_cache.get_grid_code(p)
                # print("check code p", p, code_p[:5, 0])
                grid_code_traj.append(code_p)
            part_code = np.array(grid_code_traj).transpose([0,2,1]).reshape(len(part_traj), 7, 10, 10)
            # part_code = grid_code_traj[done_idx[di-1]: done_idx[di]]
            data = {"traj": part_traj, "code": part_code}
            np.save(f"./tmp/saved_traj{done_idx[di]}.npy", data)
            draw_nav_traj(part_traj, part_code, f'{done_idx[di-1]}_{done_idx[di]}')

    return total_reward / test_step_num

def check_traj_code():
    map_model = make_coupled_net()
    state_dict = bp.checkpoints.load_pytree(map_model_ckpt_path)  # load the state dict
    bp.load_state(map_model, state_dict)
    grid_code_cache = CachedGridCode(map_model, env)
    traj = []
    code = []
    pos = np.linspace(-np.pi, np.pi, 200)
    for i in range(200):
        p = [np.sin(pos[i])*0.48 + 0.48, np.cos(pos[i])*0.48 + 0.48]
        traj.append(p)
        code.append(grid_code_cache.get_grid_code(p))
    draw_nav_traj(np.array(traj), np.array(code).transpose([0,2,1]).reshape(len(code), 7, 10, 10), "pure_traj")

if __name__ == '__main__':
    map_model_ckpt_path = "./ratio9_lap3_ratio-2024-09-20-11/Coupled_Model_2_env1.bp"
    env = Env(1)
    config = ConfigParam()
    bm.set_dt(config.dt)
    policy_ckpt = None # "./policy_ckpt/best_policy.pt"
    # test(test_step_num=500, draw=True)
    train(policy_ckpt)
    # check_traj_code()