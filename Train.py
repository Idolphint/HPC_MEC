import gc
import os

import brainpy as bp
import numpy as np

# from HPC import Hippocampus_2D
# from  MEC import Grid_2D
from Coupled_model import Coupled_Net, make_coupled_net
from env import Env, ConfigParam
from util import *
import jax
from MEC import Grid_2D
import time
import datetime

devices = jax.devices()
selected_device = devices[0]
# np.random.seed(817)
def training_func(Coupled_Model, lapnum, env,
                  train=1, v_abs=0.01, dy=0.1, get_loc=0., get_view=1, init_by_sen=True,
                  reset_stre=None, run_stre=None):
    locs, loc_feas, velocitys, total_time = env.get_train_traj(T=lapnum/v_abs/dy * 2, v=v_abs, dy=dy)
    velocitys = bm.array(velocitys)
    locs = bm.array(locs)
    loc_feas = bm.array(loc_feas)

    def initialize_net(i):  # 初始化net完全没必要用motion，要么是loc，要么仅sense
        Coupled_Model.step_run(i, velocity=bm.zeros(2, ), loc=locs[0], loc_fea=loc_feas[0],
                               get_loc=-1 if init_by_sen else 1, get_view=1, train=0)

    if reset_stre:
        Coupled_Model.mec2hpc_stre = reset_stre[0]
        Coupled_Model.sen2hpc_stre = reset_stre[1]
        Coupled_Model.hpc2mec_stre = reset_stre[2]
    # 复位到出发点上
    if init_by_sen:  # 此时Init loop中get_loc=-1，完全使用sensory信息找到grid C
        Coupled_Model.reset_state()
    else:  # 依赖准确的loc复位，init loop也会使用loc信息初始化
        Coupled_Model.reset_state(locs[0])

    indices = bm.arange(500)
    bm.for_loop(initialize_net, indices, progress_bar=True)

    if run_stre:
        Coupled_Model.mec2hpc_stre = run_stre[0]
        Coupled_Model.sen2hpc_stre = run_stre[1]
        Coupled_Model.hpc2mec_stre = run_stre[2]
    # 沿轨迹运动
    def run_net(i, velocity, loc, loc_fea):  # 20 x size
        Coupled_Model.step_run(i, velocity=velocity, loc=loc, loc_fea=loc_fea,
                               get_view=get_view, get_loc=0, train = train,
                               v_noise=0. if get_loc==1 else 0.0001)  # 只靠sen初始化意味着普通，否则要靠真实的loc

    # Coupled_Model.sen2hpc_stre = 1.
    indices = np.arange(total_time)
    bm.for_loop(run_net, (indices, velocitys, locs, loc_feas), progress_bar=True)


def testing_func(Coupled_Model, env, v_abs=0.02, init_get_loc=0., get_view=1,
                 reset_stre=None, test_traj=True, test_module_name="hpc"):
    # construct trajectories
    if test_traj:
        locs, loc_feas, velocitys, total_time = env.get_test_traj(T=3200, v_max=v_abs)
    else:
        dy = 0.02
        locs, loc_feas, velocitys, total_time = env.get_train_traj(T=1./ v_abs / dy * 2, v=v_abs, dy=dy)
    velocitys = bm.array(velocitys)
    locs = bm.array(locs)
    loc_feas = bm.array(loc_feas)

    # 初始化不指定loc，那么只有sen是可信的,因此hpc2mec部分可信 sen2hpc>0 hpc2mec>0 mec2hpc=0
    Coupled_Model.reset_state(locs[0])
    Coupled_Model.mec2hpc_stre = reset_stre[0]
    Coupled_Model.sen2hpc_stre = reset_stre[1]
    Coupled_Model.hpc2mec_stre = reset_stre[2]
    def initialize_net(i):  # 20 x size
        Coupled_Model.step_run(i, velocity=bm.zeros(2, ), loc=locs[0], loc_fea=loc_feas[0],
                               get_loc=init_get_loc, get_view=1, train=0)

    # Coupled_Model.sen2hpc_stre = 10.
    # get_loc_tmp = -1
    # indices = np.arange(200)
    # bm.for_loop(initialize_net, indices, progress_bar=True)
    indices = np.arange(2000)
    bm.for_loop(initialize_net, indices, progress_bar=True)

    # 运动时mec信息更多，其他信息也存在，适当提高sen比重
    Coupled_Model.mec2hpc_stre = 1.
    Coupled_Model.sen2hpc_stre = 1.
    Coupled_Model.hpc2mec_stre = 1.
    def run_net(i, velocity, loc, loc_fea):  # 20 x size
        Coupled_Model.step_run(i, velocity=velocity, loc=loc, loc_fea=loc_fea,
                               get_view=get_view, get_loc=0, train=0,
                               v_noise=0. if not test_traj else 0.0001)
        u_HPC = Coupled_Model.HPC_model.u
        r_HPC = Coupled_Model.HPC_model.r
        I_mec = Coupled_Model.I_mec  # mec 输入到hpc(place cell)的信号
        I_sen = Coupled_Model.I_sen  # ovc 输入到place cell的信号
        u_mec = Coupled_Model.u_mec_module
        input_hpc_module = Coupled_Model.input_hpc_module
        hpc_center = Grid_2D.get_center_tmp(input_hpc_module[:, 0])
        max_hpc_in = bm.max(input_hpc_module[:, 0])
        mec_center = Grid_2D.get_center_tmp(u_mec[:, 0])
        if test_module_name == "mec":
            return u_mec, r_HPC, I_mec, I_sen
        elif test_module_name == "mec_hpc":
            return hpc_center, max_hpc_in, mec_center
        elif test_module_name == "hpc":
            return u_HPC, r_HPC, I_mec, I_sen #, u_mec
        elif test_module_name == "only_mec":
            return u_mec
        else:
            return u_HPC, r_HPC, I_mec, I_sen  # , u_mec

    indices = bm.arange(total_time)
    if test_module_name == "only_mec":
        u_MECs = bm.for_loop(run_net, (indices, velocitys, locs, loc_feas), progress_bar=True)
        return u_MECs, locs
    elif test_module_name == "mec_hpc":
        hpc_C, max_in, mec_C = bm.for_loop(run_net, (indices, velocitys, locs, loc_feas), progress_bar=True)
        return hpc_C, max_in, mec_C, locs
    else:
        u_HPCs, r_HPCs, I_mecs, I_sens = bm.for_loop(run_net, (indices, velocitys, locs, loc_feas), progress_bar=True)
        return u_HPCs, r_HPCs, I_mecs, I_sens, locs

def train(title="", init_grid_space=False):
    Pre_train_lap_num = 2
    train_lap_num = 1

    # directory
    directory = f"ratio{config.mec_max_ratio}_"+ title + datetime.datetime.now().strftime("%Y-%m-%d-%H") + "/"
    os.makedirs(directory, exist_ok=True)

    get_loc = 0  # Grid cell: path integration or not
    get_view = 1  # Partial observation
    # 训练+测试

    # 1. 测试未训练时的细胞情况
    t0 = time.time()
    # First step: Preplay, no interaction between brain regions，首先要求只有mec2hpc的和脑区内部的，预训练一段
    # Coupled_Model.sen2hpc_stre = 0.
    # Coupled_Model.hpc2mec_stre = 0.
    if Pre_train_lap_num > 0:
        # for pi in range(Pre_train_lap_num):
        #     env.set_env_index(pi%2+1)
        training_func(Coupled_Model, lapnum=Pre_train_lap_num, env=env,
                      v_abs=0.02, dy=0.02,
                      get_loc=1, get_view=1, train=1, init_by_sen=not init_grid_space,
                      reset_stre=[0., 5., 5.], run_stre=[1., 0., 0.])

        filename_hpc = directory + f'Coupled_Model_1_Prelap{Pre_train_lap_num}'
        bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())
    # Second step: training from LV to HPC
    for sub_train in range(train_lap_num):
        # env.set_env_index(sub_train%2+1)
        # Coupled_Model.hpc2mec_stre = 1.
        # Coupled_Model.sen2hpc_stre = 1.  # sense比重调大
        # Coupled_Model.mec2hpc_stre = 1.
        training_func(Coupled_Model, lapnum=2, env=env,
                      v_abs=0.01, dy=0.01,
                      get_loc=get_loc, get_view=1, train=2, init_by_sen=not init_grid_space,
                      reset_stre=[0., 5., 5.], run_stre=[1., 1., 1.])
        hpc_u, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, env=env,
                                                        init_get_loc=-1, get_view=1,
                                                        reset_stre=[0., 5., 5.],
                                                        test_traj=False)
        plot_place_data(hpc_u=hpc_u, hpc_fr=hpc_fr, I_mec=I_mec, I_sen=I_sen,
                                          loc=loc, env=env, step=f"sub_train_{sub_train}", dir=directory, thres=3.0)

        filename_hpc = directory + f'Coupled_Model_2_step{sub_train}'
        bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())

    # HPC_u_after, _, _, _, _ = testing_func(Coupled_Model=Coupled_Model, init_get_loc=0, get_view=1,
    #                                         env=env, reset_stre=[0., 5., 5.], test_traj=True)
    #
    # max_fr_after, _, place_num_after = place_cell_select_fr(HPC_u_after, thres=thres)
    # print('After learning: Place cell number = {}, max_fr={}'.format(place_num_after, max_fr_after))

    # 检查矩阵中是否存在 NaN
    # W_rec_after = Coupled_Model.HPC_model.W_E2E
    # has_nan = np.isnan(W_rec_after).any()
    # print("W_rec中是否含有 NaN:", has_nan)
    # print(W_rec_after)
    # W_sen = Coupled_Model.HPC_model.W_sen
    # W_sen_back = Coupled_Model.HPC_model.W_sen_back  # sen_back暂时没有训练
    # print(W_sen, W_sen_back)
def train2(title):
    # 有较大的固定偏差0.5左右， TODO check
    directory = f"ratio{config.mec_max_ratio}_" + title + datetime.datetime.now().strftime("%Y-%m-%d-%H") + "/"
    os.makedirs(directory, exist_ok=True)

    Pre_train_lap_num=4
    train_lap_num = 3
    get_loc=0
    env.set_env_index(1)
    training_func(Coupled_Model, lapnum=Pre_train_lap_num, env=env,
                  v_abs=0.02, dy=0.02,
                  get_loc=0, get_view=1, train=1, reset=True,
                  reset_stre=[0., 5., 5.], run_stre=[1., 0., 0.])
    filename_hpc = directory + f'Coupled_Model_en1_Prelap{Pre_train_lap_num}'
    bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())
    for sub_train in range(train_lap_num):
        training_func(Coupled_Model, lapnum=2, env=env,
                      v_abs=0.01, dy=0.01,
                      get_loc=get_loc, get_view=1, train=2, reset=False,
                      reset_stre=[0., 5., 5.], run_stre=[1., 1., 1.])

    hpc_u, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, env=env,
                                                    init_get_loc=-1, get_view=1,
                                                    reset_stre=[0., 5., 5.],
                                                    test_traj=False)
    plot_place_data(hpc_u=hpc_u, hpc_fr=hpc_fr, I_mec=I_mec, I_sen=I_sen,
                    loc=loc, env=env, step=f"ck_env1", dir=directory, thres=3.5)

    filename_hpc = directory + f'Coupled_Model_en1_lap{train_lap_num}'
    bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())

    env.set_env_index(2)
    training_func(Coupled_Model, lapnum=Pre_train_lap_num, env=env,
                  v_abs=0.02, dy=0.02,
                  get_loc=0, get_view=1, train=1, reset=False,
                  reset_stre=[0., 5., 5.], run_stre=[1., 0., 0.])
    filename_hpc = directory + f'Coupled_Model_en2_Prelap{Pre_train_lap_num}'
    bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())
    for sub_train in range(train_lap_num):
        training_func(Coupled_Model, lapnum=2, env=env,
                      v_abs=0.01, dy=0.01,
                      get_loc=get_loc, get_view=1, train=2, reset=False,
                      reset_stre=[0., 5., 5.], run_stre=[1., 1., 1.])
    hpc_u, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, env=env,
                                                    init_get_loc=-1, get_view=1,
                                                    reset_stre=[0., 5., 5.],
                                                    test_traj=False)
    plot_place_data(hpc_u=hpc_u, hpc_fr=hpc_fr, I_mec=I_mec, I_sen=I_sen,
                    loc=loc, env=env, step=f"ck_env2", dir=directory, thres=3.5)
    filename_hpc = directory + f'Coupled_Model_en2_lap{train_lap_num}'
    bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())
def test(directory, env_step=2, prefix="", load_ckpt=None):
    env.env_index = env_step
    if load_ckpt is not None:
        filename_hpc = directory + load_ckpt
        state_dict = bp.checkpoints.load_pytree(filename_hpc)  # load the state dict
        bp.load_state(Coupled_Model, state_dict)  # unpack the state dict and load it into the network
        # 检查conn_out
        # for mi in range(7):
        #     conn_out = Coupled_Model.MEC_model_list[mi].conn_out.value
        #     print("check conn out sum",mi, conn_out.shape, bm.sum(conn_out, axis=0),
        #           bm.max(conn_out, axis=0), bm.min(conn_out, axis=0))
    # Coupled_Model.sen2hpc_stre = 1.
    # Coupled_Model.hpc2mec_stre = 1.
    # Coupled_Model.mec2hpc_stre = 1.
    init_get_loc = 0
    get_view = 1

    hpc_u, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, env=env,
                                                    init_get_loc=init_get_loc, get_view=get_view, reset_stre=[0.,5.,5.],
                                                    test_traj=False)
    thres = 3.5
    place_info_path = plot_place_data(hpc_u=hpc_u, hpc_fr=hpc_fr, I_mec=I_mec, I_sen=I_sen,
                                 loc=loc, env=env, step=f"env{env_step}{prefix}", dir=directory, thres=thres)
    print(place_info_path)
    del hpc_u, hpc_fr, I_mec, I_sen, loc
    gc.collect()
    # place_info_path = "./ratio9_lap5_2024-09-09-18/step2place_information.npy"
    # print(place_info_path)
    # TODO 需要在正式测试时看init_get_loc=0的效果才有用, 训练后init_get_loc会引入固定误差
    u_mec, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, env=env,
                                                    init_get_loc=init_get_loc, get_view=get_view, reset_stre=[0., 5., 5.],
                                                    test_traj=True, test_module_name="mec")
    # 放弃检查多个module中心？
    print("check ")
    draw_population_activity(directory, place_info_path, None, hpc_fr, I_mec,
                             I_sen, loc, env, step=f"env{env_step}{prefix}")

    # draw_grid_activity(u_mec, directory, step=f"step2_{env_step}{prefix}")

if __name__ == '__main__':
    with jax.default_device(selected_device):
        # config and env
        env_step = 1
        env = Env(env_step)
        config = ConfigParam()
        bm.set_dt(config.dt)
        Coupled_Model = make_coupled_net(config)

        # state_dict = bp.checkpoints.load_pytree("./ratio9_lap2_pure12024-09-23-11/Coupled_Model_2_env1.bp")
        # bp.load_state(Coupled_Model, state_dict)  # unpack the state dict and load it into the network

        # Coupled_Model.HPC_model.reset_sen_w()
        # 重新载入旧的sen连接
        # state_dict = bp.checkpoints.load_pytree("./ratio9_lap3_ratio-2024-09-20-11/Coupled_Model_2_env1.bp")
        # Coupled_Model.HPC_model.load_sen_w(state_dict)
        # train(title="after092011_trad", init_grid_space=True)
        # train2(title="mix_senReset")
        # test("./ratio9_after0920112024-09-24-22/", env_step=1, prefix="_after2011_pre", load_ckpt="Coupled_Model_1_Prelap3.bp")
        # train(env_step=1)
        test("./ratio9_lap3_ratio-2024-09-20-11/", env_step=env_step, prefix="pure1", load_ckpt="Coupled_Model_2_env1.bp")
        # test("./ratio9_lap5_2024-09-12-17/", env_step=2)
