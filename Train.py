import gc
import os

import brainpy as bp
# from HPC import Hippocampus_2D
# from  MEC import Grid_2D
from Coupled_model import Coupled_Net, make_coupled_net
from env import Env, ConfigParam
from util import *
import jax
import time
import datetime

devices = jax.devices()
selected_device = devices[1]

def training_func(Coupled_Model, lapnum, config,
                  train=1, v_abs=0.01, dy=0.1, get_loc=0., get_view=1, reset=0):
    locs, loc_feas, velocitys, total_time = config.env.get_train_traj(T=lapnum/v_abs/dy * 2, v=v_abs, dy=dy)
    velocitys = bm.array(velocitys)
    locs = bm.array(locs)
    loc_feas = bm.array(loc_feas)
    # 复位到出发点上
    if reset:
        Coupled_Model.reset_state(locs[0])
        def initialize_net(i):  # 20 x size
            Coupled_Model.step_run(i, velocity=bm.zeros(2, ), loc=locs[0], loc_fea=loc_feas[0],
                                   get_loc=1, get_view=0, train=0)

        indices = bm.arange(2000)
        bm.for_loop(initialize_net, indices, progress_bar=True)

    # 沿轨迹运动
    def run_net(i, velocity, loc, loc_fea):  # 20 x size
        Coupled_Model.step_run(i, velocity=velocity, loc=loc, loc_fea=loc_fea,
                               get_view=get_view, get_loc=get_loc, train = train)

    # Coupled_Model.sen2hpc_stre = 1.
    indices = np.arange(total_time)
    bm.for_loop(run_net, (indices, velocitys, locs, loc_feas), progress_bar=True)


def testing_func(Coupled_Model, config, v_abs=0.02, init_get_loc=0., get_view=1,
                 reset_stre=None, test_traj=True, test_module_name="hpc"):
    # construct trajectories
    if test_traj:
        locs, loc_feas, velocitys, total_time = config.env.get_test_traj(T=3200, v_max=v_abs)
    else:
        dy = 0.02
        locs, loc_feas, velocitys, total_time = config.env.get_train_traj(T=1./ v_abs / dy * 2, v=v_abs, dy=dy)
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
                               get_loc=init_get_loc, get_view=0, train=0)

    # Coupled_Model.sen2hpc_stre = 10.
    indices = np.arange(2000)
    bm.for_loop(initialize_net, indices, progress_bar=True)

    # 运动时mec信息更多，其他信息也存在，适当提高sen比重
    Coupled_Model.mec2hpc_stre = 1.
    Coupled_Model.sen2hpc_stre = 1.
    Coupled_Model.hpc2mec_stre = 1.
    def run_net(i, velocity, loc, loc_fea):  # 20 x size
        Coupled_Model.step_run(i, velocity=velocity, loc=loc, loc_fea=loc_fea,
                               get_view=get_view, get_loc=0, train=0)
        u_HPC = Coupled_Model.HPC_model.u
        r_HPC = Coupled_Model.HPC_model.r
        I_mec = Coupled_Model.I_mec  # mec 输入到hpc(place cell)的信号
        I_sen = Coupled_Model.I_sen  # ovc 输入到place cell的信号
        u_mec = Coupled_Model.u_mec_module
        # input_hpc_module = Coupled_Model.input_hpc_module
        if test_module_name == "mec":
            return u_mec, r_HPC, I_mec, I_sen
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
    else:
        u_HPCs, r_HPCs, I_mecs, I_sens = bm.for_loop(run_net, (indices, velocitys, locs, loc_feas), progress_bar=True)
        return u_HPCs, r_HPCs, I_mecs, I_sens, locs

def train(env_step=2):
    Pre_train_lap_num = 3
    train_lap_num = 5
    # if env_step == 2:
    #     Pre_train_lap_num = 1
    #     train_lap_num = 2
    config.env.env_index = env_step

    # directory
    directory = f"ratio{config.mec_max_ratio}_lap{train_lap_num}_" + datetime.datetime.now().strftime("%Y-%m-%d-%H")
    os.makedirs(directory, exist_ok=True)

    get_loc = 0  # Grid cell: path integration or not
    # get_view = 1  # Partial observation
    # 训练+测试

    # 1. 测试未训练时的细胞情况
    t0 = time.time()
    thres = 3.5
    HPC_u_before, _,_,_,_ = testing_func(Coupled_Model=Coupled_Model, init_get_loc=0, get_view=1,
                                         config=config, reset_stre=[0., 3., 1.], test_traj=True)
    print("check HPC u shape", HPC_u_before.shape)
    max_fr_before, place_index_before, place_num_before = place_cell_select_fr(HPC_u_before, thres=thres)
    print('Before learning: Place cell number = {}, max fr={}'.format(place_num_before, max_fr_before))

    # First step: Preplay, no interaction between brain regions，首先要求各个脑区不交流，只有mec2hpc的，预训练一段
    Coupled_Model.sen2hpc_stre = 0.
    Coupled_Model.hpc2mec_stre = 0.
    training_func(Coupled_Model, lapnum=Pre_train_lap_num, config=config,
                  v_abs=0.02, dy=0.02,
                  get_loc=1, get_view=1, train=1, reset=True)

    filename_hpc = directory + f'/Coupled_Model_1_env{env_step}'
    bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())

    # Second step: training from LV to HPC
    Coupled_Model.hpc2mec_stre = 1.
    Coupled_Model.sen2hpc_stre = 3.  # sense比重调大
    training_func(Coupled_Model, lapnum=train_lap_num, config=config,
                  v_abs=0.01, dy=0.01,
                  get_loc=get_loc, get_view=1, train=2, reset=True)

    filename_hpc = directory + f'/Coupled_Model_2_env{env_step}'
    bp.checkpoints.save_pytree(filename_hpc, Coupled_Model.state_dict())

    HPC_u_after, _, _, _, _ = testing_func(Coupled_Model=Coupled_Model, init_get_loc=0, get_view=1,
                                            config=config, reset_stre=[0., 3., 1.], test_traj=True)

    max_fr_after, _, place_num_after = place_cell_select_fr(HPC_u_after, thres=thres)
    print('After learning: Place cell number = {}, max_fr={}'.format(place_num_after, max_fr_after))

    # 检查矩阵中是否存在 NaN
    # W_rec_after = Coupled_Model.HPC_model.W_E2E
    # has_nan = np.isnan(W_rec_after).any()
    # print("W_rec中是否含有 NaN:", has_nan)
    # print(W_rec_after)
    # W_sen = Coupled_Model.HPC_model.W_sen
    # W_sen_back = Coupled_Model.HPC_model.W_sen_back  # sen_back暂时没有训练
    # print(W_sen, W_sen_back)


def test(directory, env_step=2, prefix="", load_ckpt=False):
    config.env.env_index = env_step
    if load_ckpt:
        filename_hpc = directory + 'Coupled_Model_2_env1.bp'
        state_dict = bp.checkpoints.load_pytree(filename_hpc)  # load the state dict
        bp.load_state(Coupled_Model, state_dict)  # unpack the state dict and load it into the network
    # Coupled_Model.sen2hpc_stre = 1.
    # Coupled_Model.hpc2mec_stre = 1.
    # Coupled_Model.mec2hpc_stre = 1.

    init_get_loc = 0
    get_view = 1
    hpc_u, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, config=config,
                                                    init_get_loc=init_get_loc, get_view=get_view, reset_stre=[0.,10.,10.],
                                                    test_traj=False)
    thres = 3.5
    place_info_path = plot_place_data(hpc_u=hpc_u, hpc_fr=hpc_fr, I_mec=I_mec, I_sen=I_sen,
                                 loc=loc, env=env, step=f"step2_{env_step}{prefix}", dir=directory, thres=thres)
    print(place_info_path)
    del hpc_u, hpc_fr, I_mec, I_sen, loc
    gc.collect()
    # place_info_path = "./ratio9_lap5_2024-09-09-18/step2place_information.npy"
    # print(place_info_path)
    # TODO 需要在正式测试时看init_get_loc=0的效果才有用
    u_mec, hpc_fr, I_mec, I_sen, loc = testing_func(Coupled_Model=Coupled_Model, config=config,
                                                    init_get_loc=0, get_view=get_view, reset_stre=[0., 3., 1.],
                                                    test_traj=True, test_module_name="mec")
    # 放弃检查多个module中心？
    draw_population_activity(directory, place_info_path, None, hpc_fr, I_mec,
                             I_sen, loc, env, step=f"step2_{env_step}{prefix}")

    # draw_grid_activity(u_mec, directory, step=f"step2_{env_step}")

if __name__ == '__main__':
    with jax.default_device(selected_device):
        # config and env
        env = Env()
        config = ConfigParam(env)
        bm.set_dt(config.dt)
        Coupled_Model = make_coupled_net()

        train(env_step=2)
        test("./ratio9_lap5_2024-09-12-17/", env_step=2, prefix="pure_train", load_ckpt=False)
        # train(env_step=1)
        # test("./ratio9_lap5_2024-09-12-17/", env_step=1)
        # test("./ratio9_lap5_2024-09-12-17/", env_step=2)
