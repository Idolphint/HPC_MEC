import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np
from HPC import Hippocampus_2D
from MEC import Grid_2D
from env import ConfigParam

class Coupled_Net(bp.DynamicalSystemNS):
    def __init__(self, HPC_model, MEC_model_list, num_module, config=None,
                 mec2hpc_stre=1., sen2hpc_stre=1., hpc2mec_stre=0.,
                 sen2HD_stre=0.):
        super(Coupled_Net, self).__init__()
        self.config = config
        self.HPC_model = HPC_model
        self.MEC_model_list = MEC_model_list
        self.num_module = num_module
        # landmark/boundaries parameters
        # sensory input numbers
        self.num_sen = config.num_sen
        num_mec = MEC_model_list[0].num
        # initialize dynamical variables
        self.r_sen = bm.Variable(bm.zeros(self.num_sen, ))
        self.input_back = bm.Variable(bm.zeros(self.num_sen, ))
        self.u_mec_module = bm.Variable(bm.zeros([num_mec, num_module]))
        self.input_hpc_module = bm.Variable(bm.zeros([num_mec, num_module]))
        self.I_mec = bm.Variable(bm.zeros(HPC_model.num, ))  # MEC 输入给HPC的部分
        self.I_sen = bm.Variable(bm.zeros(HPC_model.num, ))
        self.mec2hpc_stre = mec2hpc_stre
        self.sen2hpc_stre = sen2hpc_stre
        self.hpc2mec_stre = hpc2mec_stre

    def reset_state(self, loc=None):
        self.HPC_model.reset_state()
        if loc is None:
            loc = bm.array([0.,0.])
        for MEC_model in self.MEC_model_list:
            MEC_model.reset_state(loc)

    def update(self, velocity, loc, loc_fea, get_loc=0, get_view=0, train=0, v_noise=0.001):
        """
        :param velocity:
        :param loc:
        :param loc_fea: 根据pos计算出的feature
        :param get_view: 暂时不用
        :param get_loc: =1代表使用真实的pos而不是积分得到的pos
        :param train:
        :return:
        """
        bound_effect = 1
        # Update MEC states
        r_hpc = self.HPC_model.r
        I_mec = bm.zeros(self.HPC_model.num, )

        for i in range(self.num_module):
            MEC_model = self.MEC_model_list[i]
            MEC_model.update(pos=loc, velocity=velocity, r_hpc=r_hpc, hpc2mec_stre=self.hpc2mec_stre,
                             train=train, get_loc=get_loc, debug=i == 1, v_noise=v_noise)
            r_mec = MEC_model.r
            self.u_mec_module[:, i] = r_mec
            r_mec = r_mec - 0.6 * bm.max(r_mec)
            # r_mec = bm.where(r_mec<0, 0, r_mec)
            I_mec_module = bm.matmul(MEC_model.conn_out, r_mec)  # r_mec自然地分成一半正一半负

            I_mec += I_mec_module
            # jax.debug.print("{}: check i mec {} {} r_mec {} {} conn {}", i, bm.max(I_mec_module),
            #                 I_mec_module[bm.argmax(r_hpc)], bm.max(r_mec), bm.sum(r_mec), bm.sum(MEC_model.conn_out, axis=1))
            input_hpc = bm.matmul(MEC_model.conn_in, r_hpc)  # 从hpc输入到MEC的量
            input_hpc = bm.where(input_hpc > 0, input_hpc, 0)
            self.input_hpc_module[:, i] = input_hpc
        # jax.debug.print("check input hpc {}", self.input_hpc_module[:0])
        I_mec = bm.where(I_mec > 0, I_mec, 0)

        # Update Sensory inputs
        I_fea = loc_fea

        # Feedback conn
        W_sen_back = self.HPC_model.W_sen_back
        self.input_back = bm.matmul(W_sen_back, r_hpc)
        r_sen = I_fea.flatten() + get_view * self.input_back  # 使用sen_back以矫正sen的中心
        # jax.debug.print("check I_fea sen_back {} {}", I_fea.max(), self.input_back.max())
        r_sen = bm.where(r_sen > 1, 1, r_sen)

        self.r_sen.value = bound_effect * I_fea.flatten()
        self.I_mec.value = bound_effect * self.mec2hpc_stre * I_mec
        self.I_sen = bm.matmul(self.HPC_model.W_sen, r_sen) * self.sen2hpc_stre  # I_sen是经过HPC输入权重调制的
        # Update Hippocampus states
        self.HPC_model.update(I_mec=self.I_mec, I_sen=self.I_sen, I_OVCs=I_fea.flatten(), train=train)


def make_coupled_net(config=None):
    if config is None:
        config = ConfigParam()
    ratio = np.linspace(5.0, 40, config.num_mec_module)  # TODO 确认是否0，1长度的环境这个周期可以？
    strength = np.linspace(1.5, 11, config.num_mec_module)
    angle = np.linspace(0, np.pi / 3, config.num_mec_module)

    # model
    HPC_model = Hippocampus_2D(config=config)
    MEC_model_list = bm.NodeList([])
    for i in range(config.num_mec_module):
        MEC_model_list.append(Grid_2D(ratio=ratio[i], angle=angle[i], strength=strength[i], config=config))
    Coupled_Model = Coupled_Net(HPC_model=HPC_model, MEC_model_list=MEC_model_list,
                                num_module=config.num_mec_module, config=config)
    return Coupled_Model