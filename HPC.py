import brainpy as bp
import brainpy.math as bm
import numpy as np
from util import *
import jax

class Hippocampus_2D(bp.DynamicalSystemNS):
    def __init__(self, config=None):
        super(Hippocampus_2D, self).__init__()
        # dynamical parameters
        self.tau = config.tau  # The synaptic time constant
        self.tau_v = config.tau_v
        self.k = config.k_exc
        self.m = config.mbar_hpc * config.tau / config.tau_v
        # neuron number
        self.num = config.num_hpc
        self.num_sen = config.num_sen
        # Learning parameters
        self.spike_num_exc = config.spike_num_exc
        self.tau_W = config.tau_W
        self.tau_Sen = config.tau_W_sen
        self.norm_sen_exc = config.norm_sen_exc
        self.lr_sen_exc = config.lr_sen_exc
        self.lr_exc = config.lr_exc
        self.lr_back = config.lr_back
        # variables
        # Neural state
        self.r = bm.Variable(bm.zeros([self.num, ]))
        self.u = bm.Variable(bm.zeros([self.num, ]))
        self.v = bm.Variable(bm.zeros([self.num, ]))
        self.r_learn = bm.Variable(bm.zeros([self.num, ]))  # Neuron waiting to learn
        # Conn matrix
        self.W_sen_back = bm.Variable(bm.zeros([self.num_sen, self.num]))
        self.W_E2E = bm.Variable(bm.zeros([self.num, self.num]))
        self.corr = bm.Variable(bm.ones((self.num, self.num_sen)))
        self.W_sen = bm.Variable(bm.abs(bm.random.normal(0, 0.1, (self.num, self.num_sen))))
        # column_sums = bm.sum(self.W_sen, axis=1)
        # jax.debug.print("check norm {} {} {}", bm.min(column_sums), bm.max(self.W_sen), column_sums)

        # self.W_sen.value = randomize_matrix(self.W_sen, percent_zeros=percentage)
        self.W_sen.value = normalize_rows(self.W_sen) * self.norm_sen_exc
        # self.J0 = 0.001
        # self.a = 0.45
        # 定义积分器
        self.integral = bp.odeint(f=self.derivative)

    def dist(self, d):
        dis = bm.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        return dis

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num, ))
        self.u.value = bm.Variable(bm.zeros(self.num, ))
        self.v.value = bm.Variable(bm.zeros(self.num, ))

    def reset_sen_w(self):
        self.W_sen_back.value = bm.Variable(bm.zeros([self.num_sen, self.num]))
        self.W_sen.value = bm.Variable(bm.abs(bm.random.normal(0, 0.1, (self.num, self.num_sen))))
        self.W_sen.value = normalize_rows(self.W_sen) * self.norm_sen_exc

    def load_sen_w(self, state_dict):
        for k,p in state_dict.items():
            if k == "Hippocampus_2D0":
                for hpc_k, hpc_p in p.items():
                    if hpc_k == "Hippocampus_2D0.W_sen":
                        self.W_sen.value = hpc_p
                        print("W_sen load")
                    elif hpc_k == "Hippocampus_2D0.W_sen_back":
                        self.W_sen_back.value = hpc_p
                        print("W sen back load")

    def conn_update_forward(self, r_sen):
        # 这里试图将HPC的发放和I_sen绑定在一起
        # 最纯粹的hebbing学习，竞争通过函数外的top_K完成
        r_learn = self.r_learn.reshape(-1, 1)  # r_learn就是HPC细胞的发放，选取topK个
        r_sen = r_sen.reshape(-1, 1)  # r_sen是I_ovc, r_sen* W_sen后 shape与hpc_num一致
        # TODO sen是否需要筛选？
        corr_mat_sen = bm.outer(r_learn, r_sen)
        # I_sen = bm.matmul(self.W_sen, r_sen)
        # corr_mat_sen = bm.outer(r_learn, I_sen)
        self.corr = corr_mat_sen
        # tau * dW/dt = r_learn * r_sen * W_sen * lr = r_learn * I_sen
        dW = corr_mat_sen * self.W_sen * self.lr_sen_exc / self.tau_Sen * bm.dt
        W_sen = self.W_sen + dW
        W_sen = bm.where(W_sen > 0., W_sen, 0.)
        self.W_sen.value = normalize_rows(W_sen) * self.norm_sen_exc

    def conn_update_rec(self, thres=3.5):
        r_exc = self.r.reshape(-1, 1)
        r_exc = bm.where(r_exc > thres, r_exc, 0)
        corr_mat_E2E = bm.outer(r_exc, r_exc)
        u_rec = bm.matmul(self.W_E2E, self.r).reshape(-1, 1)
        dW_E = (self.lr_exc * corr_mat_E2E - self.W_E2E * (r_exc * u_rec)) / self.tau_W
        W_E2E = self.W_E2E + dW_E
        self.W_E2E.value = bm.where(W_E2E > 0, W_E2E, 0)

    def conn_update_back(self, r_sen, thres=3.5):
        # Oja学习，第一项保证噪音干扰下的OVC输入和hpc发放趋于一致，所以这项可以跟踪W_sen，后面是为了正则化。总之是为了在噪音下追踪W_sen
        # 目前怀疑是两个权重占比差别太多了
        r_sen = r_sen  # + bm.random.normal(0, 0.01, (self.num_sen,))
        r_sen = r_sen.reshape(-1, 1)
        r_exc = self.r.reshape(-1, 1)
        r_exc = bm.where(r_exc > thres, r_exc, 0)
        r_sen = bm.where(r_sen > 0.3, r_sen, 0)
        corr_mat_sen = bm.outer(r_sen, r_exc)
        u_back = bm.matmul(self.W_sen_back, self.r).reshape(-1, 1)
        dW_sen_back = (self.lr_back * corr_mat_sen - self.W_sen_back * (r_sen * u_back)) / self.tau_W
        W_sen_back = self.W_sen_back + dW_sen_back
        self.W_sen_back.value = bm.where(W_sen_back > 0, W_sen_back, 0)

    @property
    def derivative(self):
        due = lambda ue, t, E_total: (-ue + E_total - self.v) / self.tau
        dve = lambda ve, t: (-ve + self.m * self.u) / self.tau_v
        return bp.JointEq([due, dve])

    def update(self, I_mec, I_sen, I_OVCs, train=0):
        # Calculate the feedforward input
        noise = bm.random.normal(0, 0.1, (self.num))
        # Calculate the recurrent current
        I_rec = bm.matmul(self.W_E2E, self.r)
        # Calculate total input
        E_total = I_rec + I_mec + noise + I_sen
        # update neuron state
        ue, ve = self.integral(self.u, self.v, bp.share['t'], E_total, bm.dt)

        self.u.value = bm.where(ue > 0, ue, 0)
        self.v.value = ve
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        # jax.debug.print("check max hpc idx {} {} {} {} {}", bm.argmax(self.r),
        #                     bm.max(self.r), bm.max(I_sen), bm.max(I_rec), bm.max(I_mec))

        if train == 1:
            self.conn_update_rec()  # from HPC to HPC
        elif train == 2:
            self.r_learn.value = keep_top_n(self.r, self.spike_num_exc)
            self.conn_update_forward(r_sen=I_OVCs)  # from LV to HPC
            self.conn_update_rec()  # from HPC to HPC
            self.conn_update_back(r_sen=I_OVCs)  # from HPC to LV
