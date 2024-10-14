import brainpy as bp
import brainpy.math as bm
import numpy as np
from util import *
import jax

class Grid_2D(bp.DynamicalSystem):
    def __init__(self, ratio, angle, strength, config=None, tau=1., tau_v=10., mbar=1.,
                 k=1., A=3., J0=1., x_min=-bm.pi, x_max=bm.pi):
        # num=num_mec, spike_num=spike_num_grid, num_hpc=num_exc,
        #  tau=1., tau_v=10., mbar=1., tau_e=tau_e,k=1., tau_W_in=tau_W_in,
        # norm_fac = norm_fac, tau_W_out=tau_W_mec, lr_out= lr_mec_out, lr_in=lr_mec_in,
        # a=a_mec, A=10., J0=1., x_min=-bm.pi, x_max=bm.pi):
        super(Grid_2D, self).__init__()
        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.tau_W_out = config.tau_W_mec
        self.tau_W_in = config.tau_W_in
        self.spike_num = config.spike_num_exc
        self.ratio = ratio
        self.module_strength = strength
        self.tau_e = config.tau_e
        self.num_x = config.num_mec  # number of excitatory neurons for x dimension
        self.num_y = config.num_mec  # number of excitatory neurons for y dimension
        self.num = self.num_x * self.num_y
        self.num_hpc = config.num_hpc
        self.k = k / self.num * 20  # Degree of the rescaled inhibition
        self.a = config.a_mec  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0 / self.num * 20  # maximum connection value
        self.m = mbar * tau / tau_v

        # Learning parameters
        self.lr_in = config.lr_mec_in
        self.lr_out = config.lr_mec_out
        self.norm_fac = config.norm_fac
        self.angle = angle

        # feature space
        self.x_range = x_max - x_min
        phi_x = bm.linspace(x_min, x_max, self.num_x + 1)  # The encoded feature values
        self.x = phi_x[0:-1]
        self.y_range = self.x_range
        phi_y = bm.linspace(x_min, x_max, self.num_y + 1)  # The encoded feature values
        self.y = phi_y[0:-1]
        x_grid, y_grid = bm.meshgrid(self.x, self.y)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T
        self.rho = self.num / (self.x_range * self.y_range)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density
        self.coor_transform = bm.array([[1, -1 / bm.sqrt(3)], [0, 2 / bm.sqrt(3)]])
        self.rot = bm.array([[bm.cos(self.angle), -bm.sin(self.angle)], [bm.sin(self.angle), bm.cos(self.angle)]])

        # initialize conn matrix
        self.conn_mat = self.make_conn()
        # jax.debug.print("check conn in {} ", self.conn_mat)
        self.conn_out = bm.Variable(self.make_conn_out())
        self.conn_out.value = self.norm_fac * normalize_rows(self.conn_out)
        self.conn_in = bm.Variable(bm.zeros([self.num, self.num_hpc]))

        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.motion_input = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(bm.zeros(2, ))
        self.centerI = bm.Variable(bm.zeros(2, ))
        self.center_ideal = bm.Variable(bm.zeros(2, ))
        self.phase_estimate = bm.Variable(bm.zeros(2, ))

        # 定义积分器
        self.integral = bp.odeint(method='exp_euler', f=self.derivative)
        # self.path_integral = bp.odeint(method='exp_euler', f=self.derivative_path)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):  # 将相位空间拉成六边形
        d = self.circle_period(d)
        delta_x = d[:, 0] + d[:, 1] / 2
        delta_y = d[:, 1] * bm.sqrt(3) / 2
        dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
        return dis

    def make_conn_out(self):
        matrix = bm.random.normal(0, 0.01, (self.num_hpc, self.num))
        random_indices = np.random.randint(self.num, size=self.num_hpc)
        matrix[bm.arange(self.num_hpc), random_indices] = 1
        return matrix

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(v - self.value_grid)
            Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
            return Jxx

        return get_J(self.value_grid)

    def Postophase(self, pos):
        # 将pos旋转self.angle角，就得到了在本grid cell坐标系下的坐标，由于1HZ的位置只有-pi~pi，所以*ratio就是自己的相位了
        Loc = bm.matmul(self.rot, pos) * self.ratio  # （旋转self.angle角 - 0） * 放缩因子
        phase = bm.matmul(self.coor_transform, Loc) + bm.pi  # 坐标变换
        # jax.debug.print("step1 phase {}", phase)
        phase_x = bm.mod(phase[0], 2 * bm.pi) - bm.pi
        phase_y = bm.mod(phase[1], 2 * bm.pi) - bm.pi
        Phase = bm.array([phase_x, phase_y])
        # jax.debug.print("step2 phase {}", Phase)
        return Phase

    def get_stimulus_by_pos(self, pos):
        assert bm.size(pos) == 2
        Phase = self.Postophase(pos)
        d = self.dist(bm.asarray(Phase) - self.value_grid)
        return self.A * bm.exp(-0.25 * bm.square(d / self.a))


    def get_stimulus_by_motion(self, velocity, noise_stre=0.0, debug=False):
        # center 是6变形空间的相位，所以phase_estimate也是6变形空间相位，而value_grid是4变形相位，求dist却使用4边形空间逻辑？
        # integrate self motion
        noise_v = bm.random.randn(2) * noise_stre  # 1/10的噪音速度
        velocity = velocity + noise_v
        v_rot = bm.matmul(self.rot, velocity)
        v_phase = bm.matmul(self.coor_transform, v_rot * self.ratio)
        dis_phase = self.circle_period(self.center - self.phase_estimate)
        pos_e = self.phase_estimate + (dis_phase / self.tau_e + v_phase) * bm.dt
        # 新版本中不用积分器了，因为可能出错了？
        # pos_e = self.path_integral(self.phase_estimate, bp.share['t'], v_phase, bm.dt)
        self.phase_estimate.value = self.circle_period(pos_e)
        # if debug:
        #     jax.debug.print("v_noise {}, dis phase {} phase {}", v_phase, dis_phase, self.phase_estimate)
        # jax.debug.print("check mec phase estimate {} {} {} ={}", self.phase_estimate.value,v_phase,dis_phase, self.center)
        d = self.dist(self.phase_estimate - self.value_grid)
        fire = self.A * bm.exp(-0.25 * bm.square(d / self.a))
        # if debug:
        #     jax.debug.print("before {} after decode{}", self.phase_estimate, Grid_2D.get_center_tmp(fire))
        # d = self.circle_period(self.phase_estimate - self.value_grid)
        # d = bm.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        # d = self.dist(self.center_ideal - self.value_grid)
        return fire

    def get_center(self):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r > bm.max(self.r) * 0.1, self.r, 0)
        # jax.debug.print("check num grid cell {}", bm.sum(self.r>bm.max(self.r)*0.1))
        self.center[0] = bm.angle(bm.sum(exppos_x * r))
        self.center[1] = bm.angle(bm.sum(exppos_y * r))
        self.centerI[0] = bm.angle(bm.sum(exppos_x * self.input))
        self.centerI[1] = bm.angle(bm.sum(exppos_y * self.input))

    @staticmethod
    def get_center_tmp(input_hpc):
        x_grid, y_grid = bm.meshgrid(bm.linspace(-bm.pi, bm.pi, 10, endpoint=False),
                                     bm.linspace(-bm.pi, bm.pi, 10, endpoint=False))
        exppos_x = bm.exp(1j * x_grid.flatten())
        exppos_y = bm.exp(1j * y_grid.flatten())

        center = bm.zeros(2)
        center[0] = bm.angle(bm.sum(exppos_x * input_hpc))
        center[1] = bm.angle(bm.sum(exppos_y * input_hpc))
        return center

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    @property
    def derivative_path(self):
        dpos = lambda phase_estimate, t, v_phase: self.circle_period(
            self.center - phase_estimate) / self.tau_e + v_phase
        return dpos

    def conn_out_update(self, r_learn_hpc):
        r_learn = r_learn_hpc.reshape(-1, 1)
        r_grid = self.r.reshape(-1, 1)
        corr_out = bm.outer(r_learn, r_grid)
        # conn_out = self.conn_out + (self.lr_out * self.conn_out) / self.tau_W_out * bm.dt
        conn_out = self.conn_out + (self.lr_out * corr_out * self.conn_out) / self.tau_W_out * bm.dt
        conn_out = bm.where(conn_out > 0, conn_out, 0)
        self.conn_out.value = self.norm_fac * normalize_rows(conn_out) * bm.exp(-self.module_strength ** 2 / 5 ** 2)

    def conn_in_update(self, r_hpc, thres=3.5):
        r_hpc = r_hpc.reshape(-1, 1)
        r_hpc = bm.where(r_hpc > thres, r_hpc, 0)
        r_grid = self.r.reshape(-1, 1)
        r_grid = bm.where(r_grid > 0.08, r_grid, 0)
        input_hpc = bm.matmul(self.conn_in, r_hpc)
        corr_in = bm.outer(r_grid, r_hpc)
        # jax.debug.print("check corr update {} {} {} {} {} {}", bm.max(corr_in),
        #                 bm.sum(corr_in > 0.1), bm.sum(r_grid > 0.08),
        #                 bm.max(self.conn_in * (r_grid * input_hpc)),
        #                 self.lr_in, self.tau_W_in)

        conn_in = self.conn_in + (self.lr_in * corr_in - self.conn_in * (r_grid * input_hpc)) / self.tau_W_in * bm.dt
        self.conn_in.value = bm.where(conn_in > 0, conn_in, 0)
        # jax.debug.print("check update{} {}", bm.max(self.conn_in), bm.max(input_hpc))

    def reset_state(self, loc):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))
        # self.center.value = 2 * bm.pi * bm.random.rand(2) - bm.pi
        Phase = self.Postophase(loc)  # center 是相位的确切中心
        # jax.debug.print("mec reset 2 {}", Phase)
        self.phase_estimate.value = Phase
        self.center.value = Phase
        self.center_ideal.value = Phase

    def sampling_by_pos(self, r_hpc, pos, noise_stre=0.01):
        noise = bm.random.normal(0, noise_stre, (self.num,))
        self.motion_input.value = self.get_stimulus_by_pos(pos) + noise
        input_hpc = bm.matmul(self.conn_in, r_hpc)
        self.input.value = self.motion_input + input_hpc
        Irec = bm.matmul(self.conn_mat, self.r)
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share['t'], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        self.get_center()

    def phase_to_feature_vector(self, phase):
        # 计算相位与值网格之间的距离
        d = self.dist(bm.asarray(phase) - self.value_grid)
        feature_vector = self.A * bm.exp(-0.25 * bm.square(d / self.a))
        return feature_vector

    def check1(self, phase):  # phase 不可能被完美的翻译出来，只有估算
        fea = self.phase_to_feature_vector(phase)
        after_phase = Grid_2D.get_center_tmp(fea)
        print("before phase", phase, "after phase", after_phase)

    def update(self, pos, velocity, r_hpc, hpc2mec_stre=0., train=0, get_loc=1, debug=False, v_noise=0.0001):
        self.get_center()
        v_rot = bm.matmul(self.rot, velocity)
        v_phase = bm.matmul(self.coor_transform, v_rot * self.ratio)
        center_i = self.center_ideal + v_phase * bm.dt
        self.center_ideal.value = self.circle_period(center_i)  # Ideal phase using path integration
        self.centerI = self.Postophase(pos)  # Ideal phase using mapping function

        input_pos = self.get_stimulus_by_pos(pos)
        input_motion = self.get_stimulus_by_motion(velocity, noise_stre=v_noise, debug=debug)
        if get_loc==-1:
            self.input.value = bm.random.normal(0, 0.0001, (self.num,))
        elif get_loc == 0:
            self.input.value = input_motion
        elif get_loc == 1:
            self.input.value = input_pos
        # self.input = bm.where(get_loc == 1, input_pos, input_motion)
        input_hpc = bm.matmul(self.conn_in, r_hpc)
        input_hpc = bm.where(input_hpc > 0, input_hpc, 0)

        self.input.value += input_hpc * hpc2mec_stre
        Irec = bm.matmul(self.conn_mat, self.r)
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share['t'], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        r_learn_hpc = keep_top_n(r_hpc, self.spike_num)
        if debug and train == 0:
            jax.debug.print("check mec center {}, {}, motion C{}, recC{}, hpc C {} mec data {} {} {} {}", self.center, self.centerI,
                            Grid_2D.get_center_tmp(input_motion),
                            Grid_2D.get_center_tmp(Irec),
                            Grid_2D.get_center_tmp(input_hpc),
                            bm.max(self.r),
                            bm.max(input_motion), bm.max(Irec), bm.max(input_hpc))
        # if debug:
        #     jax.debug.print("check mec in {} {} {} {} {}", bm.argmax(self.r), bm.max(self.r),
        #                     bm.max(input_motion), bm.max(Irec), bm.max(input_hpc))
        if train > 0:
            self.conn_out_update(r_learn_hpc=r_learn_hpc)
            self.conn_in_update(r_hpc=r_hpc)


if __name__ == '__main__':
    from env import ConfigParam
    config = ConfigParam()
    MEC = Grid_2D(ratio=5.0, angle=0.0, strength=1.5, config=config)
    for i in range(100):
        phase = bm.random.rand(2)
        # MEC.check1(phase)

    # MEC.reset_state(bm.array([0.9, 0.5]))
    # def initialize_net(i):  # 初始化net完全没必要用motion，要么是loc，要么仅sense
    #     MEC.step_run(i, pos=bm.array([0.9, 0.5]), velocity=bm.zeros(2,), r_hpc=bm.zeros(config.num_hpc,),
    #                  hpc2mec_stre=0., train=0, get_loc=0, debug=True, v_noise=0.00)
    #
    #
    # indices = bm.arange(2000)
    # bm.for_loop(initialize_net, indices, progress_bar=True)