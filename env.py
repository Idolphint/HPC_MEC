import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax

class Env:
    def __init__(self):
        self.loc_land = bm.array([
            [0.3, 0.3],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.7, 0.7]
        ])  # shape = n_land * 2 记录每个landmark的坐标
        self.global_land = bm.array([1.2, 1.1])
        self.diameter_min = 0
        self.diameter_max = 1
        self.diameter = self.diameter_max - self.diameter_min
        self.grid_size = 0.1
        # self.grid_color = bm.random.rand(int(self.diameter//self.grid_size + 1)**2, key=42)
        self.env_index = 1

        # 下面是交互式环境的配置
        self.start_pos = np.array([0.01, 0.01])
        self.goal_pos = np.array([0.99, 0.99])
        self.agent_pos = np.array([0.01, 0.01])

    def get_color(self, pos):
        # TODO 当前每个点fea都不一样，且color范围=0，1
        pos_grid = pos / self.grid_size
        grid_num = np.matmul(pos_grid, np.array([np.divide(self.diameter, self.grid_size), 1]))
        if self.env_index == 1:
            pos_color = grid_num * self.grid_size * self.grid_size
        elif self.env_index == 2:
            pos_color = np.sqrt(np.square(pos - np.array([1.0,1.0])).sum(axis=1))
            # grid_num = grid_num.astype(np.int32)
            # pos_color = self.grid_color[grid_num]
        return pos_color

    def get_feature(self, pos):
        # 到100个点的距离
        landm_xs, landm_ys = np.meshgrid(np.linspace(0, 1, 10),
                                         np.linspace(0, 1,10))
        landms = np.array([landm_xs, landm_ys]).reshape(2,100).transpose(1,0)
        if len(pos.shape) == 1:
            dis_l = np.linalg.norm(landms-pos, axis=-1)
        else:
            dis_l = np.linalg.norm(pos[:,np.newaxis,:]-landms[np.newaxis,:], axis=-1)
        # pos_color = self.get_color(pos)
        # # 根据color分成 30个bins，生成高斯分布, 感受野为一整个环境直径
        # bins = np.linspace(0.01, self.diameter-0.01, 30)
        # dis_l = pos_color[:, np.newaxis] - bins[np.newaxis, :]
        gauss_width = 0.07 * (self.diameter_max-self.diameter_min)  # 占总环境直径的7%
        features = np.exp(- (dis_l ** 2) / (2 * gauss_width ** 2))
        return features

    def check_edge_corner(self, pos: np.array):
        return (pos <= self.diameter_min).sum() + (pos>=self.diameter_max).sum()

    def get_train_traj(self, T, v=0.01, dy=0.01, dt=1.):
        # 密集覆盖整个环境
        pos = np.random.rand(2) * (self.diameter_max-self.diameter_min-v) + self.diameter_min + v/2
        # pos = np.array([self.diameter_min+v/2,self.diameter_min+v/2])
        v_now = np.array([0,v])
        dy_now = np.array([dy, 0])
        traj_pos = []
        for i in range(int(T)):
            traj_pos.append(pos.copy())
            if self.check_edge_corner(pos + v_now): # 边
                if self.check_edge_corner(pos + dy_now):  # 角
                    v_now = np.array([v_now[1], v_now[0]])
                    dy_now = np.array([dy_now[1], dy_now[0]])
                    if self.check_edge_corner(pos + dy_now):
                        dy_now = -dy_now
                    if self.check_edge_corner(pos + v_now):
                        v_now = -v_now
                    # print("get corner", pos, dy_now, v_now)
                    pos = pos + v_now
                else:
                    v_now = -v_now
                    # print("get edge", pos, dy_now, v_now)
                    pos = pos + dy_now
            else:
                pos = pos + v_now
        traj_pos = np.stack(traj_pos, axis=0)
        # print(traj_pos)
        total_time = len(traj_pos) * dt
        velocities_matrix = bm.gradient(traj_pos, axis=0)
        traj_fea = self.get_feature(traj_pos)
        # print(traj_fea)
        return traj_pos, traj_fea, velocities_matrix, total_time

    def get_test_traj(self, T, v_max=0.01):
        t = np.arange(0, T, bm.dt)
        # 使用调整后的振幅和频率定义 x(t) 和 y(t)
        x = (self.diameter_max-self.diameter_min)/5*2 * np.sin(v_max / 10 * t) + (self.diameter_max-self.diameter_min)/2
        y = (self.diameter_max-self.diameter_min)/5*2 * np.cos(v_max * 3 / 10 * t) + (self.diameter_max-self.diameter_min)/2

        # 将位置和速度矩阵转换为NumPy数组
        positions_matrix = np.column_stack((x, y))
        loc_fea = self.get_feature(positions_matrix)
        # 重新计算速度向量 v = (vx, vy)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        velocities_matrix = np.column_stack((vx, vy))

        total_time = int(len(x) * bm.dt)
        print("check vel shape", len(x), velocities_matrix.shape, positions_matrix.shape, T)

        return positions_matrix, loc_fea, velocities_matrix, int(total_time)

    def get_line_traj(self, start, end, v_abs=0.01):
        T = int(np.sqrt(np.square(start-end).sum())/v_abs)
        pos_array = np.linspace(start, end, num=T)
        loc_fea = self.get_feature(pos_array)
        velocities_matrix = bm.gradient(pos_array, axis=0)
        return pos_array, loc_fea, velocities_matrix, T

    def step(self, action, v_abs=0.01):
        v_vec = np.array([0,0])
        if action==0:
            if self.agent_pos[1]+v_abs < self.diameter_max:
                v_vec = np.array([0,1])*v_abs
                self.agent_pos += v_vec
        elif action==1:
            if self.agent_pos[1]-v_abs > self.diameter_min:
                v_vec = -np.array([0,1])*v_abs
                self.agent_pos += v_vec
        elif action==2:
            if self.agent_pos[0]+v_abs < self.diameter_max:
                v_vec = np.array([1,0])*v_abs
                self.agent_pos += v_vec
        elif action==3:
            if self.agent_pos[0]-v_abs > self.diameter_min:
                v_vec = -np.array([1,0])*v_abs
                self.agent_pos += v_vec
        else:
            print("error action", action)

        loc = self.agent_pos
        # print("using act", action, "new pos", loc)
        fea = self.get_feature(loc[np.newaxis, :])[0]
        # print("check fea", fea)
        reward = 0
        done = False
        if np.sqrt(np.square(self.agent_pos-self.goal_pos).sum()) < v_abs*1.3:
            reward = 1
            done = True
        return loc, fea, reward, done, v_vec

    def reset(self):
        self.start_pos = np.random.rand(2)
        self.goal_pos = np.random.rand(2)
        while np.abs(self.start_pos - self.goal_pos).sum() < 0.2:
            self.goal_pos = np.random.rand(2)
        self.start_pos = np.around(self.start_pos, 2)
        self.goal_pos = np.around(self.goal_pos, 2)
        self.agent_pos = self.start_pos.copy()
        return self.step(0)

class ConfigParam:
    def __init__(self, env: Env):
        self.env = env
        self.OVC = {  # 描述OVC的关键参数
            "N_dis": 10,  # 每个landamrk有多少个描述距离的细胞，将N_dis平埔到整个环境中
            "N_theta": 10,  # 每个landmark有多少个描述角度的细胞，N_theta平铺360度
            "gauss_width": 0.07,   # 每单位宽度的env需要对应OVC的响应域
        }
        self.num_mec = 10
        self.num_mec_module = 7
        self.mec_max_ratio = 9
        self.num_hpc = 8000
        self.num_HD = 128
        self.num_sen = 100
        self.dt = 1.  # 在轨迹采样函数没改变时，dt只能是1，否则速度，loc关系将会不对应

        # Amount of cells in Hippocampus
        self.spike_num_exc = 100
        self.k_exc = 10. / self.num_hpc
        self.mbar_hpc = 1.

        # 细胞发放高斯形状控制
        self.a_mec = bm.pi / 2
        self.a_land = 0.07 * (env.diameter_max - env.diameter_min)

        # Time constants of neural dynamics
        self.tau = 1.
        self.tau_v = 5.

        # Time constants of learning
        self.tau_W = 500
        self.tau_W_in = 500
        self.tau_W_sen = 100
        self.tau_W_mec = 50
        # Parameters of MEC
        self.tau_e = 1000

        # Parameters of learning
        self.norm_sen_exc = 0.5
        self.norm_fac = 30.
        self.lr_sen_exc = 10.
        self.lr_exc = 2.5 / self.num_hpc
        self.lr_mec_out = 20
        self.lr_mec_in = 20 / self.num_hpc  # self.lr_exc
        self.lr_back = self.lr_exc


def draw_traj(traj, fea):
    fig = plt.figure()
    plt.plot([0,0,1,1], [0,1,0,1], 'r.', markersize=10)
    scatter = plt.scatter([], [], c=[],cmap='viridis', s=5)
    step = 40
    T = traj.shape[0] // step - 2
    def update(frame):
        print(frame*step, (frame+1)*step)
        data = traj[frame*step:(frame+2)*step]
        fea_ = fea[frame*step:(frame+2)*step]
        scatter.set_offsets(data)
        scatter.set_array(fea_)
        return scatter

    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    ani_filename = './traj.gif'
    ani.save(ani_filename, writer='Pillow', fps=30)
    print("save ", ani_filename)

if __name__ == '__main__':
    env = Env()
    env.get_feature(np.array([[0.1,0.2],[0.4,0.2]]))
    # traj, fea = env.get_train_traj(T=4000, v=0.03, dy=0.1)
    # draw_traj(traj, fea)

