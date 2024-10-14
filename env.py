import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import maximum_filter

class Env:
    def __init__(self, env_id=1, max_step=500):
        self.loc_land = np.array([
            [0.3, 0.3],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.7, 0.7]
        ])  # shape = n_land * 2 记录每个landmark的坐标
        self.global_land = np.array([1.2, 1.1])
        self.grid_size = 0.1
        # self.grid_color = np.random.rand(int(self.diameter//self.grid_size + 1)**2, key=42)
        self.env_index = env_id
        if self.env_index == 1:
            self.diameter_min = 0
            self.diameter_max = 1
            self.diameter = self.diameter_max - self.diameter_min
        elif self.env_index == 2:
            self.diameter_min = -0.1
            self.diameter_max = 1.05
            self.diameter = self.diameter_max - self.diameter_min
        # 下面是交互式环境的配置
        self.start_pos = np.array([0.01, 0.01])
        self.goal_pos = np.array([0.99, 0.99])
        self.agent_pos = np.array([0.01, 0.01])
        self.step_num = 0
        self.max_step=max_step

    def set_env_index(self, idx):
        self.env_index = idx
        if self.env_index == 1:
            self.diameter_min = 0
            self.diameter_max = 1
            self.diameter = self.diameter_max - self.diameter_min
        elif self.env_index == 2:
            self.diameter_min = -0.1
            self.diameter_max = 1.05
            self.diameter = self.diameter_max - self.diameter_min
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
        total_fea = np.zeros((pos.shape[0], 200))
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
        gauss_width = 0.08  # * (self.diameter_max-self.diameter_min)  # 占总环境直径的7%
        features = np.exp(- (dis_l ** 2) / (2 * gauss_width ** 2))
        return features
        # total_fea[:, (self.env_index-1)*100:self.env_index*100] = features
        # return total_fea

    def check_edge_corner(self, pos: np.array):
        return (pos <= self.diameter_min).sum() + (pos>=self.diameter_max).sum()

    def get_train_traj(self, T, v=0.01, dy=0.01, dt=1.):
        # 密集覆盖整个环境
        pos = np.random.rand(2) * (self.diameter_max-self.diameter_min-v) + self.diameter_min + v/2
        # pos = np.array([0.5,0.5])
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
        velocities_matrix = np.gradient(traj_pos, axis=0)
        traj_fea = self.get_feature(traj_pos)
        # print(traj_fea)
        return traj_pos, traj_fea, velocities_matrix, total_time

    def get_test_traj(self, T, v_max=0.01, dt=0.1):
        t = np.arange(0, T, dt)
        # 使用调整后的振幅和频率定义 x(t) 和 y(t)
        tmp_max = self.diameter_max
        tmp_min = self.diameter_min
        # tmp_max = 1.0
        # tmp_min = 0.0
        x = (tmp_max-tmp_min)/5*2 * np.sin(v_max / 10 * t) + (tmp_max-tmp_min)/2
        y = (tmp_max-tmp_min)/5*2 * np.cos(v_max * 3 / 10 * t) + (tmp_max-tmp_min)/2

        # 将位置和速度矩阵转换为NumPy数组
        positions_matrix = np.column_stack((x, y))
        loc_fea = self.get_feature(positions_matrix)
        # 重新计算速度向量 v = (vx, vy)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        velocities_matrix = np.column_stack((vx, vy))

        total_time = int(len(x) * dt)
        print("check vel shape", len(x), velocities_matrix.shape, positions_matrix.shape, T)

        return positions_matrix, loc_fea, velocities_matrix, int(total_time)

    def get_line_traj(self, start, end, v_abs=0.01):
        T = int(np.sqrt(np.square(start-end).sum())/v_abs)
        pos_array = np.linspace(start, end, num=T)
        loc_fea = self.get_feature(pos_array)
        velocities_matrix = np.gradient(pos_array, axis=0)
        return pos_array, loc_fea, velocities_matrix, T

    def step(self, action, v_abs=0.01):
        v_vec = np.array([0,0])

        if isinstance(action, np.ndarray):
            assert len(action) == 2
            action = np.clip(action, a_min=-1, a_max=1) * v_abs
            self.agent_pos = np.clip(self.agent_pos+action, a_min=self.diameter_min, a_max=self.diameter_max)
        elif action==0:
            v_vec = np.array([0, 1]) * v_abs
        elif action==1:
            v_vec = -np.array([0, 1]) * v_abs
        elif action==2:
            v_vec = np.array([1, 0]) * v_abs
        elif action==3:
            v_vec = -np.array([1, 0]) * v_abs
        elif action == -1:
            v_vec = np.array([0,0])
        else:
            print("error action", action)

        if np.all(self.diameter_min < self.agent_pos+v_vec) and np.all(self.agent_pos+v_vec< self.diameter_max):
            self.agent_pos += v_vec
        else:
            v_vec = np.array([0,0])

        loc = self.agent_pos
        # print("using act", action, "new pos", loc)
        fea = self.get_feature(loc[np.newaxis, :])[0]
        # print("check fea", fea)
        reward = 0
        done = False
        if np.sqrt(np.square(self.agent_pos-self.goal_pos).sum()) < v_abs*1.5:
            reward = 1 - self.step_num/self.max_step/2
            done = True
        # elif self.agent_pos.min() <= 0.0 or self.agent_pos.max()>=1.0:
        #     reward = -0.01
        self.step_num += 1
        return loc, fea, reward, done, v_vec

    def reset(self):
        self.start_pos = np.random.rand(2)
        self.goal_pos = np.random.rand(2)
        while np.abs(self.start_pos - self.goal_pos).sum() < 0.1:
            self.goal_pos = np.random.rand(2)
        self.start_pos = np.around(self.start_pos, 2)
        self.goal_pos = np.around(self.goal_pos, 2)
        self.agent_pos = self.start_pos.copy()
        self.step_num = 0
        return self.step(-1)


class EnvZhanTing(Env):
    # 从costmap读取地图，分割成为格子取得可行动区域，手动抽取landmark
    # map训练时还是用edge_corner获得路径
    # 训练时从各自中随机抽取出发点和目标点，
    def __init__(self, costmap=None, max_step=500):
        super().__init__()
        # 定义真实-sim的转换参数
        self.ori_costmap = costmap
        self.resolution = 0.03  # 1pixel=5cm
        self.shift = np.array([-21.15258178710937, -9.597175598144531])  # 要知道0，0block的左上角具体是什么坐标！
        self.zone_begin_map = np.array([450,0])
        self.map_width = 1039
        self.map_height = 707
        self.reachable_thre = 10

        # 虚拟环境参数
        # self.loc_land = np.array([
        #     # 特殊点
        #     # 普通点
        # ])
        self.diameter_min = np.array([0, 0])
        self.diameter_max = np.array([1.13,1.0])  # 0.01~=50cm
        self.diameter = np.max(self.diameter_max - self.diameter_min)

        # 下面是交互式环境的配置
        self.start_pos = np.array([0.01, 0.01])
        self.goal_pos = self.diameter_max - np.array([0.01, 0.01])
        self.agent_pos = self.start_pos.copy()
        self.step_num = 0
        self.max_step = max_step

    def get_feature(self, pos):
        assert np.all(pos>self.diameter_min) and np.all(pos<self.diameter_max), "invalid pos"
        # 到100个点的距离
        landm_xs, landm_ys = np.meshgrid(np.linspace(0, self.diameter_max[0], 11),
                                         np.linspace(0, self.diameter_max[1],10))
        landms = np.array([landm_xs, landm_ys]).reshape(2,110).transpose(1,0)
        if len(pos.shape) == 1:
            dis_l = np.linalg.norm(landms-pos, axis=-1)
        else:
            dis_l = np.linalg.norm(pos[:,np.newaxis,:]-landms[np.newaxis,:], axis=-1)

        gauss_width = 0.09  # * (self.diameter_max-self.diameter_min)  # 占总环境直径的7%
        features = np.exp(- (dis_l ** 2) / (2 * gauss_width ** 2))
        return features

    def get_test_traj(self, T, v_max=0.01, dt=0.1):
        t = np.arange(0, T, dt)
        # 使用调整后的振幅和频率定义 x(t) 和 y(t)

        x = ((self.diameter_max[0]-self.diameter_min[0])/5*2 * np.sin(v_max / 10 * t) +
             (self.diameter_max[0]-self.diameter_min[0])/2)
        y = ((self.diameter_max[1]-self.diameter_min[1])/5*2 * np.cos(v_max * 3 / 10 * t) +
             (self.diameter_max[1]-self.diameter_min[1])/2)

        # 将位置和速度矩阵转换为NumPy数组
        positions_matrix = np.column_stack((x, y))
        loc_fea = self.get_feature(positions_matrix)
        # 重新计算速度向量 v = (vx, vy)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        velocities_matrix = np.column_stack((vx, vy))

        total_time = int(len(x) * dt)
        print("check vel shape", len(x), velocities_matrix.shape, positions_matrix.shape, T)

        return positions_matrix, loc_fea, velocities_matrix, int(total_time)

    def reset(self):
        self.start_pos = np.random.rand(2) * (self.diameter_max-self.diameter_min-0.02) + self.diameter_min + 0.01
        self.goal_pos = np.random.rand(2) * (self.diameter_max-self.diameter_min-0.02) + self.diameter_min+0.01
        while np.abs(self.start_pos - self.goal_pos).sum() < 0.1:
            self.goal_pos = np.random.rand(2) * (self.diameter_max-self.diameter_min-0.02) + self.diameter_min+0.01
        self.start_pos = np.around(self.start_pos, 2)
        self.goal_pos = np.around(self.goal_pos, 2)
        self.agent_pos = self.start_pos.copy()
        self.step_num = 0
        return self.step(-1)

    def reset_zt(self, curr_pos, goal_pos):
        self.goal_pos = self.pos2idx(goal_pos)
        self.start_pos = self.pos2idx(curr_pos)
        self.start_pos = np.around(self.start_pos, 2)
        self.goal_pos = np.around(self.goal_pos, 2)
        self.agent_pos = self.start_pos.copy()
        self.step_num = 0
        return self.step(-1)

    @staticmethod
    def simply_costmap(pixel_map: np.ndarray):
        # 假设costmap=-1代表不可达区域，=0代表边界障碍，>0代表可通行
        pool_map = -maximum_filter(-pixel_map, size=10)
        print("check max pool shape", pool_map.shape)
        movable_area = np.where(pool_map > 10)
        print("check moveable_area", movable_area)
        x_min, x_max, y_min, y_max = np.min(movable_area[:,0]), np.max(movable_area[:,0]), \
        np.min(movable_area[:,1]), np.max(movable_area[:,1])
        print(x_min, x_max, y_min, y_max)
        return pool_map, x_min, x_max, y_min, y_max

    def idx2pos(self, idx: np.ndarray):
        map_idx = idx * 500. + self.zone_begin_map
        pos = map_idx * self.resolution + self.shift
        # pos0 = (self.map_width - map_idx[..., 1]) * self.resolution + self.shift[0]
        # pos1 = map_idx[..., 0] * self.resolution + self.shift[1]
        # # pos = idx*self.resolution + self.shift
        # pos = np.stack([pos0, pos1], axis=-1)
        print("check idx2 position!", pos)
        return pos

    def pos2idx(self, position: np.ndarray):
        # pos坐标对应的map是否有障碍物
        # center (-21.15258178710937, -9.597175598144531, 0.0) w-h 1039 707
        map_idx = np.round((position[:2]-self.shift) / self.resolution, decimals=0)
        # x_index = int(self.map_width - round((position[0] - self.shift[0]) / self.resolution))
        # y_index = int(round((position[1] - self.shift[1]) / self.resolution))
        # map_idx = np.array([y_index, x_index])

        norm_pos = (map_idx-self.zone_begin_map) / 500.0
        print("check postion 2 idx", position, map_idx, norm_pos)
        return norm_pos

    # def pos2idx(self, pos: np.ndarray):
    #     idx = (pos-self.shift)/self.blockW
    #     return idx

class ConfigParam:
    def __init__(self):
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
        self.num_sen = 110
        self.dt = 1.  # 在轨迹采样函数没改变时，dt只能是1，否则速度，loc关系将会不对应

        # Amount of cells in Hippocampus
        self.spike_num_exc = 100
        self.k_exc = 10. / self.num_hpc
        self.mbar_hpc = 1.

        # 细胞发放高斯形状控制
        self.a_mec = np.pi / 2

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
        self.norm_sen_exc = 1.5  # 原本0.5太小了
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


def cost_modify():
    #center (-21.15258178710937, -9.597175598144531, 0.0) w-h 1039 707
    # 但是odom返回的充电点坐标是-1.41， 2.29
    map_ = np.load("./tmp/ori_costmap.npy").astype(np.float32)
    map_[(map_>0)&(map_<5)] = 0
    print(map_.shape, type(map_), np.any(np.isnan(map_)))
    pool_map = cv2.resize(map_, dsize=None, fx=0.2, fy=0.2)
    # pool_map = -maximum_filter(-map, size=10)
    print("check max pool shape", pool_map.shape)
    show_map = pool_map.copy()
    # show_map[show_map > 0] = 100
    show_map[show_map == 0] = 255
    show_map[show_map==-1] = 180

    plt.imshow(show_map)
    plt.show()
    # movable_area = np.where(pool_map > 10)
    # print("check moveable_area", movable_area)
    # x_min, x_max, y_min, y_max = np.min(movable_area[:, 0]), np.max(movable_area[:, 0]), \
    #     np.min(movable_area[:, 1]), np.max(movable_area[:, 1])
    # print(x_min, x_max, y_min, y_max)

if __name__ == '__main__':
    # env = Env()
    # env.get_feature(np.array([[0.1,0.2],[0.4,0.2]]))
    cost_modify()
    # traj, fea = env.get_train_traj(T=4000, v=0.03, dy=0.1)
    # draw_traj(traj, fea)

