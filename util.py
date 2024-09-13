import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def normalize_rows(matrix):
    column_sums = bm.sum(matrix, axis=1)
    normalized_matrix = matrix / column_sums[:, np.newaxis]
    return normalized_matrix

def keep_top_n(mat, n): #keep top n elements values while leave all others zero
    n = int(n)
    mat = mat.flatten()
    sorted_indices = bm.argsort(mat)[::-1]  # 对展平后的数组进行排序，并获取排序后的索引
    top_n_indices = sorted_indices[:n]  # 获取前n大元素的索引
    result = bm.zeros_like(mat)  # 创建一个与展平后数组大小相同的全零数组
    result[top_n_indices] = mat[top_n_indices]
    return result

def place_cell_select_fr(HPC_fr, thres=0.002):
    HPC_fr = bm.as_numpy(HPC_fr)
    max_fr = np.max(HPC_fr, axis=0)
    place_index = np.argwhere(max_fr > thres)
    place_num = place_index.shape[0]
    return max_fr, place_index, place_num

def place_center(HPC_fr, place_index, loc):
    place_num = int(place_index.shape[0])
    Center = np.zeros([place_num, 2])
    fr_probe = HPC_fr[:,place_index.reshape(-1,)]
    for i in range(place_num):
        max_time = np.argmax(fr_probe[:,i],axis=0)
        Center[i,:] = loc[max_time,:]
    return Center

def get_center_hpc(fr, center_x, center_y, thres=0.2, method="max"):
    # 平均法计算中心
    # Comparing each element with 1/10th of the row's max value
    if method == "avg":
        max_values = fr.max(axis=1).reshape(-1, 1)
        fr = np.where(fr < max_values*thres, 0, fr)
        sum_fr = np.sum(fr, axis=1)
        sum_fr = sum_fr.reshape(-1,1)
        print("check sum fr", sum_fr)
        Cx = np.matmul(fr, center_x.reshape([-1, 1]))/sum_fr
        Cy = np.matmul(fr, center_y.reshape([-1, 1]))/sum_fr
    elif method == "max":
        # 最大值法计算中心
        max_index = np.argmax(fr, axis=1)
        T = fr.shape[0]
        Cx = np.zeros(T,)
        Cy = np.zeros(T,)
        for i in range(T):
            Cx[i] = center_x[max_index[i]]
            Cy[i] = center_y[max_index[i]]
    else:
        print("center method", method, "not in valid list")
    return Cx, Cy


def init_model(Coupled_Model, loc0, loc_fea0):
    loc0 = bm.array(loc0)
    loc_fea0 = bm.array(loc_fea0)
    Coupled_Model.mec2hpc_stre = 0  # =0可以保证hpc能够耦合上多个地图
    Coupled_Model.sen2hpc_stre = 1
    Coupled_Model.hpc2mec_stre = 1

    Coupled_Model.reset_state(loc=loc0)

    def initialize_net(i):  # 20 x size
        Coupled_Model.step_run(i, velocity=bm.zeros(2, ), loc=loc0, loc_fea=loc_fea0,
                               get_loc=1, get_view=0, train=0)

    # Coupled_Model.sen2hpc_stre = 10.
    indices = np.arange(500)
    bm.for_loop(initialize_net, (indices), progress_bar=True)
    Coupled_Model.mec2hpc_stre = 1  # =0可以保证hpc能够耦合上多个地图
    Coupled_Model.sen2hpc_stre = 1
    Coupled_Model.hpc2mec_stre = 1


def plot_place_data(hpc_u, hpc_fr, I_mec, I_sen, loc, env, step, dir, thres=3.5):
    max_u, place_index, place_num = place_cell_select_fr(hpc_u, thres=thres)
    max_r = np.max(hpc_fr, axis=0)
    print(step + ':place cell number = {}'.format(place_num))
    place_score = max_r[place_index.reshape(-1, )]
    plt.figure()
    Center = place_center(hpc_fr, place_index, loc)
    # print("check center", Center)
    sc = plt.scatter(Center[:, 0], Center[:, 1], c=place_score)
    cbar = plt.colorbar(sc, label='Place Cell Response')
    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10, label='Landmarks')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.axis('equal')
    plt.title("place field distribution")
    plt.legend(loc='lower right')
    figname = dir + step + 'place_field_distribution.png'
    plt.savefig(figname)

    place_cell_indices = place_index.reshape(-1, )
    Center_mec = place_center(I_mec, place_index, loc)
    Center_sen = place_center(I_sen, place_index, loc)

    mat_dict = {'Max fr': max_r, 'place_index': place_cell_indices, 'Center': Center, 'Center_mec': Center_mec,
                'Center_sen': Center_sen}
    filename = dir + step + 'place_information.npy'
    np.save(filename, mat_dict)
    print("finish!")
    return filename


def draw_population_activity(directory, place_info_path, u_mec, hpc_fr, I_mec, I_sen, loc, env, step):
    data = np.load(place_info_path, allow_pickle=True).item()
    place_index = data['place_index']  # 可能是place index顺序有问题？
    place_cell_coordinates = data['Center']

    # Sort place cell index 获得了在发放的place cell的fr，mec的输入和sen的输入
    place_fr = hpc_fr[:, place_index.reshape(-1, )]
    place_mec = I_mec[:, place_index.reshape(-1, )]
    place_sen = I_sen[:, place_index.reshape(-1, )]

    # Decode population activities of place cells
    decoded_x, decoded_y = get_center_hpc(place_fr, place_cell_coordinates[:, 0], place_cell_coordinates[:, 1])
    decoded_x_mec, decoded_y_mec = get_center_hpc(place_mec, place_cell_coordinates[:, 0], place_cell_coordinates[:, 1],
                                                  thres=0.4)  # Grid cell input
    decoded_x_sen, decoded_y_sen = get_center_hpc(place_sen, place_cell_coordinates[:, 0],
                                                  place_cell_coordinates[:, 1])  # LV cell input

    # Plot trajectory  画出来place_fr的轨迹中心，place_mec的轨迹中心，place_sen的轨迹中心
    x = loc[:, 0]
    y = loc[:, 1]
    print("check traj shape", x.shape, decoded_x.shape, decoded_x_mec.shape)
    fig = plt.figure()
    plt.plot(x, y, label="Groudtruth Trajectory")
    plt.plot(decoded_x[10:-10], decoded_y[10:-10], label="Decoded Trajectory HPC")
    plt.plot(decoded_x_mec[10:-10], decoded_y_mec[10:-10], label="Decoded Trajectory MEC")
    plt.plot(decoded_x_sen[10:-10], decoded_y_sen[10:-10], label="Decoded Trajectory SEN")

    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10)
    plt.xlim(0, 1)  # -2.5, 2.5)
    plt.ylim(0, 1)  # -2.5, 2.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Testing result")
    plt.legend()
    plt.grid(True)
    figname = directory + step + 'Place_decoding.png'
    plt.savefig(figname)

    # Generate animation of population activities of place cells
    n_step = 8
    data1 = place_fr[::n_step, :]
    print("check the shape of data to gen gif", data1.shape, place_fr.shape, data1)
    T = data1.shape[0]
    # 创建画布和轴
    fig = plt.figure()
    scatter = plt.scatter([], [], c=[], cmap='viridis', s=25)
    plt.plot(x, y)
    plt.plot(decoded_x[10:-10], decoded_y[10:-10])
    plt.plot(decoded_x_mec[10:-10], decoded_y_mec[10:-10])
    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10)
    plt.xlim(0, 1)  # -2.5, 2.5)
    plt.ylim(0, 1)  # -2.5, 2.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # 更新线条的函数
    def update(frame):
        if frame % 20 == 0:
            print("drawing frame", frame)
        data = data1[frame].flatten()
        scatter.set_offsets(place_cell_coordinates)
        # scatter.set_offsets(np.column_stack((place_cell_coordinates[:, 0], place_cell_coordinates[:, 1])))
        scatter.set_array(data)  # 每一帧这些place cell的发放是多大
        return scatter

    # print("begin saving gif")
    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    #
    # # 保存动画为gif文件
    ani_filename = directory +step+ 'test_Population_activities.gif'
    ani.save(ani_filename, writer='Pillow', fps=30)

def draw_grid_activity(mec_r: bm.Array, directory, step):
    mec_r = mec_r.transpose([0,2,1]).reshape(-1, 7,10,10)
    print(mec_r.shape)
    n_step = 8
    data1 = mec_r[::n_step, ...]
    print("check the shape of data to gen gif", data1.shape)
    T = data1.shape[0]
    # mec_r = mec_r.reshape(T,n,10,10)
    x_grid = np.linspace(0,10,10)
    y_grid = np.linspace(0,10,10)
    x_mesh, y_mesh = bm.meshgrid(x_grid, y_grid)
    print(x_mesh, x_mesh.shape)
    fig, axs = plt.subplots(2, 2, figsize=(2 * 2, 2*2))
    scatters = []
    for i in range(4):
        scatter = axs[i//2][i%2].scatter(x_mesh, y_mesh, c=[], cmap='viridis', s=25)
        axs[i//2][i%2].set_title(f'Activity for n={i}')
        scatters.append(scatter)

    def update(frame):
        if frame % 20 == 0:
            print("drawing frame", frame)
        for i in range(4):
            data = data1[frame,i].flatten()
            scatters[i].set_array(data)
        return scatters

    # print("begin saving gif")
    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    #
    # # 保存动画为gif文件
    ani_filename = directory +step+ 'mec_activities.gif'
    ani.save(ani_filename, writer='Pillow', fps=30)


def draw_nav_traj(traj, traj_grid_code, prefix=""):
    x_grid = np.linspace(0, 1, 10)
    y_grid = np.linspace(0, 1, 10)
    x_mesh, y_mesh = bm.meshgrid(x_grid, y_grid)
    fig, axs = plt.subplots(2, 2, figsize=(4 * 2, 4 * 2))
    scatters = []

    axs[0][0].plot(traj[:, 0], traj[:, 1])
    axs[0][0].set_xlim(0, 1)
    axs[0][0].set_ylim(0, 1)
    axs[0][0].set_title("agent traj")
    sct0 = axs[0][0].scatter([],[],c='green',s=25)
    scatters.append(sct0)
    for i in range(1, 4):
        scatter = axs[i // 2][i % 2].scatter(x_mesh, y_mesh, c=[], cmap='viridis', s=40)
        axs[i // 2][i % 2].set_title(f'Activity for n={i}')
        scatters.append(scatter)

    n_step = 1
    traj = traj[::n_step]
    traj_grid_code = traj_grid_code[::n_step]
    print("check shape", traj.shape, traj_grid_code.shape)
    T = traj.shape[0]
    def update(frame):
        if frame % 20 == 0:
            print("drawing frame", frame)
        scatters[0].set_offsets(traj[frame])
        for i in range(1, 4):
            data = traj_grid_code[frame, (i-1)*2].flatten()
            # print(data[:5])
            scatters[i].set_array(data)
        return scatters

    # print("begin saving gif")
    ani = FuncAnimation(fig, update, frames=T, interval=300, blit=True)
    #
    # # 保存动画为gif文件
    ani_filename = f'./policy_ckpt/test_traj_with_grid_code_{prefix}.gif'
    ani.save(ani_filename, writer='Pillow', fps=2)

if __name__ == '__main__':
    # x = bm.random.normal(0,0.2,(100,100,7))
    # draw_grid_activity(x, "x", "x")
    data = np.load("./tmp/saved_traj11.npy",allow_pickle=True).item()
    part_traj = data["traj"]
    part_code = data["code"]
    draw_nav_traj(part_traj, part_code, '11')