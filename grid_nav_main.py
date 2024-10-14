import sys

import rospy
from nav_msgs.msg import Odometry, Path
from nav_msgs.srv import GetPlan,GetPlanRequest,GetPlanResponse
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion,  quaternion_from_euler
import angles
import numpy as np
import torch
from env import EnvZhanTing, ConfigParam
from policy_train_bpk import simply_traj, GridPolicy, CachedGridCode
device = torch.device('cuda:0')


class GridNav:
    # 接受ros信息（坐标，goal），返回goal_plan
    def __init__(self,
                 grid_code_cache:CachedGridCode,
                 policy:GridPolicy,
                 env:EnvZhanTing,
                 config=None
                 ):
        self.global_pose_sub_ = rospy.Subscriber("/lio_odom", Odometry, self.poseCallback, queue_size=10)
        self.get_plan_service = rospy.Service("/get_global_plan", GetPlan, self.getGlobalPlan)
        self.visual_global_plan_pub_ = rospy.Publisher("/vis_global_plan", Path, queue_size=1)

        self.current_pose_ = np.array([0,0,0])
        self.global_goal_pose_ = np.array([0, 0, 0])

        self.policy = policy
        self.grid_code_cache = grid_code_cache
        self.env = env
        self.config = config

        self.max_plan_step = 300
        self.global_trajectory = []
        self.global_orient = []
        self.is_new_global_plan = False
        self.is_plan_timeout = False
        self.is_goal_received_ = False
        self.odom_updated = False

    def poseCallback(self, odom_data):
        orientation = odom_data.pose.pose.orientation
        __, __, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])  # 当前角度
        # 角度转弧度，并完成了lio_odom与雷达pos角度相反数的转换
        yaw = angles.normalize_angle(yaw / 180 * np.pi)
        self.current_pose_ = np.array([odom_data.pose.pose.position.x,
                                       odom_data.pose.pose.position.y,
                                       yaw])
        self.odom_updated = True

    def getPath(self, step_v_abs=0.03):
        rate = rospy.Rate(12)
        while not rospy.is_shutdown():
            rospy.loginfo_once("==============================nav_thread===========================")
            if self.is_goal_received_ and self.odom_updated:
                start_pos = self.current_pose_
                goal_p = self.global_goal_pose_
                test_traj = []
                loc, fea, r, done, _ = self.env.reset_zt(start_pos, goal_p)
                final_grid_code = self.grid_code_cache.get_grid_code(self.env.goal_pos)
                test_traj.append(loc.copy())
                for si in range(self.max_plan_step):
                    if done:
                        print("=============================got plan======================")
                        self.is_goal_received_ = False
                        self.is_new_global_plan = True
                        # self.pubPath(self.global_trajectory, self.global_orient )
                        print("Finished state, we are closing to goal position!")
                        break
                    grid_code_now = self.grid_code_cache.get_grid_code(self.env.agent_pos)
                    state = final_grid_code - grid_code_now
                    state = torch.Tensor(state).to(device)
                    state = state.transpose(1, 0).reshape(1, self.config.num_mec_module,
                                                          self.config.num_mec * self.config.num_mec)
                    action, logp, entropy, value = self.policy.get_actions(state, train=False)
                    act = action.item()
                    loc, fea, reward, done, v_vec = self.env.step(act, v_abs=step_v_abs)
                    # print("step ", i, "loc=", self.env.agent_pos, act)
                    test_traj.append(loc.copy())
                if done:
                    test_traj.append(self.env.goal_pos)
                    traj, angle = simply_traj(np.array(test_traj))
                    # test_traj转真实坐标
                    self.global_trajectory = self.env.idx2pos(traj)
                    angle += start_pos[2]
                    self.global_orient = np.append(angle, goal_p[2])
                else:
                    print("===Time out, brained nav can not plan the path, please select another goal===")
                    # self.is_plan_timeout = True
                    # self.global_trajectory = []
                    # self.global_orient = []
                    # self.is_goal_received_ = False
                    # 备选方案
                    traj, angle = simply_traj(np.stack([self.env.start_pos, self.env.goal_pos], axis=0))
                    self.global_trajectory = self.env.idx2pos(traj)
                    angle += start_pos[2]
                    self.global_orient = np.append(angle, goal_p[2])
                    self.is_goal_received_ = False
                    self.is_new_global_plan = True
            rate.sleep()

    def getGlobalPlan(self, req=GetPlanRequest()):
        print("move base call ====================")
        goal_data = req.goal
        orientation = goal_data.pose.orientation
        (__, __, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.global_goal_pose_ = np.array([goal_data.pose.position.x, goal_data.pose.position.y, yaw])
        self.is_goal_received_ = True
        print('new goal recieved')

        res = GetPlanResponse()
        path_out = Path()
        path_out.header.frame_id = "map"
        rat = rospy.Rate(10)
        count_num = 0

        while True:
            if not self.is_new_global_plan and not self.is_plan_timeout:
                count_num += 1
                rat.sleep()
                if count_num > 30000:
                    print("get global plan timeout")
                    break
            else:
                break
        if self.is_new_global_plan and len(self.global_trajectory) >= 1:
            for index in range(len(self.global_trajectory)):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = self.global_trajectory[index][0]
                pose.pose.position.y = self.global_trajectory[index][1]
                q = quaternion_from_euler(0., 0., self.global_orient[index])
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]

                path_out.poses.append(pose)
            res.plan = path_out
            self.is_new_global_plan = False
            self.visual_global_plan_pub_.publish(path_out)
            print("visual path has been published!")
        elif self.is_plan_timeout:
            self.is_plan_timeout = False

        res.plan = path_out
        # print(path_out)
        # self.save_path_to_csv(res.plan)
        # self.pub_path.publish(res.plan)
        return res

if __name__ == "__main__":
    rospy.init_node('brainMapNav', anonymous=True)

    env = EnvZhanTing()
    config = ConfigParam()
    policy_ckpt = "./policy_ckpt/2024-10-12-13ppo+linear+2Code+loc/best_policy.pt"
    grid_code_cache = CachedGridCode()
    policy = GridPolicy(device, 0.90, config.num_mec_module).to(device)
    if policy_ckpt is not None:
        policy.load_state_dict(torch.load(policy_ckpt))
        print("load policy ckpt from", policy_ckpt)

    nav_vo = GridNav(grid_code_cache, policy, env, config)
    nav_vo.getPath()
    print('starting...')

    rospy.spin()