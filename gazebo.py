import time
import robot_control_interface_gazebo as gz_interface
import time
import math
import os
import numpy as np

img_h = img_w = 64
channel = 3

class GAZEBO(object):
    def __init__(self, portStr=None):
        if portStr is None:
            os.environ["GAZEBO_MASTER_URI"] = "http://localhost:11343"
        else:
            os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + portStr
        # np.set_printoptions(precision=4, linewidth=100)
        self.interface = gz_interface.robot_control_interface_gazebo("test")
        self.interface.setdebug(1)
        # self.interface.GAZEBO_StartAndLoadWorld("/home/ccchang/gazebo_interface2/robot_control_interface/model/vs060_gazebo.xml",True)
        # self.interface.GAZEBO_InitAllSensors("/home/ccchang/gazebo_interface2/robot_control_interface/model/vs060_gazebo.xml")
        # self.interface.GAZEBO_StartAndLoadWorld("/home/ccchang/robot_control_interface/model/dataset_gen.xml",True)
        # self.interface.GAZEBO_InitAllSensors("/home/ccchang/robot_control_interface/model/dataset_gen.xml")

        print("before start world")
        self.interface.GAZEBO_StartAndLoadWorld("/home/ccchang/robot_control_interface/model/dataset_gen_for_ik.xml",False)
        
        # self.interface.GAZEBO_StartAndLoadWorld("/home/ccchang/robot_control_interface/model/dataset_gen_for_ik.urdf",True)
        # self.interface.GAZEBO_StartAndLoadWorld("/home/ccchang/robot_control_interface/model/example_urdf.xml",True)
        print("after start world")

        # self.interface.GAZEBO_InitAllSensors("/home/ccchang/robot_control_interface/model/dataset_gen_for_ik.xml")

        self.robot_model_name = "vs060"
        self.interface.setdebug(1)
        # self.origin_panel_pq = self.interface.GAZEBO_GetLinkPQByName("panel0", "panel_link")
        self.is_sucked = False
        self.step_limit = 800
        self.step_count = 0
        self.goal_count = 0
        self.collection_count = 0
        self.state_collection_list = []
        self.render_period = 1
        self.joint_num = 6
        # self.pos_range = [[-np.pi, np.pi],[-0.75*np.pi, 2*np.pi],[-0.75*np.pi, 2*np.pi],
        #                  [-0.75*np.pi, 2*np.pi],[-0.75*np.pi, 2*np.pi],[-0.75*np.pi, 2*np.pi]]
        self.pos_range = [[-1.57,1.57],[-0.75,2.1],[-0.75,2.1],[-1.57,1.57],[-1.57,1.57],[-1.57,1.57]]

        self.state_dim = 7 ###TCP pose(x,y,z,quaternion)
        self.act_dim = 6
        self. _max_episode_steps = 800

        #Combine the camera with end effector on robot
        # self.interface.GAZEBO_AddJoint("vs060", "J5", "tips_camera", "tips_camera_link")
        # self.interface.GAZEBO_AddJoint("vs060", "J6", "calibration", "c_link")

        self.interface.GAZEBO_Step(1)
        # self.reset(random_pos=True)

    def reset(self, random_pos=True):
        self.interface.GAZEBO_Reset()
        self.step_count = 0
        self.goal_count = 0
        # self.interface.GAZEBO_RemoveJoint("vs060", "suction")
        # Camera joint will be remove either, temporary solution: munually add joint again
        # self.interface.GAZEBO_AddJoint("vs060", "J5", "camera", "camera_link")
        # self.interface.GAZEBO_AddJoint("vs060", "J6", "calibration", "c_link")

        # panel_pq = self.origin_panel_pq.copy()
        end_pos = [1.57, 0.96, 0.828, 0., 1.36, 0.]  # first step end

        pos = np.zeros([self.joint_num, ])
        if random_pos:
            ###3 joints ver
            # pos = end_pos.copy()
            # pos[1] = np.random.uniform(0.2, 0.7)
            # pos[2] = np.random.uniform(0.2, 0.7)
            # pos[4] = np.random.uniform(1.1,1.8)

            #absolute joint posision
            pos[0] = np.random.uniform(-1.57, 1.57)
            pos[1] = np.random.uniform(-0.75, 2.1)
            pos[2] = np.random.uniform(-0.75, 2.1)
            pos[3] = np.random.uniform(-1.57, 1.57)
            pos[4] = np.random.uniform(-1.57, 1.57)
            pos[5] = np.random.uniform(-1.57, 1.57)

        else:
            pos = [0,0,0,0,0,0]

        # self.tmp += 0.01
        # pos[1] = self.tmp

        pos_dict = make_6joint_dict(pos)
        velocity_dict = make_6joint_dict(np.zeros([self.joint_num, ]))
        torque_dict = make_6joint_dict(np.zeros([self.joint_num, ]))
        self.interface.GAZEBO_SetModel(self.robot_model_name, pos_dict, velocity_dict, torque_dict)
        # self.target_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("panel", "panel_link"))
        ###panel pos after calibration
        self.target_pq = [-0.01, 0.56, 0.335]

        self.initial_tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
        self.interface.GAZEBO_Step(self.render_period)
        return None

    def get_state(self):
        # initial_tcp_pq = self.initial_tcp_pq
        # joint_pq = joint_dict_to_list(self.interface.GAZEBO_GetPosition(self.robot_model_name))
        # tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName(self.robot_model_name, "J6"))

        ### Tmp pos after calibration
        tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))

        # if self.is_sucked:
        #     goal_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("virtual_goal2", "vg2_link"))
        # else:
        #     goal_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("panel", "panel_link"))

        # obs = np.concatenate([joint_pq/2*np.pi, tcp_pq/2*np.pi, np.array(self.is_sucked).reshape(-1)], axis=0)

        # camera_img = self.interface.GAZEBO_GetRawImageByCameraName("camera_sensor")
        # camera_img2 = self.interface.GAZEBO_GetRawImageByCameraName("camera2_sensor")

        #Normalize to 0-1
        # obs_c = ((camera_img["rawimage"]+128)/256).reshape(img_h,img_w,channel)

        return tcp_pq

    def step(self, actions, render=True):   #actions shape:(act_dim,)

        #Act by relative position
        joint_pos = joint_dict_to_list(self.interface.GAZEBO_GetPosition(self.robot_model_name))

        # j_list = [0,1,2,4]
        # for i in range(len(j_list)):
        #     j = j_list[i]
        #     joint_pos[j] = np.clip(joint_pos[j]+actions[i], self.pos_range[j][0], self.pos_range[j][1])

        # print(actions)
        for i in range(len(actions)):
            # joint_pos[i] = np.clip(joint_pos[i]+actions[i], self.pos_range[i][0], self.pos_range[i][1])
            joint_pos[i] = actions[i]

        joint_pos_dict = make_6joint_dict(joint_pos)
        vel_dict = make_6joint_dict(np.zeros([self.joint_num,]))
        torque_dict = make_6joint_dict(np.zeros([self.joint_num,]))
        self.interface.GAZEBO_SetModel(self.robot_model_name, joint_pos_dict, vel_dict, torque_dict)
        self.interface.GAZEBO_Step(self.render_period)

        s_next = self.get_state()
        r, done, is_goal = self.reward_function()
        # print("reward:",r)
        # r, done, is_goal = self.reward_function_absolute_pos()

        if self.step_count > self.step_limit:
            done = True
        self.step_count += 1

        return s_next, r, done, is_goal


    def reward_function_absolute_pos(self):
        end_pos = [1.57, 0.96, 0.828, 0., 1.36, 0.]  # first step end
        end_pos2 = [1.59, -1., -0.575, 0., -1.56, 0.]  # second step end , horizontally move
        r = 0.
        done = is_goal = False
        # tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))

        if self.is_sucked:
            goal_pos = end_pos2
        else:
            goal_pos = end_pos
        robot_joint = joint_dict_to_list(self.interface.GAZEBO_GetPosition("vs060"))

        # print("goal_pos:",goal_pos)
        # print("robot j:", robot_joint)
        # input("stop")

        diff = abs(np.array(goal_pos) - np.array(robot_joint))

        if (diff>=0.8).any():
            r = -0.1
            done = True
            print("out of bound!")
            return r, done, is_goal

        abs_dist = np.sqrt(np.sum(np.square(diff)))
        r -= abs_dist

        # gaussian = 1 / (np.sqrt(2 * np.pi) * 2) * np.e ** (-0.5 * (float(abs_dist) / 2) ** 2)
        # r += 0.01 * gaussian

        collist = self.interface.GAZEBO_GetContactNames(0.00)
        if len(collist) != 0:
            if self.is_sucked:
                for str in collist:
                    if (str[0] != 'panel_collision' or str[1] != "J6_geom") and (str[0] != 'J6_geom' or str[1] != "panel_collision"):
                        print("collist:",collist)
                        done = True
                        break
                    elif (diff < 0.1).all():
                        r += 2
                        print("Goal!!!")
                        done = True
                        is_goal = True
                        break
            else:
                for str in collist:
                    if (str[0] == 'panel_collision' and str[1] == "J6_geom") or (str[0] == 'J6_geom' and str[1] == "panel_collision"):
                        if (diff<0.1).all():
                            r += 1
                            self.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
                            self.is_sucked = True
                            # is_goal = True
                            # done = True
                        else:
                            done = True
                            print("Not accurate enough")
                    else:
                        r -= -0.01
                        done = True
                        break
        return r, done, is_goal

    def reward_function(self):
        r = 0.
        done = is_goal = False
        return r,done,is_goal
        # tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
        ### Tmp pos after calibration
        tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("calibration", "c_link"))

        if self.is_sucked:
            goal_pos = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("virtual_goal2", "vg2_link"))
            diff = [(tcp_pq[i] - goal_pos[i]) for i in range(3)]
            abs_dist = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            r -= 0.1*abs_dist
            # vertical_diff = np.sqrt(np.square(diff[2]))
            # r -= 0.05*vertical_diff
            # r -= 0.01
        else:
            diff = [(tcp_pq[i] - self.target_pq[i]) for i in range(3)]
            abs_dist = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            r -= 0.1*abs_dist

        collist = self.interface.GAZEBO_GetContactNames(0.00)
        if len(collist) != 0:
            if self.is_sucked:
                for str in collist:
                    if (str[0] == 'panel_collision' and str[1] == 'vg2_collision') or (str[1] == 'panel_collision' and str[0] == 'vg2_collision'):
                        r += 2
                        done = True
                        is_goal = True
                        break
                    elif (str[0] != 'panel_collision' or str[1] != "J6_geom") and (str[0] != 'J6_geom' or str[1] != "panel_collision"):
                        print("collist:",collist)
                        done = True
                        break
            else:
                for str in collist:
                    if (str[0] == 'panel_collision' and str[1] == "J6_geom") or (str[0] == 'J6_geom' and str[1] == "panel_collision"):
                        r += 1
                        self.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
                        self.is_sucked = True
                        # is_goal = True
                        # done = True

                        #Check the angle formed by tcp and the panel, (90 degree is the best), way1
                        # q = Quaternion(tcp_pq[3:7])
                        # r_mat = q.rotation_matrix
                        # rotate_coefficient = np.sum([0, 0, 1] * r_mat[2])
                        # print("rotate_coefficient:", rotate_coefficient)
                        # if abs(-1-rotate_coefficient) <= 0.1:
                        #     r += 10.
                        #     self.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
                        #     self.is_sucked = True
                        # else:
                        #     done = True
                        #     break

                        # Check the angle formed by tcp and the panel, (90 degree is the best), way2
                        # factor = self.get_orthogonal_factor()
                        # print("factor:", factor)
                        # print("=========================================================")
                        # if abs(1 - factor) <= 0.02:
                        #     r += 10.

                        # Collect initial states
                        # if SETTING.SAVE_STATE is True and self.is_sucked is False and self.collection_count < 1000000:
                        #     self.collection_count += 1
                        #     joint_pq = joint_dict_to_list(self.interface.GAZEBO_GetPosition("vs060"))
                        #     tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
                        #     panel_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("panel", "panel_link"))
                        #     state_info = np.concatenate([joint_pq, tcp_pq, panel_pq])
                        #     self.state_collection_list.append(state_info)
                        #     if self.collection_count % 50 == 0:
                        #         np.save("collection_tmp.npy", self.state_collection_list)
                        #         states = np.load("collection_tmp.npy")
                        #         print("current collection size:", len(states))

                    else:
                        print("die")
                        r -= 0.1
                        done = True
                        break

        return r, done, is_goal


    def set_pose_by_joint(self, joint_pos_list):
        joint_pos_dict = make_6joint_dict(joint_pos_list)
        vel_dict = make_6joint_dict(np.zeros([self.joint_num,]))
        torque_dict = make_6joint_dict(np.zeros([self.joint_num,]))
        self.interface.GAZEBO_SetModel(self.robot_model_name, joint_pos_dict, vel_dict, torque_dict)
        self.interface.GAZEBO_Step(self.render_period)


def get_orthogonal_factor(self):
    tcp_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
    panel_pq = pq_dict_to_list(self.interface.GAZEBO_GetLinkPQByName("panel", "panel_link"))
    product = tcp_pq[3]*panel_pq[3] + tcp_pq[4]*panel_pq[4] + tcp_pq[5]*panel_pq[5] + tcp_pq[6]*panel_pq[6]
    return 1-np.square(product)

# Dict converter for gazebo API
def make_7dimpos_dict(data):
    """
    Args:
        7 dim numpy.ndarray : position and quaternion [x,y,z,QW,QX,QY,QZ]
    Returns:
        dictionary of position and quaternion : {X:,Y:,Z:,QW:,QX:,QY:,QZ:}
    """
    return {"X": data[0], "Y": data[1], "Z": data[2], "QW": data[3], "QX": data[4], "QY": data[5], "QZ": data[6]}
def make_6joint_dict(data):
    """
    Args:
        n dim numpy.ndarray : euler angle [Joint1,Joint2,...Jointn]
    Returns:
        dictionary of joints info with euler angle: {Joint1:,Joint2:,...Jointn:}
    """
    return {"joint1": data[0], "joint2": data[1], "joint3": data[2], "joint4": data[3], "joint5": data[4],
            "joint6": data[5]}
def pq_dict_to_list(dict):
    """
    Args:
        dictionary of position and quaternion : {X:,Y:,Z:,QW:,QX:,QY:,QZ:}
        notice that the elements are unsorted
    Returns:
        7 dim numpy.ndarray : position and quaternion [x,y,z,QW,QX,QY,QZ]
    """
    ans = np.ndarray(len(dict), dtype=float)
    ans[0] = dict['X']
    ans[1] = dict['Y']
    ans[2] = dict['Z']
    ans[3] = dict['QW']
    ans[4] = dict['QX']
    ans[5] = dict['QY']
    ans[6] = dict['QZ']
    return ans
def joint_dict_to_list(dict):
    """
    Args:
        dictionary of joints info with euler angle: {Joint1:,Joint2:,...Jointn:}
        notice that the elements are unsorted
    Returns:
        n dim numpy.ndarray : euler angle [Joint1,Joint2,...Jointn]
    """
    ans = np.ndarray(6, dtype=float)
    for key, val in dict.items():
        if key == "fixed_joint":  # ignore the fix joint
            continue
        idx = int(key[-1])
        ans[idx - 1] = val
    return ans


def euler_to_quaternion(yaw, pitch, roll):  #yaw(Z), pitch(Y), roll(X)
    # qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    # qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    # qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    # qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    # return [qx, qy, qz, qw]
    return [qw, qx, qy, qz]



def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch = math.asin(t2)
    if abs(t2) >= 1:
        pitch = np.pi/2 if t2>=0 else -1*np.pi/2
    else:
        pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]