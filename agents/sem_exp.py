import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
import imageio
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from envs.utils.fmm_planner import FMMPlanner
# from envs.habitat.objectgoal_env import ObjectGoal_Env
from envs.habitat.objectgoal_env21 import ObjectGoal_Env21
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu

from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping
import torch

class UnTrapHelper:
    def __init__(self):
        # env.untrap.total_id表示在某个地点的撞击次数大于3次的情况的总次数
        # env.epi_id 表示在某个地点的撞击次数大于3次，env.untrap.total_id就加1
        self.total_id = 0
        self.epi_id = 0

    def reset(self):
        # 这两个参数暂时还不知道是干嘛的,后面再看
        self.total_id += 1
        self.epi_id = 0

    def get_action(self):
        self.epi_id += 1
        if self.epi_id == 1:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3 #3
        else:
            if self.total_id % 2 == 0:
                return 3 #3
            else:
                return 2

class Sem_Exp_Env_Agent(ObjectGoal_Env21):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=InterpolationMode.NEAREST)])

        # # initialize semantic segmentation prediction model
        # if args.sem_gpu_id == -1:
        #     args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.device = torch.device('cuda:'+ str(args.sem_gpu_id))  
        # self.device = args.device
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.red_sem_pred = load_rednet(
            self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
        )
        self.red_sem_pred.eval()
        # self.red_sem_pred.to(self.device)


        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        self.replan_count = 0
        self.collision_n = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.visualize_img_list = []
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器  
        
        self.fail_case = {}
        self.fail_case['collision'] = 0
        self.fail_case['success'] = 0
        self.fail_case['detection'] = 0
        self.fail_case['exploration'] = 0
        self.fail_type = ['exploration', 'collision', 'detection', 'success']

        self.action_type = ['Done', 'Forward', 'Left', 'Right']

        # env.block_threshold这是开启env.untrap的阈值
        self.block_threshold = 3
        self.untrap = UnTrapHelper()

        # if args.visualize or args.print_images:
        #     # env.legend = none
        #     self.legend = cv2.imread('docs/output2.png')
        #     self.vis_image = None
        #     self.rgb_vis = None
        #     self.sem_vis = np.zeros((480, 640)) 


    def reset(self):
        args = self.args

        self.replan_count = 0
        self.collision_n = 0

        obs, info = super().reset()
        obs = self._preprocess_obs(obs)

        self.obs_shape = obs.shape

        # Episode initializations
        # args.map_size_cm = 4800; args.map_resolution = 5
        # map_shape (960, 960) 一个像素点代表了5cm * 5cm的实际场景大小
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        # env.curr_loc以m为单位的真实坐标
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        
        self.prev_blocked = 0
        self.block_threshold = 3
        # if args.visualize or args.print_images:
        #     # env.legend = none；env.goal_name 当前目标对应的字符
        #     # env.vis_image就是我们的可视化图片， 包含了两个标题，两个框，一个图例，但是图例暂时没有
        #     self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        self.untrap = UnTrapHelper()
        self._previous_action = -1
        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), self.fail_case, False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0
            # self.collision_map = np.zeros(self.visited.shape)
            self.info['clear_flag'] = 0

        action = self._plan(planner_inputs)

        if self.collision_n > 20 or self.replan_count > 20:
            self.info['clear_flag'] = 1
            self.collision_n = 0

        # if self.args.visualize or self.args.print_images:
        #     self._visualize(planner_inputs, self.action_type[int(action)])

        if action >= 0:
            # act
            action = {'action': action}
            obs, rew, done, info = super().step(action)
            
            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = action['action']
            self.obs = obs
            self.info = info

            info['g_reward'] += rew

            return obs, self.fail_case, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), 0., False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1 and not planner_inputs["new_goal"]:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                
                self.prev_blocked += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
            else:
                if self.prev_blocked >= self.block_threshold:
                    self.untrap.reset()
                self.prev_blocked = 0

        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        if replan:
            self.replan_count += 1
            if self.args.eval:
                print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        # Deterministic Local Policy
        if (stop and planner_inputs['found_goal'] == 1) :
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward
        
        if self.args.deactivate_traphelper == False:
            # 如果是前进撞击了多次(这里给的是3次),就考虑需要转向了
            if self.prev_blocked >= self.block_threshold:
                if self._previous_action == 1:
                    action = self.untrap.get_action()
                    # print("trap helper begins!")
                else:
                    action = 1
        # env._previous_action是上一步的动作
        self._previous_action = action

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        # semantic = obs[:,:,4:5].squeeze()
        # print("obs: ", semantic.shape)
        if args.use_gtsem:
            self.rgb_vis = rgb
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            # for i in range(16):
            #     sem_seg_pred[:,:,i][semantic == i+1] = 1
        else: 
            red_semantic_pred, semantic_pred = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15))   
            for i in range(0, 15):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping[i]] = 1
                if args.print_images:
                    self.sem_vis[red_semantic_pred == mp_categories_mapping[i]] = i + 1
                    # print('nima:', np.where(self.sem_vis>0))

            if args.print_images:
                self.sem_vis[(semantic_pred[:,:,0] == 0) & (self.sem_vis == 0 + 1)] = 0
                self.sem_vis[(semantic_pred[:,:,1] == 0) & (self.sem_vis == 1 + 1)] = 0
                self.sem_vis[(semantic_pred[:,:,3] == 0) & (self.sem_vis == 3 + 1)] = 0
                self.sem_vis[semantic_pred[:,:,4] == 1] = 4 + 1
                self.sem_vis[semantic_pred[:,:,5] == 1] = 5 + 1         
            # print('cishi:', np.where(self.sem_vis>0))
            sem_seg_pred[:,:,0][semantic_pred[:,:,0] == 0] = 0
            sem_seg_pred[:,:,1][semantic_pred[:,:,1] == 0] = 0
            sem_seg_pred[:,:,3][semantic_pred[:,:,3] == 0] = 0
            sem_seg_pred[:,:,4][semantic_pred[:,:,4] == 1] = 1
            sem_seg_pred[:,:,5][semantic_pred[:,:,5] == 1] = 1
            # print('sem:', np.where(sem_seg_pred>0))


        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(image, depth).squeeze().cpu().detach().numpy()
            
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
            # # 这是每一种物体的得分
            # semantic_scores = semantic_scores.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        
        return red_semantic_pred,  semantic_pred, 
    
                
