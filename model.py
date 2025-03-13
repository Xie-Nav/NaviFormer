import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# from utils.distributions import Categorical, DiagGaussian
from torch.distributions.categorical import Categorical
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du
from transformer import BertAttention, BertXAttention
from arguments import get_trans_args


class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16):
        # input_shape (24, 240, 240);
        # recurrent = 0; hidden_size = 256; num_sem_categories = 16
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)
        
        # out_size = (240 / 16)*(240 / 16) = 15 * 15
        out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)

        # 这就是论文中提到了5层卷积
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 3 - 1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            # Flatten()
        )

        self.main_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            # Flatten()
            # nn.Conv2d(32, 16, 3, stride=1, padding=1),
            # nn.ReLU()
        )

        self.main_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(1, 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        
        configs = get_trans_args()
        self.encoder_sence = BertAttention(configs)
        self.global_pos_embedding = get_gloabal_pos_embedding(15, 16)


        configs.hidden_size = 20; configs.dim_feedforward = 40
        self.encoder_agent = BertAttention(configs)
        self.global_terrain_embedding = get_gloabal_pos_embedding(15, 10)

        configs.hidden_size = 8
        self.decoder_exp =  BertXAttention(configs, ctx_dim = 32, wights = False)
        self.global_exp_embedding = get_gloabal_pos_embedding(15, 4)

        configs.hidden_size = 32; configs.dim_feedforward = 64
        self.decoder =  BertXAttention(configs, ctx_dim = 20)
        

        self.flatten = Flatten()
        self.linear1 = nn.Linear(out_size * 32, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        # 对朝向的编码
        self.orientation_emb = nn.Embedding(72, 8)
        # 对各个目标物体的编码
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):

        # extras torch.Size([bs, 2]) 里面装的值一个是朝向, 另一个是目标物体的编号
        objects_embedding = self.main(torch.cat((inputs[:,1: 3, :, :], inputs[:,5:, :, :]), 1))
        gpu_id = objects_embedding.get_device()
        objects_embedding = objects_embedding + self.global_pos_embedding.cuda(gpu_id)
        # image_embedding大小变成了[1, 256, 7*7]
        bs, _, c, _ = objects_embedding.shape
        objects_embedding = objects_embedding.reshape(bs, -1, c * c)
        obj_att = objects_embedding.permute(0, 2, 1)
        obj_att , _ = self.encoder_sence(obj_att) # , 1e-3 * wight_exp
        

        exp_embedding = self.main_3(inputs[:, 0:1, :, :])
        gpu_id = exp_embedding.get_device()
        exp_embedding = exp_embedding + self.global_exp_embedding.cuda(gpu_id)
        bs_exp, _, c_exp, _ = exp_embedding.shape
        exp_embedding = exp_embedding.reshape(bs_exp, -1, c_exp * c_exp)
        exp_embedding = exp_embedding.permute(0, 2, 1)
        wight_exp  = self.decoder_exp(exp_embedding, obj_att, wights = True)
        

        terrain_embedding = self.main_2(inputs[:, 3:5, :, :])
        gpu_id = terrain_embedding.get_device()
        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])
        goal_embedding = torch.cat((orientation_emb, goal_emb), 1)
        x_unsqueezed = goal_embedding.unsqueeze(-1).repeat_interleave(15, dim=-1)  # 现在 x_unsqueezed 的形状是 [1, 16, 1]  
        # 最后，再次使用 unsqueeze 和 repeat_interleave 来扩展第三个维度  
        goal_embedding = x_unsqueezed.unsqueeze(-1).repeat_interleave(15, dim=-1)  # 最终 x_final 的形状是 [1, 16, 15, 15]  
        terrain_embedding = torch.cat((terrain_embedding, goal_embedding), 1)
        terrain_embedding = terrain_embedding + self.global_terrain_embedding.cuda(gpu_id)
        bs_ter, _, c_ter, _ = terrain_embedding.shape
        terrain_embedding = terrain_embedding.reshape(bs_ter, -1, c_ter * c_ter)
        terrain_embedding = terrain_embedding.permute(0, 2, 1)
        ter_att, _ = self.encoder_agent(terrain_embedding)

        hs, _ = self.decoder(obj_att, ter_att, wight_exp)
        # hs = hs.permute(1, 2, 0)

        nav_embedding = self.flatten(hs)        

        # x = torch.cat((nav_embedding, orientation_emb, goal_emb), 1)

        x = nn.ReLU()(self.linear1(nav_embedding))
        
        # g_policy.network.is_recurrent = 0
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs

def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 1:
            self.network = Goal_Oriented_Semantic_Policy(
                obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        # probs=None
        # logits torch.Size([bs, 4])
        # validate_args = None
        # masks torch.Size([bs, 4])这是表示那个输出应该被bask掉的指标
        self.masks = masks
        self.device = logits.device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)
        

class RL_Policy_IAM(nn.Module):
    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):
        # obs_shape (24, 240, 240); action_space Discrete(4)
        # model_type = 1; 
        # base_kwargs={'recurrent': args.use_recurrent_global = 0,
                                        #   'hidden_size': g_hidden_size = 256,
                                        #   'num_sem_categories': ngc - 8 = 16
                                        #   }).to(device)
        super(RL_Policy_IAM, self).__init__()
        
        # obs_shape (24, 240, 240);
        # base_kwargs={'recurrent': args.use_recurrent_global = 0,
                                        #   'hidden_size': g_hidden_size = 256,
                                        #   'num_sem_categories': ngc - 8 = 16
                                        #   }).to(device)
        # g_policy.network是网络的主体结构
        self.network = Goal_Oriented_Semantic_Policy(
                obs_shape, **base_kwargs)
        # g_policy.network.output_size = 256 ; action_space.n = 4
        self.linear = nn.Linear(self.network.output_size, action_space.n)

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            # inputs BEV输出 torch.Size([bs, 4 + 2 + 2 + 16, 480, 480])
            # rnn_hxs torch.Size([bs, 1])
            # masks 某一个进程停止训练的标志 torch.Size([bs])
            # extras torch.Size([bs, 2]) 里面装的值一个是朝向, 另一个是目标物体的编号
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False, invalid_action_masks=None):

        # inputs BEV输出 torch.Size([bs, 4 + 2 + 2 + 16, 480, 480])
        # rnn_hxs torch.Size([bs, 1])
        # masks 某一个进程停止训练的标志 torch.Size([bs])
        # extras torch.Size([bs, 2]) 里面装的值一个是朝向, 另一个是目标物体的编号
        # deterministic=False,
        # invalid_action_masks torch.Size([bs, 4])
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)

        # torch.Size([bs, 4])
        actor_features = self.linear(actor_features) # 256 -> 4
        if invalid_action_masks is not None:
            # actor_features torch.Size([bs, 4])
            # invalid_action_masks torch.Size([bs, 4])这是表示那个输出应该被bask掉的指标
            dist = CategoricalMasked(logits = actor_features, masks = invalid_action_masks)
        else:
            dist = Categorical(logits=actor_features)

        action = dist.sample()

        action_log_probs = dist.log_prob(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None, invalid_action_masks=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        actor_features = self.linear(actor_features) # 256 -> 4
        if invalid_action_masks is not None:
            dist = CategoricalMasked(logits = actor_features, masks = invalid_action_masks)
        else:
            dist = Categorical(logits=actor_features)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
        

class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(200 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        # sem_map_module.init_grid torch.Size([bs, 17, 100, 100, 48])
        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)


        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.stair_mask_radius = 30
        self.stair_mask = self.get_mask(self.stair_mask_radius).to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last):
        # obs torch.Size([bs, 3+1+16, 120, 160])
        # pose_obs torch.Size([bs, 3])
        # maps_last torch.Size([bs, 2 + 2 +16, 480, 480])
        # poses_last torch.Size([bs, 3])
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        # Depth图像初步转化为点云坐标
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        # 点云坐标加上相机高度，并且乘以和俯仰角相关的旋转矩阵
        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        # 点云坐标乘以和agent旋转角度相关的矩阵
        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        # sem_map_module.init_grid torch.Size([bs, 17, 100, 100, 48])
        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 50) / z_resolution - min_h)
        mid_z = int(self.agent_height / z_resolution - min_h)

        # 14~35层 高度障碍物
        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        # 21~25层高度表示由楼梯的情况
        agent_height_stair_proj = voxels[..., mid_z-5:mid_z].sum(4)
        # 所有高度的点表示所有的地形信息
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_stair_pred = agent_height_stair_proj[:, 0:1, :, :]
        # sem_map_module.map_pred_threshold = 1
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_stair_pred = fp_stair_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_stair_pred = torch.clamp(fp_stair_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        agent_view_stair = agent_view.clone().detach()
        # 也就第一层agent_view_stair和agent_view不同，agent_view的障碍图包含的障碍物更多一些
        agent_view_stair[:, 0:1, y1:y2, x1:x2] = fp_stair_pred

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # translated[:, 18:19, :, :] = -self.max_pool(-translated[:, 18:19, :, :])

        diff_ob_ex = translated[:, 1:2, :, :] - self.max_pool(translated[:, 0:1, :, :])

        diff_ob_ex[diff_ob_ex>0.8] = 1.0
        diff_ob_ex[diff_ob_ex!=1.0] = 0.0

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)
        # 明确哪些是障碍物信息，哪些是地形信息
        map_pred[:, 0:1, :, :][diff_ob_ex == 1.0] = 0.0

        # stairs view
        rot_mat_stair, trans_mat_stair = get_grid(st_pose, agent_view_stair.size(),self.device)

        rotated_stair = F.grid_sample(agent_view_stair, rot_mat_stair, align_corners=True)
        translated_stair = F.grid_sample(rotated_stair, trans_mat_stair, align_corners=True)

        # stair_mask = torch.zeros(self.map_size_cm // self.resolution, self.map_size_cm // self.resolution).to(self.device)

        # s_y = int(current_poses[0][1]*100/5)
        # s_x = int(current_poses[0][0]*100/5)
        # limit_up = self.map_size_cm // self.resolution - self.stair_mask_radius - 1
        # # limit_be = self.stair_mask_radius
        # if s_y > limit_up:
        #     s_y = limit_up
        # if s_y < self.stair_mask_radius:
        #     s_y = self.stair_mask_radius
        # if s_x > limit_up:
        #     s_x = limit_up
        # if s_x < self.stair_mask_radius:
        #     s_x = self.stair_mask_radius
        # stair_mask[int(s_y-self.stair_mask_radius):int(s_y+self.stair_mask_radius), int(s_x-self.stair_mask_radius):int(s_x+self.stair_mask_radius)] = self.stair_mask

        # translated_stair[0, 0:1, :, :] *= stair_mask
        # translated_stair[0, 1:2, :, :] *= stair_mask


        diff_ob_ex = translated_stair[:, 1:2, :, :] - translated_stair[:, 0:1, :, :]

        diff_ob_ex[diff_ob_ex>0.8] = 1.0
        diff_ob_ex[diff_ob_ex!=1.0] = 0.0

        maps3 = torch.cat((maps_last.unsqueeze(1), translated_stair.unsqueeze(1)), 1)

        map_pred_stair, _ = torch.max(maps3, 1)

        map_pred_stair[:, 0:1, :, :][diff_ob_ex == 1.0] = 0.0

        # translated torch.Size([bs, 20, 480, 480])，这里的translated也就是agent在当前帧看到的场景信息投影到BEV图上
        # map_pred和map_pred_stair torch.Size([bs, 20, 480, 480]) 就是第一层的Obstacle Map有所不同
        # map_pred_stair中第一个进程的Obstacle Map和Exploread Area更新的场景地形可能是一个圆形的场景区域
        # current_poses torch.Size([bs, 3]) 是当前的agent的local位姿,以m为单位  
        return translated, map_pred, map_pred_stair, current_poses


    def get_mask(self, step_size):
        size = int(step_size) * 2 
        mask = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2)) ** 2 + \
                ((j + 0.5) - (size // 2)) ** 2 <= \
                        step_size ** 2:
                    mask[i, j] = 1
        return mask

