# Copyright (c) 2025 Bo Liu
# Licensed under the MIT License

import torch
import torch.nn as nn
import pdb
from einops import rearrange
import torch.nn.functional as F
from diff_gaussian_rasterization_ch128 import GaussianRasterizer, GaussianRasterizationSettings
import torchvision.utils as vutils
from gsplat.rendering import rasterization
import gsplat
def get_orthographic_projection_matrix(left, right, bottom, top, near, far, device):
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = -2.0 / (far - near)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(far + near) / (far - near)
    P[3, 3] = 1.0
    return P

def create_rotation_translation_from_lookat(camera_pos_np, look_at_target_np, up_vec_np):
    forward_z = look_at_target_np - camera_pos_np
    forward_z /= np.linalg.norm(forward_z)
    right_x = np.cross(forward_z, up_vec_np)
    right_x /= np.linalg.norm(right_x)
    up_y = np.cross(right_x, forward_z)
    up_y /= np.linalg.norm(up_y)
    R_wc_np = np.array([right_x, up_y, -forward_z])
    T_wc_np = -R_wc_np @ camera_pos_np
    return R_wc_np, T_wc_np

def look_at(eye, center, up):
    z = torch.nn.functional.normalize(center - eye, p=2, dim=-1)  # look direction
    x = torch.nn.functional.normalize(torch.cross(up, z), p=2, dim=-1)
    y = torch.cross(z, x)
    R = torch.stack([x, y, z], dim=0)  # [3,3]
    t = -R @ eye
    RT = torch.eye(4, device=eye.device)
    RT[:3, :3] = R
    RT[:3, 3] = t
    return RT

# --- 终极修正版函数：自动计算紧凑视锥体 ---
def get_ortho_camera_params(plane_type, grid_size, scene_range_min_np, scene_range_max_np, data_device="cuda"):
    """
    返回正交相机的外参和内参矩阵（extrinsic & intrinsic）
    """
    # 1. 定义场景包围盒的8个顶点
    min_c = scene_range_min_np
    max_c = scene_range_max_np
    corners = np.array([
        [min_c[0], min_c[1], min_c[2]],
        [max_c[0], min_c[1], min_c[2]],
        [min_c[0], max_c[1], min_c[2]],
        [min_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], min_c[2]],
        [min_c[0], max_c[1], max_c[2]],
        [max_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], max_c[2]],
    ])
    corners = torch.from_numpy(corners).float().to(data_device)
    corners_hom = torch.cat([corners, torch.ones(8, 1, device=data_device)], dim=1)

    # 2. 设置相机位置
    scene_center_np = (scene_range_min_np + scene_range_max_np) / 2.0
    max_dim = torch.max(scene_range_max_np - scene_range_min_np)
    camera_distance_from_center = max_dim * 2.0 
    
    if plane_type == 'xy':
        campos_np = scene_center_np + torch.tensor([0.0, 0.0, camera_distance_from_center])
        up_vec_np = torch.tensor([0.0, 1.0, 0.0])
    elif plane_type == 'xz':
        campos_np = scene_center_np + torch.tensor([0.0, camera_distance_from_center, 0.0])
        up_vec_np = torch.tensor([0.0, 0.0, 1.0])
    elif plane_type == 'yz':
        campos_np = scene_center_np + torch.tensor([camera_distance_from_center, 0.0, 0.0])
        up_vec_np = torch.tensor([0.0, 1.0, 0.0])
    else:
        raise ValueError("Invalid plane_type.")

    # 3. 计算视图矩阵 V = [R | t]

    view_matrix = look_at(campos_np, scene_center_np, up_vec_np)

    # 构造一个简单的透视相机
    K = torch.eye(3, device=data_device)
    K[0, 0] = K[1, 1] = K[0, 2] = K[1, 2] = grid_size / 2
    view_matrix = view_matrix.to(data_device)
    # 8. 返回结果
    return {
        "intrinsic_matrix": K,      # 3x3: 相机空间 xy -> 像素坐标
        "extrinsic_matrix": view_matrix,      # 3x4: 世界到相机
        "view_matrix": view_matrix,                # 4x4 视图矩阵
        "camera_center": torch.tensor(campos_np, dtype=torch.float32, device=data_device),
        "image_height": grid_size,
        "image_width": grid_size,
    }


timers = {}
def start_timer(name):
    
    timers[name] = {
        'start': torch.cuda.Event(enable_timing=True),
        'end': torch.cuda.Event(enable_timing=True)
    }
    timers[name]['start'].record()
        
def end_timer(name):
    timers[name]['end'].record()
    torch.cuda.synchronize()
    elapsed_time = timers[name]['start'].elapsed_time(timers[name]['end'])
    print(f"{name} 耗时: {elapsed_time:.2f} ms")


def project_depth_to_3d(depth, c2w_cond, K):
    """
    depth: [B, H, W]
    c2w_cond: [B, 4, 4] (相机到世界矩阵)
    K: [B, 3, 3] (内参矩阵)
    返回: [B, H, W, 3] (世界坐标系下的3D点), [B, H, W] (有效深度mask)
    """
    B, H, W = depth.shape
    device = depth.device

    depth_mask = depth != 0.0
    depth = depth.clone()
    depth[depth_mask] += 0.01

    # 生成像素网格
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    uv_homogeneous = torch.stack([u.float(), v.float(), torch.ones_like(u).float()], dim=-1)  # [H, W, 3]
    uv_homogeneous = uv_homogeneous.unsqueeze(0).expand(B, H, W, 3)  # [B, H, W, 3]

    # 计算相机内参矩阵的逆
    K_inv = torch.inverse(K)  # [B, 3, 3]

    # 将像素坐标转换到相机坐标系下的方向向量
    uv_homogeneous_flat = uv_homogeneous.reshape(B, -1, 3)  # [B, H*W, 3]
    cam_coords_flat = torch.bmm(uv_homogeneous_flat, K_inv.transpose(1, 2))  # [B, H*W, 3]

    # 乘以深度得到相机坐标系下的 3D 点
    depth_flat = depth.reshape(B, -1, 1)  # [B, H*W, 1]
    points_cam_flat = cam_coords_flat * depth_flat  # [B, H*W, 3]

    # 齐次坐标
    ones = torch.ones((B, H*W, 1), device=device, dtype=points_cam_flat.dtype)
    points_cam_homogeneous_flat = torch.cat([points_cam_flat, ones], dim=-1)  # [B, H*W, 4]

    # 相机到世界
    points_world_homogeneous_flat = torch.bmm(points_cam_homogeneous_flat, c2w_cond.transpose(1, 2))  # [B, H*W, 4]

    # 取前三个分量
    points_world_flat = points_world_homogeneous_flat[:, :, :3]  # [B, H*W, 3]

    # reshape 回原始图像尺寸
    world_points = points_world_flat.reshape(B, H, W, 3)  # [B, H, W, 3]
    depth_mask = depth_mask  # [B, H, W]

    return world_points, depth_mask

import torch

# ... 其他代码 ...
def norm_points_bounds(world_points: torch.Tensor, scene_bounds: list = [-1, 1, -1, 1,-1, 1]):
    normed_x = 2 * (world_points[..., 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[..., 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[..., 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(3), normed_y.unsqueeze(3), normed_z.unsqueeze(3)], dim=-1)
    
def get_plane_coords(world_points: torch.Tensor, mask, plane: str):
    """
    将3D世界坐标投影到指定平面坐标
    Args:
        world_points: [H, W, 3] 或 [N, 3] 的3D坐标
        plane: 'xy' | 'xz' | 'yz'
        scene_bounds: 场景坐标范围，用于归一化
    Returns:
        plane_coords: [..., 2] 对应平面的2D坐标 (归一化到[-1,1])
    """

    
    if plane == 'xy':
        coords = world_points[mask][..., [0, 1]]  # 取XY坐标
    elif plane == 'xz':
        coords = world_points[mask][..., [0, 2]]  # 取XZ坐标
    elif plane == 'yz':
        coords = world_points[mask][..., [1, 2]]  # 取YZ坐标
    else:
        raise ValueError(f"Invalid plane type: {plane}")

    return coords

class StandardCrossViewAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        # 使用标准的 MultiheadAttention (batch_first=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # 输入输出为 [B, L, C]
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)  # 新增

    def forward(self, x, context):
        """
        x: [B, H*W, C] (query)
        context: [B, H*W, V, C] (key/value)
        """
        B, L, V, C = context.shape
        
        # 1. 归一化
        x_norm = self.norm1(x)
        context_norm = self.norm1(context)
        
        # 2. 重塑为 [B*L, ...] 以进行批量处理
        x_flat = x_norm.view(B * L, C)  # [B*L, C]
        context_flat = context_norm.view(B * L, V, C)  # [B*L, V, C]
        
        # 3. 添加 query 维度
        x_flat = x_flat.unsqueeze(1)  # [B*L, 1, C]
        
        # 4. 批量注意力计算
        attn_out, _ = self.attn(x_flat, context_flat, context_flat)  # [B*L, 1, C]
        attn_out = attn_out.squeeze(1)  # [B*L, C]
        
        # 5. 重塑回原始形状
        attn_out = attn_out.view(B, L, C)  # [B, L, C]
        
        # 6. 残差连接和 FFN
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm))  # 在FFN后加Dropout
        
        return x

class StandardCrossPlaneAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        """
        x: [B, H*W, C]
        context: [B, H*W, S, C] -> [B, H*W*S, C]
        """
        B, L, S, C = context.shape
               # 1. 归一化
        x_norm = self.norm1(x)
        context_norm = self.norm1(context)
        
        # 2. 重塑为 [B*L, ...] 以进行批量处理
        x_flat = x_norm.view(B * L, C)  # [B*L, C]
        context_flat = context_norm.view(B * L, S, C)  # [B*L, V, C]
        
        # 3. 添加 query 维度
        x_flat = x_flat.unsqueeze(1)  # [B*L, 1, C]
        
        # 4. 批量注意力计算
        attn_out, _ = self.attn(x_flat, context_flat, context_flat)  # [B*L, 1, C]
        attn_out = attn_out.squeeze(1)  # [B*L, C]
        
        # 5. 重塑回原始形状
        attn_out = attn_out.view(B, L, C)  # [B, L, C]
        
        # 6. 残差连接和 FFN
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        
        return x
    

class ImageTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, n_heads=4, feature_channels=64, hidden_channels=64, num_samples=3, num_layers=2):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels
        self.num_samples = num_samples
        self.num_layers = num_layers
        self.feature_channels = feature_channels
        # 初始化 query tensors
        self.query_tensors = nn.ParameterList([
            nn.Parameter(torch.randn(grid_size, grid_size, hidden_channels)) for _ in range(3)
        ])
        for i in range(3):
            nn.init.xavier_uniform_(self.query_tensors[i])
        # 替换为标准 Attention
        self.attentions_cross_view = nn.ModuleList([
            StandardCrossViewAttention(hidden_channels, n_heads) for _ in range(num_layers)
        ])
        # self.attentions_cross_plane = nn.ModuleList([
        #     StandardCrossPlaneAttention(hidden_channels, n_heads) for _ in range(num_layers)
        # ])

        # self.input_feature_conv = nn.Conv2d(feature_channels, hidden_channels, 3, 1, 1)
        self.bg_color = torch.zeros(hidden_channels, device="cuda", requires_grad=False)

        self.sample_points = {}
        for i, plane in enumerate(['xy', 'xz', 'yz']):
            self.sample_points[plane] = self.sample_points_along_axis(plane)
        
    def sample_points_along_axis(self, plane):
        """沿平面法线方向采样"""
        H, W, S = self.grid_size, self.grid_size, self.num_samples
        device = 'cuda:0'

        x = torch.linspace(-1, 1, W, device=device, requires_grad=False)  # 这些是固定的采样点，不需要梯度
        y = torch.linspace(-1, 1, H, device=device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # [H, W]

        if plane == 'xy':
            # z 轴采样 S 个点
            z_samples = torch.linspace(-1, 1, S, device=device, requires_grad=False)
            # 扩展 grid_x, grid_y 到 [H, W, S]
            xx = grid_x.unsqueeze(-1).expand(H, W, S)  # [H, W, S]
            yy = grid_y.unsqueeze(-1).expand(H, W, S)  # [H, W, S]
            zz = z_samples.view(1, 1, S).expand(H, W, S)  # [H, W, S]
            coords = torch.stack([xx, yy, zz], dim=-1)  # [H, W, S, 3]

        elif plane == 'xz':
            y_samples = torch.linspace(-1, 1, S, device=device, requires_grad=False)
            xx = grid_x.unsqueeze(-1).expand(H, W, S)
            yy = y_samples.view(1, 1, S).expand(H, W, S)
            zz = grid_y.unsqueeze(-1).expand(H, W, S)
            coords = torch.stack([xx, yy, zz], dim=-1)  # [H, W, S, 3]

        else:  # 'yz'
            x_samples = torch.linspace(-1, 1, S, device=device, requires_grad=False)
            xx = x_samples.view(1, 1, S).expand(H, W, S)
            yy = grid_x.unsqueeze(-1).expand(H, W, S)
            zz = grid_y.unsqueeze(-1).expand(H, W, S)
            coords = torch.stack([xx, yy, zz], dim=-1)  # [H, W, S, 3]

        return coords  # [H, W, S, 3]
    
    def project_to_other_planes(self, points, plane, cross_view_plane_feats):
        """
        将3D点投影到其他两个平面
        Args:
            points: [H, W, num_samples, 3] 3D点坐标
            plane: 当前平面
        Returns:
            other_plane_features: [H, W, num_samples*2, C] 其他两个平面的特征
        """

        H, W = points.shape[:2]
        B = cross_view_plane_feats[0].shape[0]
        features = []
        
        # 创建平面到索引的映射
        plane_to_idx = {'xy': 0, 'xz': 1, 'yz': 2}
        
        for other_plane in ['xy', 'xz', 'yz']:
            if other_plane == plane:
                continue
                
            # 获取投影坐标
            if other_plane == 'xy':
                coords = points[..., [0, 1]]
            elif other_plane == 'xz':
                coords = points[..., [0, 2]]
            else:  # 'yz'
                coords = points[..., [1, 2]]
                
            # 将坐标映射到网格索引
            grid_coords = (coords * 0.5 + 0.5) * (self.grid_size - 1)
            grid_coords = grid_coords.clamp(0, self.grid_size-1)

            # 获取特征
            plane_feat = rearrange(cross_view_plane_feats[plane_to_idx[other_plane]], "b (h w) c -> b c h w", h=H, w=W) # [C, H, W]
            # 重塑grid_coords以适应grid_sample的要求
            grid_coords = grid_coords.view(H*W*self.num_samples, 1, 2)  # [1, H*W*num_samples, 1, 2]
            grid_coords = grid_coords.unsqueeze(0).expand(B, H*W*self.num_samples, 1, 2)

            # 使用grid_sample采样特征
            sampled_feat = F.grid_sample(
                plane_feat,
                grid_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # [B, C, H*W*num_samples, 1]


            # 重塑采样后的特征
            sampled_feat = sampled_feat.squeeze(-1)  # [B, C, H*W*num_samples]
            sampled_feat = sampled_feat.view(B, self.hidden_channels, H, W, self.num_samples)  # [B, C, H, W, num_samples]
            sampled_feat = sampled_feat.permute(0, 2, 3, 4, 1)  # [B, H, W, num_samples, C]
            features.append(sampled_feat)
            
        return torch.cat(features, dim=3)  # [B, H, W, num_samples*2, C]
    
    def Splatting_Points_to_Plane_trilinear(self, features, pts):
        """
        features: [BV, C, N]  # N = H*W
        pts:      [BV, N, 3]  # 已归一化到 [-1, 1]
        returns: triplane: [BV, 3, C, R, R]
        """
        BV, C, N = features.shape
        device = features.device
        R = self.grid_size

        # 转置 features 为 [BV, N, C]
        features = features.permute(0, 2, 1)  # [BV, N, C]

        # 初始化 triplane 和权重累加器: [BV, 3, C, R, R]
        triplane = torch.zeros(BV, 3, C, R, R, device=device)
        weight_accumulator = torch.zeros(BV, 3, 1, R, R, device=device)  # 用于归一化的权重累加器

        # 将点归一化到 grid 坐标 [0, R-1]
        grid_coords = (pts + 1) * (R - 1) / 2  # [BV, N, 3]
        grid_coords = grid_coords.clamp(0, R - 1 - 1e-5)
        x, y, z = grid_coords[..., 0], grid_coords[..., 1], grid_coords[..., 2]

        # 获取整数坐标和权重
        x0 = x.long(); x1 = x0 + 1; wx = x - x0.float()
        y0 = y.long(); y1 = y0 + 1; wy = y - y0.float()
        z0 = z.long(); z1 = z0 + 1; wz = z - z0.float()

        w000 = (1-wx) * (1-wy) * (1-wz)
        w001 = (1-wx) * (1-wy) * wz
        w010 = (1-wx) * wy * (1-wz)
        w011 = (1-wx) * wy * wz
        w100 = wx * (1-wy) * (1-wz)
        w101 = wx * (1-wy) * wz
        w110 = wx * wy * (1-wz)
        w111 = wx * wy * wz

        # 确保索引不越界
        x0 = x0.clamp(0, R-1); x1 = x1.clamp(0, R-1)
        y0 = y0.clamp(0, R-1); y1 = y1.clamp(0, R-1)
        z0 = z0.clamp(0, R-1); z1 = z1.clamp(0, R-1)

        # 创建批量索引
        batch_idx = torch.arange(BV, device=device).unsqueeze(1)  # [BV, 1]

        # 注入到 XY 平面（固定 z）
        triplane[batch_idx, 0, :, x0, y0] += w000.unsqueeze(-1) * features
        triplane[batch_idx, 0, :, x0, y1] += w010.unsqueeze(-1) * features
        triplane[batch_idx, 0, :, x1, y0] += w100.unsqueeze(-1) * features
        triplane[batch_idx, 0, :, x1, y1] += w110.unsqueeze(-1) * features
        
        # 累加权重到 XY 平面
        weight_accumulator[batch_idx, 0, 0, x0, y0] += w000
        weight_accumulator[batch_idx, 0, 0, x0, y1] += w010
        weight_accumulator[batch_idx, 0, 0, x1, y0] += w100
        weight_accumulator[batch_idx, 0, 0, x1, y1] += w110

        # 注入到 XZ 平面（固定 y）
        triplane[batch_idx, 1, :, x0, z0] += w000.unsqueeze(-1) * features
        triplane[batch_idx, 1, :, x0, z1] += w001.unsqueeze(-1) * features
        triplane[batch_idx, 1, :, x1, z0] += w100.unsqueeze(-1) * features
        triplane[batch_idx, 1, :, x1, z1] += w111.unsqueeze(-1) * features
        
        # 累加权重到 XZ 平面
        weight_accumulator[batch_idx, 1, 0, x0, z0] += w000
        weight_accumulator[batch_idx, 1, 0, x0, z1] += w001
        weight_accumulator[batch_idx, 1, 0, x1, z0] += w100
        weight_accumulator[batch_idx, 1, 0, x1, z1] += w111

        # 注入到 YZ 平面（固定 x）
        triplane[batch_idx, 2, :, y0, z0] += w000.unsqueeze(-1) * features
        triplane[batch_idx, 2, :, y0, z1] += w001.unsqueeze(-1) * features
        triplane[batch_idx, 2, :, y1, z0] += w010.unsqueeze(-1) * features
        triplane[batch_idx, 2, :, y1, z1] += w111.unsqueeze(-1) * features
        
        # 累加权重到 YZ 平面
        weight_accumulator[batch_idx, 2, 0, y0, z0] += w000
        weight_accumulator[batch_idx, 2, 0, y0, z1] += w001
        weight_accumulator[batch_idx, 2, 0, y1, z0] += w010
        weight_accumulator[batch_idx, 2, 0, y1, z1] += w111

        # 归一化：除以权重累加器（避免除零）
        weight_accumulator = weight_accumulator.clamp(min=1e-8)
        triplane = triplane / weight_accumulator

        return triplane  # [BV, 3, C, R, R]
    
    def Splatting_Points_to_Plane(self, feature, pts):
        """
        将归一化后的点云投影到三平面上，并考虑透明度权重
        
        参数:
            bounded_GS_feats: 归一化后的点云特征，形状为 [N, C]
                            其中前三位是坐标，第四位是透明度，后面是特征
            tri_planes: 已经初始化的三平面字典，包含'xy'、'xz'、'yz'三个键
            
        返回:
            直接修改传入的tri_planes
        """
        # 分离坐标、透明度和特征
        coords = pts  # 前三位是坐标
        features = feature # 特征

        screenspace_points = torch.zeros_like(coords, dtype=coords.dtype, requires_grad=True, device=coords.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2D = screenspace_points

        shs = None
        B, C, N = feature.shape

        colors_precomp =  rearrange(features,'B C N -> B N C')
        
        colors_precomp_count = torch.ones_like(colors_precomp, dtype=torch.float32, device="cuda", requires_grad=False)

        cov3D_precomp = None
        scales = torch.full([B, N, 3], 1.0 / self.grid_size, dtype=torch.float32, device="cuda", requires_grad=False)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda", requires_grad=False)
        rotations = identity_quat.repeat(B, N, 1)
        opacities = torch.full([B, N, 1], 0.067, dtype=torch.float32, device="cuda", requires_grad=False)

        extris = []
        intris = []
        # 对每个平面进行投影
        tri_planes = {}
        for plane in ['xy', 'xz', 'yz']:
            # 选择对应的坐标轴
            if plane == 'xy':
                param = self.camera_params_xy
            if plane == 'xz':
                param = self.camera_params_xz
            if plane == 'yz':
                param = self.camera_params_yz       

            extri = param["extrinsic_matrix"].unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4)
            intri = param["intrinsic_matrix"].unsqueeze(0).unsqueeze(0).expand(B, 1, 3, 3)
            extris.append(extri)
            intris.append(intri)

        extris = torch.cat(extris, dim=1)
        intris = torch.cat(intris, dim=1)

        rendered_image, radii, depth = rasterization(
            means = coords,
            quats=rotations,
            scales=scales,  # [N, 3]
            opacities=opacities.squeeze(-1),  # [N,]
            colors=colors_precomp,
            viewmats=extris,
            Ks=intris,
            width = param["image_width"],
            height = param["image_height"],
            near_plane = 0.1,
            far_plane = 10,
            backgrounds=self.bg_color,
            camera_model = "ortho")

        count_image, radii, depth = rasterization(
            means = coords,
            quats=rotations,
            scales=scales,  # [N, 3]
            opacities=opacities.squeeze(-1),  # [N,]
            colors=colors_precomp_count,
            viewmats = extris,
            Ks = intris,
            width = param["image_width"],
            height = param["image_height"],
            near_plane = 0.1,
            far_plane = 10,
            backgrounds=self.bg_color,
            camera_model = "ortho")


        count_image = count_image.clamp(min=1e-6)
        tri_planes = rendered_image / count_image

        return tri_planes   
     
    def forward(self, image_features, means, scene_bounds=None):

        B, Nv = image_features.shape[:2]

        # Nv=1
        assert image_features.shape[2] == self.feature_channels, \
            f"特征通道数不匹配: 期望 {self.feature_channels}, 实际 {image_features.shape[2]}"  
        # image_features = rearrange(image_features, "b v h w c -> (b v) c h w")
        # # image_features = self.input_feature_conv(image_features)
        # image_features = rearrange(image_features, "(b v) c h w -> b v c h w", b = B, v = Nv)

        # 计算场景边界
        if scene_bounds == None:
            scene_bounds = []
            for b in range(0, B):
                bounds = torch.tensor([10,-10,10,-10,10,-10], dtype=float, device='cuda')
                world_points = rearrange(means, "b v h w xyz -> b v (h w) xyz")
                    
                bounds[0] = min(world_points[b, ..., 0].min(), bounds[0])
                bounds[1] = max(world_points[b, ..., 0].max(), bounds[1])
                bounds[2] = min(world_points[b, ..., 1].min(), bounds[2])
                bounds[3] = max(world_points[b, ..., 1].max(), bounds[3])
                bounds[4] = min(world_points[b, ..., 2].min(), bounds[4])
                bounds[5] = max(world_points[b, ..., 2].max(), bounds[5])

                padding = 0.05
                scene_bound = [
                    bounds[0] - padding * (bounds[1] - bounds[0]),
                    bounds[1] + padding * (bounds[1] - bounds[0]),
                    bounds[2] - padding * (bounds[3] - bounds[2]),
                    bounds[3] + padding * (bounds[3] - bounds[2]),
                    bounds[4] - padding * (bounds[5] - bounds[4]),
                    bounds[5] + padding * (bounds[5] - bounds[4]),            
                ]
                scene_bounds.append(scene_bound)
        # render_point, r_mask = project_depth_to_3d(
        #     depth[0,0], c2w[0], intrinsic[0]
        # )
        query_tensor_input = self.query_tensors

        splat_feats = []
        bounded_points = []
        for b in range(B):

            world_points = means[b]
            # ds_mask = voxel_grid_downsample_mask(world_points, 0.015, 'mean')

            # mask = ds_mask & mask

            bounded_point = norm_points_bounds(world_points, scene_bounds[b])
            bounded_points.append(bounded_point.unsqueeze(0))

            # ft = image_features.permute(0, 1, 3, 4, 2)
            # save_xyz_tensor_as_ply(world_points, ft[b], "pt.ply")

                # 对每个平面进行处理
        bounded_points = torch.cat(bounded_points, dim=0)



        feature = rearrange(image_features, "b v c h w -> (b v) c (h w)")
        pts = rearrange(bounded_points, "b v h w c -> (b v) (h w) c")

        tri_plane = self.Splatting_Points_to_Plane_trilinear(feature, pts)

        splat_feats = rearrange(tri_plane, "(b v) p c h w -> b p v c h w", b=B, v=Nv)


        # save_feats = splat_feats.mean(dim=2)[0]
        # for i, plane in enumerate(['xy', 'xz', 'yz']):
        #     save_feat = save_feats.squeeze(0)[i]
        #     save_feat_img = save_feat.detach().cpu()
        #     save_feat_img = (save_feat_img - save_feat_img.min()) / (save_feat_img.max() - save_feat_img.min() + 1e-8)
        #     vutils.save_image(save_feat_img, f'save_feat_{plane}.png')




        query_tensor_input = []
        for i in range(3):
            query_tensor_input.append(self.query_tensors[i].unsqueeze(0).expand(B, self.grid_size, self.grid_size, self.hidden_channels))

        for n in range(0, self.num_layers):

            cross_view_plane_feats = []
            for i, plane in enumerate(['xy', 'xz', 'yz']):

                queries = query_tensor_input[i]  # [B, H, W, C]
        

                # 重塑query和特征以进行批处理
                queries_flat = queries.view(B, -1, self.hidden_channels)  # [B, H*W, C]

                plane_feat = splat_feats[:, i].permute(0, 3, 4, 1, 2).view(B, -1, Nv, self.hidden_channels)
                
                attn_output = self.attentions_cross_view[n](
                    queries_flat,  # [B, HW, C]
                    plane_feat          # [B, HW, V, C]
                )  # 输出: [B, HW, C]
                cross_view_plane_feats.append(attn_output)
            # cross_plane_plane_feats = []
            # for i, plane in enumerate(['xy', 'xz', 'yz']):
            #     plane_feat = cross_view_plane_feats[i]  # [B, H*W, C]
            #     # 采样点
            #     sampled_points = self.sample_points[plane]
            #     # 投影到其他平面并获取特征
            #     other_plane_features = self.project_to_other_planes(sampled_points, plane, cross_view_plane_feats)
            #     other_plane_features = other_plane_features.view(B, -1, self.num_samples * 2, self.hidden_channels)

            #     attn_output = self.attentions_cross_plane[n](
            #             plane_feat,  # [B, H*W, C]
            #             other_plane_features  # [B, H*W, self.num_samples * 2, C]
            #         )  # [B, H*W, C]
        
            #     cross_plane_plane_feats.append(rearrange(attn_output, "b (h w) c -> b h w c", h=self.grid_size, w=self.grid_size))
            
            query_tensor_input = [rearrange(cross_view_plane_feats[0], "b (h w) c -> b h w c", h=self.grid_size, w=self.grid_size), 
                                            rearrange(cross_view_plane_feats[1], "b (h w) c -> b h w c", h=self.grid_size, w=self.grid_size), 
                                            rearrange(cross_view_plane_feats[2], "b (h w) c -> b h w c", h=self.grid_size, w=self.grid_size)]
        for i in range(3):
            query_tensor_input[i] = query_tensor_input[i].permute(0, 3, 1, 2).unsqueeze(1)

        return torch.cat(query_tensor_input, dim=1), scene_bounds

import numpy as np

def save_xyz_tensor_as_ply(tensor, tensor2, output_file):
    """
    将 (H, W, 3) 的张量（表示 XYZ 坐标）保存为 PLY 点云文件。
    
    参数:
        tensor (numpy.ndarray): 输入张量，形状为 (H, W, 3)，表示 XYZ 坐标。
        output_file (str): 输出的 PLY 文件路径。
    """

    
    # 展平张量为 (H*W, 3)
    point_cloud = tensor.reshape(-1, 3)
    color = tensor2.reshape(-1, 3).cpu().numpy()
    
    # 归一化颜色到 0-255，只做一次归一化
    color = ((color - color.min()) / (color.max() - color.min() + 1e-8) * 255).astype(np.uint8)
    
    # 写入 PLY 文件
    with open(output_file, 'w') as f:
        # 写入头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")    # 改为 uchar
        f.write("property uchar green\n")  # 改为 uchar
        f.write("property uchar blue\n")   # 改为 uchar
        f.write("end_header\n")
        
        # 写入点云数据
        for i in range(0, len(point_cloud)):
            x, y, z = point_cloud[i]
            r, g, b = color[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def voxel_grid_downsample_mask(
    points: torch.Tensor,
    voxel_size: float,
    reduction: str = 'first'
) -> list[torch.Tensor]:
    """
    对有组织的点云进行体素网格降采样，并返回一个布尔 mask。

    Args:
        points (torch.Tensor): 输入点云，维度为 (N, H, W, C)。
                                N 是批次大小，H, W 是图像尺寸，C 是坐标维度 (例如 3 表示 x, y, z)。
        voxel_size (float): 体素网格的大小。一个较大的值会导致更多的降采样。

    Returns:
        torch.Tensor: 一个布尔 mask，维度为 (N, H, W)。
                      如果 mask[n, h, w] 为 True，表示原始点 points[n, h, w] 被保留；
                      如果为 False，表示该点被降采样移除。
    """
    if points.dim() != 4:
        raise ValueError("输入点云的维度必须是 (N, H, W, C)。")
    
    N, H, W, C = points.shape
    device = points.device

    # 初始化一个与原始点云相同形状的布尔 mask，全部设为 False
    # 之后被保留的点将设为 True
    boolean_mask = torch.full((N, H, W), False, dtype=torch.bool, device=device)

    # 将点云展平以便进行通用点云操作 (N, H*W, C)
    flat_points_batch = points.view(N, H * W, C)

    for i in range(N):
        current_points = flat_points_batch[i] # 当前视图的展平点云 (H*W, C)
        
        # 如果当前点云为空，则跳过
        if current_points.numel() == 0:
            continue

        # 计算每个点所属的体素坐标
        # 将点坐标归一化到体素网格，然后取整得到体素索引
        min_coords = current_points.min(dim=0, keepdim=True).values
        min_coords = min_coords.to(current_points.dtype)

        # 将点云平移到正空间，再进行体素化
        shifted_points = current_points - min_coords
        voxel_coords = (shifted_points / voxel_size).floor().long()

        # 计算一维体素 ID
        max_voxel_coords = voxel_coords.max(dim=0, keepdim=True).values + 1
        
        # 处理特殊情况：如果点云太小或全部在一个体素内导致 max_voxel_coords 无效
        if max_voxel_coords.numel() == 0 or torch.any(max_voxel_coords <= 0):
            # 如果只有一个点，它当然被保留
            if current_points.shape[0] == 1:
                boolean_mask[i, 0, 0] = True # 假设这个点来自 (0,0) 位置
            # 对于其他无法体素化的情况，所有点都不保留（或根据需求决定）
            continue 

        voxel_ids = voxel_coords[:, 0] \
                    + voxel_coords[:, 1] * max_voxel_coords[0, 0] \
                    + voxel_coords[:, 2] * max_voxel_coords[0, 0] * max_voxel_coords[0, 1]

        # 核心逻辑：找出每个体素中的第一个点
        # 对 voxel_ids 进行排序，并记录原始索引
        sorted_voxel_ids, sort_indices = torch.sort(voxel_ids)
        
        # 找出每个唯一 voxel_id 第一次出现的位置
        # 第一个元素总是第一个出现，后面的元素如果与前一个不同，也是第一个出现
        first_occurrence_mask_1d = torch.cat([
            torch.tensor([True], device=device), # 第一个元素总是 True
            sorted_voxel_ids[1:] != sorted_voxel_ids[:-1] # 比较相邻元素是否不同
        ])
        
        # 获取被保留的点的原始展平索引
        # 这些索引指向的是 flat_points_batch[i] 中的位置
        preserved_flat_indices = sort_indices[first_occurrence_mask_1d]

        # 将这些展平索引转换回 (H, W) 格式，并设置布尔 mask 为 True
        # 原始索引 flat_idx = h * W + w
        # 那么 h = flat_idx // W
        # w = flat_idx % W
        
        preserved_h = preserved_flat_indices // W
        preserved_w = preserved_flat_indices % W
        
        # 将对应位置的布尔 mask 设为 True
        boolean_mask[i, preserved_h, preserved_w] = True

    return boolean_mask
