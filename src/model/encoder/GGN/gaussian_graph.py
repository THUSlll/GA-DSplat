# Copyright (c) 2024 Shengjun Zhang
# Copyright (c) 2025 Bo Liu
# Licensed under the MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from einops import rearrange
from ....geometry.projection import project
try:
    from pytorch3d.ops import knn_points
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not found. Please install pytorch3d for KNN-based merging.")
# ----------------
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

def scatter_add(src, index, out, dim=-1):
    """A simple scatter_add implementation for older PyTorch versions or clarity.
    Assumes index has the same shape as src up to the last dimension.
    """
    return out.scatter_add_(dim, index.expand_as(src), src)

def compute_adjacency_tilde(extrinsics: torch.Tensor, k_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据相机外参计算邻接矩阵 A 和归一化的邻接矩阵 A_tilde = D^{-1}A。
    这里使用一种简化的重叠度计算方法，并只保留每个节点的 Top-K 个连接。

    参数:
        extrinsics (torch.Tensor): 相机外参矩阵。形状: (B, V, 4, 4)。

    返回:
        tuple[torch.Tensor, torch.Tensor]: 邻接矩阵 A 和归一化的邻接矩阵 A_tilde。
                                            形状均为: (B, V, V)。
    """
    B, V, _, _ = extrinsics.shape
    device = extrinsics.device
    
    # 提取相机中心 (世界坐标系下)
    # 外参矩阵 [R|t] 将世界坐标转换到相机坐标。
    # 相机中心 c_cam = -R^T * t (在世界坐标系下)
    R = extrinsics[:, :, :3, :3]  # (B, V, 3, 3)
    t = extrinsics[:, :, :3, 3]   # (B, V, 3)
    cam_centers = -torch.matmul(R.transpose(-1, -2), t.unsqueeze(-1)).squeeze(-1) # (B, V, 3)

    # --- 简化重叠度计算 (基于相机中心距离的反比) ---
    # 计算所有相机中心对之间的欧氏距离平方
    cam_centers_expanded_1 = cam_centers.unsqueeze(2) # (B, V, 1, 3)
    cam_centers_expanded_2 = cam_centers.unsqueeze(1) # (B, 1, V, 3)
    dist_sq = torch.sum((cam_centers_expanded_1 - cam_centers_expanded_2) ** 2, dim=-1) # (B, V, V)

    # 将距离转换为相似度（重叠度的一种近似）
    # 距离越近，相似度越高。为避免除以零，加一个很小的epsilon。
    epsilon = 1e-6
    # 使用负距离平方来表示相似度，这样相似度越高值越大
    # 或者使用 1 / (1 + distance) 如之前
    # overlap_raw = 1.0 / (1.0 + torch.sqrt(dist_sq + epsilon)) # (B, V, V)
    # 为了简化 Top-K 操作，我们直接使用负距离平方 (越大越相似)
    similarity_raw = -dist_sq # (B, V, V)

    # 创建初始邻接矩阵 A，对角线为0 (自连接稍后处理)
    A = similarity_raw.clone()
    A.diagonal(dim1=-2, dim2=-1)[:] = float('-inf') # (B, V, V) # 将对角线设为 -inf 以便 Top-K 不选自己（除非 K >= V）

    # --- 保留 Top-K 边 ---
    if k_neighbors > 0 and k_neighbors < V:
        # 对每一行（每个节点）找 top-K 最大的相似度（最小的负距离平方）
        # topk 返回值和索引
        _, topk_indices = torch.topk(A, k_neighbors, dim=-1, sorted=False) # (B, V, k_neighbors)
        
        # 创建一个与 A 同形状的掩码，初始化为 False
        topk_mask = torch.zeros_like(A, dtype=torch.bool) # (B, V, V)
        
        # 构造用于 scatter 的索引
        # 我们需要将 topk_indices (B, V, k) 的索引 scatter 到 topk_mask (B, V, V) 的最后一个维度
        # topk_indices 的形状是 (B, V, k_neighbors)
        # 我们需要一个行索引 (B, V, k_neighbors) 指向 topk_mask 的前两个维度
        batch_indices = torch.arange(B, device=device).view(B, 1, 1) # (B, 1, 1)
        node_indices = torch.arange(V, device=device).view(1, V, 1) # (1, V, 1)
        
        # 使用 scatter_ 将 True 值放到 top-K 位置
        topk_mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices, dtype=torch.bool))
        
        # 应用掩码到原始相似度
        # 将非 Top-K 的位置设为 0，Top-K 的位置保留原始相似度值（或转换回距离）
        # 为了保持 A 的语义是重叠度/相似度，我们保留原始计算的 overlap_raw 值
        # 但 overlap_raw 是 1/(1+d)，我们用 similarity_raw = -d^2 来找 topk
        # 最简单的是直接用 overlap_raw 的值，但只保留 topk 位置
        A_overlap = 1.0 / (1.0 + torch.sqrt(dist_sq + epsilon)) # (B, V, V) - 重新计算 overlap
        A = A_overlap * topk_mask.float() # (B, V, V)
        
    else:
        # 如果 k_neighbors <= 0 或 >= V，则保留所有边
        A = 1.0 / (1.0 + torch.sqrt(dist_sq + epsilon)) # (B, V, V)

    # 设置自连接 (对角线为1)
    A.diagonal(dim1=-2, dim2=-1)[:] = 1.0 # (B, V, V)

    # --- 计算度矩阵 D 和归一化 A_tilde = D^{-1}A ---
    # 度矩阵 D 是 A 的行和
    degrees = torch.sum(A, dim=-1, keepdim=True) # (B, V, 1)
    # 防止除以零
    degrees_inv = 1.0 / (degrees + (degrees == 0).float()) # (B, V, 1)

    # A_tilde = D^{-1} @ A (矩阵乘法)
    # 由于 D^{-1} 是对角矩阵，可以简化为逐行缩放
    A_tilde = A * degrees_inv # (B, V, V)

    return A, A_tilde # (B, V, V), (B, V, V)
    
class GGNLinearLayer(nn.Module):
    """
    实现了 Gaussian Graph Network (GGN) 中的线性层功能。
    该层根据高斯点在不同视图间的投影关系和视图间的邻接权重来聚合和更新特征。
    """
    def __init__(self, in_channels: int, out_channels: int, feature_map_height: int, feature_map_width: int, k_neighbors: int = 3):
        """
        初始化GGN线性层。

        参数:
            in_channels (int): 输入特征的通道数 (C)。
            out_channels (int): 输出特征的通道数 (D)。
            feature_map_height (int): 特征图的高度 (H)。
            feature_map_width (int): 特征图的宽度 (W)。
            k_neighbors (int): 用于构建邻接矩阵的每个节点的最近邻数量（不包括自身连接）。
                               默认为3，旨在使每个视图（除自身外）与其他视图连接。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_h = feature_map_height
        self.feature_w = feature_map_width
        self.k_neighbors = k_neighbors
        # self.linear = nn.Sequential(
        #     nn.Linear(in_channels, out_channels),
        #     nn.GELU(),
        #     nn.Linear(out_channels, out_channels)
        # )


    def forward(self, features: torch.Tensor, xy_coords: torch.Tensor, extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。

        参数:
            features (torch.Tensor): 输入的高斯点特征。形状: (B, V, G, C) 或 (B, V, HW, N, C)。
            xy_coords (torch.Tensor): 高斯点投影到各视图的2D坐标。形状: (B, V_src, V_tgt, G, 2)。
            extrinsics (torch.Tensor): 相机外参矩阵。形状: (B, V, 4, 4)。

        返回:
            tuple[torch.Tensor, torch.Tensor]: 更新后的高斯点特征 和 邻接矩阵 A。
                                               形状: (B, V, G, D), (B, V, V)。
        """
        # --- 输入处理 ---

        features = self.linear(features)

        if features.dim() == 5:
            B, V, HW, N, C = features.shape
            G = HW * N
            features = features.reshape(B, V, G, C)
        else:
            B, V, G, C = features.shape
        

        if xy_coords.shape[3] != G:
             _, _, _, HW_coord, _ = xy_coords.shape
             N_mult = G // HW_coord
             xy_coords = xy_coords.unsqueeze(4).expand(B, V, V, HW_coord, N_mult, 2).reshape(B, V, V, G, 2)

        device = features.device

        # --- 计算邻接矩阵 A 和 A_tilde ---
        A, A_tilde = compute_adjacency_tilde(extrinsics) # (B, V, V), (B, V, V)
        # A_tilde[i, j] 表示从视图 j 到视图 i 的归一化权重

        # --- 特征聚合 ---
        output_features_list = []

        for i in range(V): # 对于每个目标视图 i
            # 初始化聚合特征网格 (在视图 i 的图像平面)
            aggregated_grid = torch.zeros(B, self.feature_h * self.feature_w, C, device=device) # (B, H*W, C)

            # 获取所有源视图投影到目标视图 i 的坐标
            projections_on_i = xy_coords[:, :, i, :, :] # (B, V, G, 2)

            # 将2D坐标转换为1D像素索引
            x = torch.clamp(torch.round(projections_on_i[..., 0]), 0, self.feature_w - 1).long() # (B, V, G)
            y = torch.clamp(torch.round(projections_on_i[..., 1]), 0, self.feature_h - 1).long() # (B, V, G)
            flat_indices = y * self.feature_w + x # (B, V, G)

            # 准备源特征和索引用于 scatter_add
            source_features_flat = features.reshape(B, V * G, C) # (B, V*G, C)
            indices_flat = flat_indices.reshape(B, V * G) # (B, V*G)

            # --- 应用邻接权重 ---
            # 对于目标视图 i，我们需要权重 a_tilde_{i,j} (j=0,...,V-1)
            weights_for_i = A_tilde[:, i, :].unsqueeze(-1) # (B, V, 1)
            # 将权重广播到所有高斯点
            weights_flat = weights_for_i.expand(B, V, G).reshape(B, V * G, 1) # (B, V*G, 1)
            # 加权源特征
            weighted_source_features = source_features_flat * weights_flat # (B, V*G, C)

            # --- 执行聚合 (scatter_add) ---
            # 使用加权后的特征进行聚合
            # --- 执行聚合 (scatter_add) ---
            for b in range(B):
                # 确保索引在有效范围内
                indices_b = torch.clamp(indices_flat[b], 0, self.feature_h * self.feature_w - 1)
                indices_for_scatter = indices_b.unsqueeze(1).expand(-1, C)  # (V*G, C)
                aggregated_grid[b].scatter_add_(0, indices_for_scatter, weighted_source_features[b])

            # --- 提取自身视图 i 的聚合特征 ---
            # 获取视图 i 中的高斯点在其自身图像平面上的投影坐标
            self_projections = xy_coords[:, i, i, :, :] # (B, G, 2)
            x_self = torch.clamp(torch.round(self_projections[..., 0]), 0, self.feature_w - 1).long() # (B, G)
            y_self = torch.clamp(torch.round(self_projections[..., 1]), 0, self.feature_h - 1).long() # (B, G)
            self_indices_flat = (y_self * self.feature_w + x_self) # (B, G)

            # 使用 gather 从 aggregated_grid 中提取每个点对应的聚合特征
            expanded_self_indices = self_indices_flat.unsqueeze(-1).expand(-1, -1, C) # (B, G, C)
            # gathered_features[b, g, c] = aggregated_grid[b, self_indices_flat[b, g], c]
            gathered_features = torch.gather(aggregated_grid, 1, expanded_self_indices) # (B, G, C)
            
            output_features_list.append(gathered_features) # (B, G, C)

        # --- 整合所有视图的输出 ---
        # stacked_features: (B, V, G, C)
        stacked_features = torch.stack(output_features_list, dim=1)
        
        return stacked_features, A # (B, V, G, D), (B, V, V)
    

import torch
import torch.nn as nn

class GGNPoolingLayer(nn.Module):
    """
    实现了 Gaussian Graph Network (GGN) 中的池化层功能 (迭代合并)。
    该层根据高斯点的 3D 位置相似性和图结构 (邻接矩阵 A) 来迭代聚合连接的高斯图，
    旨在减少冗余并生成更紧凑的场景表示。
    注意：此版本返回一个掩码，用于指示保留的点，而不是直接返回筛选后的张量。
    这样可以方便地对高斯点的其他属性（如 scales, rotations, opacities 等）进行一致的筛选。
    """
    def __init__(self, lambda_threshold: float = 0.1):
        """
        初始化高斯池化层。
        参数:
            lambda_threshold (float): 用于合并的距离阈值 (lambda)。
        """
        super().__init__()
        self.lambda_threshold = lambda_threshold
        self.lambda_sq_threshold = lambda_threshold ** 2

    def find_connected_components(self, A: torch.Tensor) -> list[list[int]]:
        """查找图的连通分量。"""
        # A shape: (V, V)
        V = A.shape[0]
        visited = [False] * V
        components = []
        adj_matrix = A > 0
        for i in range(V):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    component.append(node)
                    # 使用 .nonzero 更简洁 (返回二维 tensor [[idx1], [idx2], ...])
                    neighbors = adj_matrix[node].nonzero(as_tuple=False).squeeze(-1)
                    for neighbor_idx_tensor in neighbors:
                        neighbor_idx = neighbor_idx_tensor.item()
                        if not visited[neighbor_idx]:
                            visited[neighbor_idx] = True
                            stack.append(neighbor_idx)
                components.append(sorted(component)) # 对组件内视图索引排序，便于处理
        return components

    def merge_component(self, features: torch.Tensor, means: torch.Tensor,
                        xy_coords: torch.Tensor, component: list[int]) -> torch.Tensor:
        """
        在一个连通分量内进行迭代合并，并返回一个指示保留点的掩码。
        新思路实现：使用 PyTorch3D KNN 替代计算所有距离矩阵，提高效率并保持梯度。
        目标：如果一个点到聚合点云中最近点的距离平方小于阈值，
             且该点与最近点在当前视图下投影坐标匹配，则删除该点。
        """
        device = features.device
        V_component, G, C = features.shape
        # 初始化掩码，维度为 (V_component, G)
        keep_mask_component = torch.ones(V_component, G, dtype=torch.bool, device=device)

        if V_component <= 1 or not PYTORCH3D_AVAILABLE:
            # 如果只有一个视图，或 PyTorch3D 不可用，则全部保留 (或作为 fallback)
            return keep_mask_component

        # --- 迭代聚合策略 ---
        # 1. 初始化：将第一个视图的所有点作为初始聚合集
        initial_view_local_idx = 0
        # current_agg_means 存储聚合集中点的 3D 坐标 (G_agg_total, 3)
        current_agg_means = means[initial_view_local_idx] # (G, 3)
        # current_agg_xy_on_views 存储聚合集中点投影到所有视图的坐标 (V_component, G_agg_total, 2)
        current_agg_xy_on_views = xy_coords[initial_view_local_idx, :, :, :] # (V_component, G, 2)

        # 2. 迭代处理后续视图
        for view_local_idx in range(1, V_component):
            means_current_view = means[view_local_idx] # (G_current, 3)
            xy_current_view = xy_coords[view_local_idx, view_local_idx, :, :] # (G_current, 2)
            G_current = means_current_view.shape[0]

            if G_current == 0 or current_agg_means.shape[0] == 0:
                # 当前视图无点，或聚合集为空，则当前视图所有点都保留（默认为 True）
                continue

            G_agg_total = current_agg_means.shape[0]
            xy_agg_on_current_view = current_agg_xy_on_views[view_local_idx, :, :] # (G_agg_total, 2)

            # --- 使用 PyTorch3D 进行 KNN 搜索 ---
            # 1. 准备 KNN 输入 (需要 batch 维度)
            # PyTorch3D knn_points 期望输入形状为 (N, P, D)
            # N: batch size (我们设为 1)
            # P: 点数
            # D: 维度 (3D 坐标)
            # 查询点: 当前视图的点 (1, G_current, 3)
            # 参考点: 聚合点云 (1, G_agg_total, 3)
            query_points = means_current_view.unsqueeze(0) # (1, G_current, 3)
            reference_points = current_agg_means.unsqueeze(0) # (1, G_agg_total, 3)

            # 2. 执行 KNN 搜索 (K=1)
            K = 1
            # knn_points 返回一个 named tuple: (dists, idx, ...)
            # dists: (1, G_current, K) 最近邻距离的平方
            # idx: (1, G_current, K) 最近邻在 reference_points 中的索引
            knn_result = knn_points(query_points, reference_points, K=K, return_nn=False)

            # 3. 提取结果
            distances_sq = knn_result.dists.squeeze(0).squeeze(-1) # (G_current,) # 最近邻距离的平方
            
            # 条件1：距离足够近 (检查到聚合集中最近点的距离)
            distance_check_mask = distances_sq <= self.lambda_sq_threshold # (G_current,)

            # --- 修正：检查投影匹配 (与聚合集中任意点匹配) - 纯PyTorch实现 ---
            HASH_BASE = 100000
            # 1. 计算聚合集中所有点的投影坐标的哈希ID (G_agg_total,)
            agg_ids = xy_agg_on_current_view[:, 0] * HASH_BASE + xy_agg_on_current_view[:, 1] # (G_agg_total,)
            
            # 2. 计算当前视图所有点的投影坐标的哈希ID (G_current,)
            current_ids = xy_current_view[:, 0] * HASH_BASE + xy_current_view[:, 1] # (G_current,)
            
            # # 3. 进行广播比较，生成一个 (G_current, G_agg_total) 的布尔矩阵
            # #    match_matrix[i, j] 为 True 当且仅当 current_ids[i] == agg_ids[j]
            # match_matrix = (current_ids.unsqueeze(-1) == agg_ids.unsqueeze(0)) # (G_current, 1) vs (1, G_agg_total) -> (G_current, G_agg_total)
            
            # # 4. 沿着聚合集维度 (dim=1) 检查任意一个匹配，得到 (G_current,) 的掩码
            # projection_match_mask = torch.any(match_matrix, dim=1) # (G_current,) 如果当前点与聚合集中任意点匹配，则为 True

            projection_match_mask = torch.isin(current_ids, agg_ids)
            # --- 合并决策 ---
            # 同时满足：距离近 AND (与聚合集中任意点)投影匹配，则丢弃
            discard_mask = distance_check_mask & projection_match_mask # (G_current,)
            keep_mask_current_view = ~discard_mask # (G_current,)

            # --- 应用当前视图的掩码 ---
            keep_mask_component[view_local_idx] = keep_mask_current_view

            # 7. 更新聚合集：将当前视图新保留的点加入聚合集
            if torch.any(keep_mask_current_view):
                new_means_to_add = means_current_view[keep_mask_current_view] # (G_new, 3)
                new_xy_on_views_to_add = xy_coords[view_local_idx, :, keep_mask_current_view, :] # (V_component, G_new, 2)

                current_agg_means = torch.cat([current_agg_means, new_means_to_add], dim=0) # (G_agg_total_new, 3)
                current_agg_xy_on_views = torch.cat([current_agg_xy_on_views, new_xy_on_views_to_add], dim=1) # (V_component, G_agg_total_new, 2)

        # 初始视图的所有点默认保留 (由初始化 ones 确定)
        return keep_mask_component

    # forward 方法保持不变
    def forward(self, features: torch.Tensor, means: torch.Tensor,
                xy_coords: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数 - 迭代合并池化。
        """

        B, V, G, C = features.shape
        device = features.device
        _, _, _, _, three = means.shape
        assert three == 3, "Means should have 3 coordinates"
        means = rearrange(means, "b v h w c -> b v (h w) c")
        # 确保 xy_coords 是整数类型 (Long)
        if xy_coords.is_floating_point():
            xy_coords_long = torch.round(xy_coords).long()
        elif xy_coords.dtype != torch.long:
            xy_coords_long = xy_coords.long()
        else:
            xy_coords_long = xy_coords
        components_list = [self.find_connected_components(A[b]) for b in range(B)]
        batch_features_out = []
        batch_means_out = []
        batch_masks_out = []
        for b in range(B):
            features_b = features[b]       # (V, G, C)
            means_b = means[b]             # (V, G, 3)
            xy_coords_b = xy_coords_long[b] # (V, V, G, 2)
            A_b = A[b]                     # (V, V)
            components_b = components_list[b] # List of List[int] (view indices)
            # 初始化该 batch 样本的保留掩码 (V, G)
            keep_mask_b = torch.zeros(V, G, dtype=torch.bool, device=device)
            # 对每个连通分量进行处理
            for component_view_indices in components_b:
                if not component_view_indices:
                    continue
                # --- 提取子图数据 ---
                num_views_in_component = len(component_view_indices)
                if num_views_in_component == 0:
                    continue
                # 从完整 batch 数据中提取该连通分量的数据
                features_component = features_b[component_view_indices] # (N, G, C)
                means_component = means_b[component_view_indices]       # (N, G, 3)
                # 提取连通分量内部视图间的投影坐标
                xy_coords_component_internal = xy_coords_b[component_view_indices][:, component_view_indices, :, :] # (N, N, G, 2)
                # --- 调用 merge_component ---
                # mask_for_component_local: (num_views_in_component, G)
                try:
                    mask_for_component_local = self.merge_component(
                        features_component, means_component, xy_coords_component_internal, list(range(num_views_in_component))
                    )
                except Exception as e:
                    print(f"Error in merge_component for batch {b}, component {component_view_indices}: {e}")
                    # 可以选择保留所有点或丢弃所有点作为 fallback
                    mask_for_component_local = torch.ones(num_views_in_component, G, dtype=torch.bool, device=device)

                # --- 将局部掩码映射回全局掩码 ---
                for i, original_view_idx in enumerate(component_view_indices):
                    keep_mask_b[original_view_idx, :] = keep_mask_b[original_view_idx, :] | mask_for_component_local[i, :]
            # --- 使用最终掩码筛选所有属性 ---
            keep_indices_flat = torch.where(keep_mask_b.flatten())[0] # (G_out_b,)
            features_b_flat = features_b.reshape(V * G, C) # (V*G, C)
            means_b_flat = means_b.reshape(V * G, 3)       # (V*G, 3)
            final_features_b = features_b_flat[keep_indices_flat] # (G_out_b, C)
            final_means_b = means_b_flat[keep_indices_flat]       # (G_out_b, 3)
            batch_features_out.append(final_features_b)
            batch_means_out.append(final_means_b)
            batch_masks_out.append(keep_mask_b) # (V, G)
        # --- 处理变长输出 ---
        max_G_out = max([feat.shape[0] for feat in batch_features_out], default=0)
        if max_G_out == 0:
            padded_features = torch.zeros(B, 0, C, device=device, dtype=features.dtype)
            padded_means = torch.zeros(B, 0, 3, device=device, dtype=means.dtype)
            final_masks = torch.stack(batch_masks_out, dim=0) if batch_masks_out else torch.zeros(B, V, G, dtype=torch.bool, device=device)
            return padded_features, padded_means, final_masks # (B, 0, C), (B, 0, 3), (B, V, G)
        padded_features = torch.zeros(B, max_G_out, C, device=device, dtype=features.dtype)
        padded_means = torch.zeros(B, max_G_out, 3, device=device, dtype=means.dtype)
        for b in range(B):
            G_out_b = batch_features_out[b].shape[0]
            if G_out_b > 0:
                padded_features[b, :G_out_b, :] = batch_features_out[b]
                padded_means[b, :G_out_b, :] = batch_means_out[b]
        final_masks = torch.stack(batch_masks_out, dim=0) # (B, V, G)
        return padded_features, padded_means, final_masks # (B, G_max, C), (B, G_max, 3), (B, V, G)


class GaussianGraph(nn.Module):
    def __init__(self, in_channels, out_channels, window_size=1, gamma=0.1):
        super().__init__()
        self.window_size = window_size
        self.gamma = gamma
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        
    def forward(self, means, depths, gs_feats, intrinsics, extrinsics):
        '''
        Input:
            means: position of gaussians (b, v, h, w, xyz)
            depths: depths of pixels (b, v, h, w, srf, dpt)
            gs_feats: features of gaussians (b, v, h, w, c)
        Output:
            gs_feats: features of gaussians (b, v, h, w, c')
        '''
        b, v, h, w, xyz = means.shape
        device = means.device

        # build graph
        for i in range(b):
            view_feat_list = []
            for j in range(v):
                sq_j_feat = rearrange(gs_feats[i, j], "h w c -> (h w) c", h=h, w=w)
                norm = 1
                for k in range(v):                    
                    if j != k and j >= k - self.window_size and j <= k + self.window_size:
                        sq_k_feat = rearrange(gs_feats[i, k], "h w c -> (h w) c", h=h, w=w)
                        points_ndc, valid_z = project(points=means[i,j].reshape(-1, 3),
                                                    intrinsics=intrinsics[i,k],
                                                    extrinsics=extrinsics[i,k],
                                                    epsilon=1e-8)
                        valid_x = (points_ndc[:, 0] >= 0) & (points_ndc[:, 0] < 1)
                        valid_y = (points_ndc[:, 1] >= 0) & (points_ndc[:, 1] < 1)
                        
                        # convert ndc to pixel
                        points2d = torch.zeros_like(points_ndc)
                        points2d[:, 0] = (points_ndc[:, 0]) * w
                        points2d[:, 1] = (points_ndc[:, 1]) * h
                        points2d = points2d.floor().long()
                        mask = valid_x & valid_y & valid_z
                        update_feat = torch.zeros(sq_k_feat.shape, dtype=sq_k_feat.dtype, device=device) # (h*w, c)
                        query_2d = torch.chunk(points2d[mask], 2, dim=-1) # ((M, 1), (M, 1))
                        query_2d = (query_2d[1], query_2d[0])
                        update_feat[mask] = gs_feats[i,k][query_2d[0], query_2d[1], :].squeeze(1) # (h*w, c)
                        sq_j_feat = sq_j_feat + update_feat * self.gamma * torch.sum(mask) / (h * w)
                        norm = norm + self.gamma * torch.sum(mask) / (h * w)

                j_feat = rearrange(sq_j_feat / norm, "(h w) c -> h w c", h=h, w=w)
                view_feat_list.append(j_feat.unsqueeze(0)) # (1, h, w, c)
                
            gs_feats[i] = torch.cat(view_feat_list, dim=0) 
        
        # network
        gs_feats = rearrange(gs_feats, "b v h w c -> (b v) c h w")
        gs_feats = self.net(gs_feats)
        gs_feats = rearrange(gs_feats, "(b v) c h w -> b v h w c", b=b, v=v)       
        
        return gs_feats
