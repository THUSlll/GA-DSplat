# Copyright (c) 2024 Haofei Xu
# Copyright (c) 2025 Bo Liu
# Licensed under the MIT License

from dataclasses import dataclass
import imp
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, GaussianProject, GaussianUnproject
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead

from .GGN.gaussian_graph import GGNLinearLayer, GGNPoolingLayer, GaussianGraph, compute_adjacency_tilde
from .triplane.triplane_transformerV4 import ImageTriplaneGenerator, save_xyz_tensor_as_ply
import pdb
import math

from .common.typing import *
ValidScale = Union[Tuple[float, float], Tuple[Tensor, Tensor]]
def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    
    # 确保inp_scale和tgt_scale能够正确广播到dat的维度
    if isinstance(inp_scale[0], Tensor):
        # 如果inp_scale是tensor，需要确保维度匹配
        # 将inp_scale的维度扩展以匹配dat的维度
        inp_min = inp_scale[0].view(*inp_scale[0].shape, *([1] * (dat.dim() - inp_scale[0].dim())))
        inp_max = inp_scale[1].view(*inp_scale[1].shape, *([1] * (dat.dim() - inp_scale[1].dim())))
    else:
        inp_min = inp_scale[0]
        inp_max = inp_scale[1]
    
    if isinstance(tgt_scale[0], Tensor):
        # 如果tgt_scale是tensor，需要确保维度匹配
        tgt_min = tgt_scale[0].view(*tgt_scale[0].shape, *([1] * (dat.dim() - tgt_scale[0].dim())))
        tgt_max = tgt_scale[1].view(*tgt_scale[1].shape, *([1] * (dat.dim() - tgt_scale[1].dim())))
    else:
        tgt_min = tgt_scale[0]
        tgt_max = tgt_scale[1]

    dat = (dat - inp_min) / (inp_max - inp_min)
    dat = dat * (tgt_max - tgt_min) + tgt_min
    return dat

@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int

class TriplaneFeatureFusion(nn.Module):
    def __init__(self, dim_A=132, dim_B=384, hidden_dim=256, initial_gate_value=0.1):
        """
        门控融合模块，用于融合来自像素的特征 (feat_A) 和来自三平面的全局特征 (feat_B)。
        
        Args:
            dim_A (int): 像素特征 feat_A 的维度。
            dim_B (int): 三平面特征 feat_B 的维度。
            hidden_dim (int): 门控网络中隐藏层的维度。
            initial_gate_value (float): 门控网络输出的初始期望值 (建议 0.1-0.2)。
        """
        super().__init__()
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.hidden_dim = hidden_dim
        self.initial_gate_value = initial_gate_value

        # 将 triplane 特征映射到与高斯特征相同的维度
        self.proj = nn.Linear(dim_B, dim_A)

        # 门控网络：决定 triplane 特征的贡献程度
        # 输入是拼接后的特征，输出是 [0, 1] 范围内的门控权重
        self.gate = nn.Sequential(
            nn.Linear(dim_A + dim_B, hidden_dim),
            nn.ReLU(inplace=True),  # 使用 inplace=True 节省内存
            nn.Linear(hidden_dim, dim_A)
            # 注意：这里没有在 Sequential 中包含 Sigmoid
        )
        
        # --- 关键修改：初始化门控网络 ---
        # 目标：让门控网络在训练初期输出一个较小的值 (如 0.1)，以保护预训练模型。
        # 方法：初始化最后一层线性层的偏置 (bias)，使得在输入为0时，输出接近 log(initial_gate_value / (1 - initial_gate_value))
        # 这样，经过 Sigmoid 后，输出就接近 initial_gate_value。
        gate_logit_init = math.log(initial_gate_value / (1 - initial_gate_value))  # 计算对应的logit
        with torch.no_grad():
            # 初始化最后一层线性层的偏置
            self.gate[-1].bias.fill_(gate_logit_init)
            # 初始化最后一层线性层的权重为小的随机值 (如 Xavier 初始化)
            # 这比初始化为0更好，可以打破对称性，让网络从一开始就具备学习能力
            nn.init.xavier_uniform_(self.gate[-1].weight, gain=nn.init.calculate_gain('relu'))

            

    def forward(self, feat_A: torch.Tensor, feat_B: torch.Tensor):
        """
        执行门控融合。
        融合策略: fused = (1 - gate) * feat_A + gate * (W * feat_B)
        这是一种门控的残差连接，其中三平面特征作为"残差"被注入。

        Args:
            feat_A: 原始高斯特征，来自预训练模型。形状 [N, dim_A] 或 [B, N, dim_A]
            feat_B: 三平面特征，提供全局上下文。形状 [N, dim_B] 或 [B, N, dim_B]

        Returns:
            fused: 融合后的增强特征，形状 [N, dim_A] 或 [B, N, dim_A]
        """

        feat_B_mean = feat_B.mean(dim=-1, keepdim=True)
        feat_B_std = feat_B.std(dim=-1, keepdim=True) + 1e-8
        feat_A_mean = feat_A.mean(dim=-1, keepdim=True)
        feat_A_std = feat_A.std(dim=-1, keepdim=True) + 1e-8
        
        # 将feat_B投影到feat_A的分布
        feat_B_proj = self.proj(feat_B)
        feat_B_proj = (feat_B_proj - feat_B_mean) / feat_B_std * feat_A_std + feat_A_mean
        
        # 生成门控信号
        gate_input = torch.cat([feat_A, feat_B], dim=-1)  # [..., dim_A + dim_B]
        # 先通过线性层，再应用 Sigmoid
        # 这样可以确保我们能精确控制偏置的初始化
        gate_logits = self.gate(gate_input)  # [..., dim_A]
        gate = torch.sigmoid(gate_logits)   # [..., dim_A], 值在 [0, 1] 之间

        # 门控残差融合
        # 三平面特征以残差的形式被注入，初始贡献很小，随着训练逐渐增加
        fused = (1 - gate) * feat_A + gate * feat_B_proj

        return fused, gate.mean()

    
class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )
        fuse_channels = in_channels + 64 * 3
        self.triplane_fuse = nn.Sequential(
                nn.Conv2d(fuse_channels, in_channels,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(in_channels,
                          in_channels, 3, 1, 1, padding_mode='replicate')
            )
        # self.ggnLinear = GGNLinearLayer(132, 132, 256, 256)

        self.ggnPooling = GGNPoolingLayer(0.05)

        self.gaussian_unproject = GaussianUnproject()
        
        self.gaussian_project = GaussianProject()

        self.ggn = GaussianGraph(132, 132)

        self.triplane_generator = ImageTriplaneGenerator(feature_channels=64)  

        # self.triplane_fuse = TriplaneFeatureFusion(132, 3 * 64, 256)

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")
        # [B, V, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        
        num_depths = len(depth_preds)

        if self.cfg.supervise_intermediate_depth and num_depths > 1:
            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            densities = torch.cat([densities] * num_depths, dim=0)
            out = torch.cat(
                [out] * num_depths, dim=0)

            b *= num_depths

            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)
            
        else:
            context_extrinsics = context["extrinsics"]
            context_intrinsics = context["intrinsics"]          

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").unsqueeze(0).unsqueeze(0)

        xy_ray = xy_ray.expand(b, v, *xy_ray.shape[2:])

        means = self.gaussian_unproject.forward(
            rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
            rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,                
        )
        xy = self.gaussian_project.forward(
                means,
                rearrange(context_extrinsics,
                        "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics,
                        "b v i j -> b v () () () i j")
        )
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy = rearrange(
            xy,
            "b v0 v1 hw 1 1 c -> b v0 v1 hw c"
        ) / pixel_size
        xy = xy.int()

        feat = rearrange(features, "(b v) c h w -> b v c h w", b = b, v = v)

        means = rearrange(means, "b v (h w) 1 1 c -> b v h w c", h = h, w = w)

        trip, scene_bounds = self.triplane_generator(feat, means, None)
 
        trip_feat = self.query_triplane(rearrange(means, "b v h w c -> b (v h w) c"), trip, torch.tensor(scene_bounds, device=device))
   
  
        fuse_feat = torch.cat([rearrange(trip_feat, "b (v h w) c -> (b v) c h w", v=v, h = h, w=w), out], dim=1)

        feat = self.triplane_fuse(fuse_feat)

        feat = self.ggn(means, depths, rearrange(feat, "(b v) c h w -> b v h w c", v=v, h = h, w=w), context_intrinsics, context_extrinsics)

        gaussians = self.gaussian_head(rearrange(feat, "b v h w c -> (b v) c h w", v=v, h = h, w=w))  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")
        
        pooling_mask = None

        A, _ = compute_adjacency_tilde(context_extrinsics, 3)

        pooling_feat, pooling_means, pooling_mask = self.ggnPooling(raw_gaussians, means, xy, A)

        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()

        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)
            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                pooling_mask,
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                pooling_mask,
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        # save_xyz_tensor_as_ply(gaussians.means[0].detach().cpu(), gaussians.harmonics[..., 0][0].detach().cpu(), "gaussians.ply")

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths,
                "gate": -1
            }

        return {
                "gaussians": gaussians,
                "gate": -1
            }

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim
    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
        scene_bounds: Tensor = None
    ) -> Tensor:

        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        #radius = positions.abs().max()
        # pdb.set_trace()
        if scene_bounds is None:
            positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))
        else:
            x_min = scene_bounds[:, 0]  # shape: [b]
            x_max = scene_bounds[:, 1]  # shape: [b]
            y_min = scene_bounds[:, 2]  # shape: [b]
            y_max = scene_bounds[:, 3]  # shape: [b]
            z_min = scene_bounds[:, 4]  # shape: [b]
            z_max = scene_bounds[:, 5]  # shape: [b]
            positions_x = scale_tensor(positions[:, :, 0], (x_min, x_max), (-1.0, 1.0))
            positions_y = scale_tensor(positions[:, :, 1], (y_min, y_max), (-1.0, 1.0))
            positions_z = scale_tensor(positions[:, :, 2], (z_min, z_max), (-1.0, 1.0))

        positions = torch.cat([positions_x.unsqueeze(2), positions_y.unsqueeze(2), positions_z.unsqueeze(2)], dim=-1)

        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
                (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
                dim=-3,
            ).float()

        out: Float[Tensor, "B3 Cp 1 N"] = nn.functional.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=True,
            mode="bilinear",
        )
        out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)

        if not batched:
            out = out.squeeze(0)

        return out       
    @property
    def sampler(self):
        return None
