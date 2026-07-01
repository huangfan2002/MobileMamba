"""Course-aligned MobileMamba variants.

This module keeps the upstream MobileMamba implementation intact and adds
explicit MRFFI variants that match the course report description:

- WTE-Mamba global branch: MBWTConv2d = SS2D + Haar/DB1 wavelet enhancement.
- MK-DeConv local branch: parallel 3x3 and 5x5 depthwise convolutions.
- Identity branch: untouched residual channels.

The file is placed at ``model/[!_]*.py`` level so ``model.__init__`` auto-imports
it and registers the new model names.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from model import MODEL
from model.mobilemamba.mobilemamba import (
    BN_Linear,
    CFG_MobileMamba_B1,
    CFG_MobileMamba_B2,
    CFG_MobileMamba_B4,
    CFG_MobileMamba_S6,
    CFG_MobileMamba_T2,
    CFG_MobileMamba_T4,
    Conv2d_BN,
    DWConv2d_BN_ReLU,
    FFN,
    MBWTConv2d,
    PatchMerging,
    Residual,
    nearest_multiple_of_16,
    replace_batchnorm,
)
from timm.layers import DropPath


class MultiKernelDWConv2d_BN_ReLU(nn.Module):
    """Efficient MK-DeConv branch used by MRFFI.

    The upstream implementation uses one depthwise kernel per stage. The course
    report describes the local branch as two parallel depthwise paths, 3x3 and
    5x5, followed by channel concatenation. This class implements that layout
    while preserving the original Conv-BN-ReLU utility and BN fusion behavior.
    """

    def __init__(self, channels, kernels=(3, 5), bn_weight_init=1):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive for MK-DeConv")
        if len(kernels) != 2:
            raise ValueError("MRFFI local branch expects exactly two kernels")

        first = channels // 2
        second = channels - first
        self.split_sizes = (first, second)

        self.branch3 = (
            DWConv2d_BN_ReLU(first, first, kernel_size=kernels[0], bn_weight_init=bn_weight_init)
            if first > 0
            else nn.Identity()
        )
        self.branch5 = DWConv2d_BN_ReLU(second, second, kernel_size=kernels[1], bn_weight_init=bn_weight_init)

    def forward(self, x):
        x3, x5 = torch.split(x, self.split_sizes, dim=1)
        return torch.cat([self.branch3(x3), self.branch5(x5)], dim=1)


class MRFFIModule(nn.Module):
    """Multi-Receptive Field Feature Interaction module.

    Input feature maps are split along channels into three complementary paths:

    1. global-frequency path: WTE-Mamba via MBWTConv2d;
    2. local multi-scale path: 3x3/5x5 MK-DeConv;
    3. identity path: raw feature preservation.
    """

    def __init__(
        self,
        dim,
        global_ratio=0.8,
        local_ratio=0.2,
        global_kernel=5,
        local_kernels=(3, 5),
        ssm_ratio=1,
        forward_type="v052d",
    ):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        self.local_channels = int(local_ratio * dim)
        if self.global_channels + self.local_channels > dim:
            self.local_channels = max(dim - self.global_channels, 0)
        self.identity_channels = dim - self.global_channels - self.local_channels
        if self.identity_channels < 0:
            raise ValueError(
                f"invalid MRFFI split: dim={dim}, global={self.global_channels}, local={self.local_channels}"
            )

        self.global_op = (
            MBWTConv2d(
                self.global_channels,
                self.global_channels,
                kernel_size=global_kernel,
                wt_levels=1,
                wt_type="db1",
                ssm_ratio=ssm_ratio,
                forward_type=forward_type,
            )
            if self.global_channels > 0
            else nn.Identity()
        )
        self.local_op = (
            MultiKernelDWConv2d_BN_ReLU(self.local_channels, kernels=local_kernels)
            if self.local_channels > 0
            else nn.Identity()
        )
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def extra_repr(self):
        return (
            f"dim={self.dim}, global_channels={self.global_channels}, "
            f"local_channels={self.local_channels}, identity_channels={self.identity_channels}"
        )

    def forward(self, x):
        x_g, x_l, x_i = torch.split(
            x,
            [self.global_channels, self.local_channels, self.identity_channels],
            dim=1,
        )
        x_g = self.global_op(x_g)
        x_l = self.local_op(x_l)
        return self.proj(torch.cat([x_g, x_l, x_i], dim=1))


class MRFFIBlockWindow(nn.Module):
    def __init__(
        self,
        dim,
        global_ratio=0.8,
        local_ratio=0.2,
        kernels=5,
        ssm_ratio=1,
        forward_type="v052d",
    ):
        super().__init__()
        self.attn = MRFFIModule(
            dim,
            global_ratio=global_ratio,
            local_ratio=local_ratio,
            global_kernel=kernels,
            local_kernels=(3, 5),
            ssm_ratio=ssm_ratio,
            forward_type=forward_type,
        )

    def forward(self, x):
        return self.attn(x)


class MRFFIBlock(nn.Module):
    def __init__(
        self,
        block_type,
        ed,
        global_ratio=0.8,
        local_ratio=0.2,
        kernels=5,
        drop_path=0.0,
        has_skip=True,
        ssm_ratio=1,
        forward_type="v052d",
    ):
        super().__init__()
        if block_type != "s":
            raise ValueError(f"unsupported MobileMamba block type: {block_type}")

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.0))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))
        self.mixer = Residual(
            MRFFIBlockWindow(
                ed,
                global_ratio=global_ratio,
                local_ratio=local_ratio,
                kernels=kernels,
                ssm_ratio=ssm_ratio,
                forward_type=forward_type,
            )
        )
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.0))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))
        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        return shortcut + self.drop_path(x) if self.has_skip else x


class CourseMRFFIMobileMamba(nn.Module):
    """MobileMamba backbone using explicit MRFFI blocks."""

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        stages=("s", "s", "s"),
        embed_dim=(192, 384, 448),
        global_ratio=(0.8, 0.7, 0.6),
        local_ratio=(0.2, 0.2, 0.3),
        depth=(1, 2, 2),
        kernels=(7, 5, 3),
        down_ops=(("subsample", 2), ("subsample", 2), ("",)),
        distillation=False,
        drop_path=0.0,
        ssm_ratio=1,
        forward_type="v052d",
    ):
        super().__init__()
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1),
        )

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        for i, (stg, ed, dpth, gr, lr, do) in enumerate(
            zip(stages, embed_dim, depth, global_ratio, local_ratio, down_ops)
        ):
            dpr = dprs[sum(depth[:i]) : sum(depth[: i + 1])]
            for d in range(dpth):
                eval("self.blocks" + str(i + 1)).append(
                    MRFFIBlock(
                        stg,
                        ed,
                        global_ratio=gr,
                        local_ratio=lr,
                        kernels=kernels[i],
                        drop_path=dpr[d],
                        ssm_ratio=ssm_ratio,
                        forward_type=forward_type,
                    )
                )
            if do[0] == "subsample":
                next_stage = eval("self.blocks" + str(i + 2))
                next_stage.append(
                    nn.Sequential(
                        Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                        Residual(FFN(embed_dim[i], int(embed_dim[i] * 2))),
                    )
                )
                next_stage.append(PatchMerging(*embed_dim[i : i + 2]))
                next_stage.append(
                    nn.Sequential(
                        Residual(
                            Conv2d_BN(
                                embed_dim[i + 1],
                                embed_dim[i + 1],
                                3,
                                1,
                                1,
                                groups=embed_dim[i + 1],
                            )
                        ),
                        Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2))),
                    )
                )

        self.blocks1 = nn.Sequential(*self.blocks1)
        self.blocks2 = nn.Sequential(*self.blocks2)
        self.blocks3 = nn.Sequential(*self.blocks3)
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if "attention_biases" in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


def _build_course_model(model_cfg, num_classes=1000, distillation=False, fuse=False):
    cfg = deepcopy(model_cfg)
    model = CourseMRFFIMobileMamba(num_classes=num_classes, distillation=distillation, **cfg)
    if fuse:
        replace_batchnorm(model)
    return model


@MODEL.register_module
def MobileMamba_T2_MRFFI(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    return _build_course_model(CFG_MobileMamba_T2, num_classes=num_classes, distillation=distillation, fuse=fuse)


@MODEL.register_module
def MobileMamba_T4_MRFFI(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    return _build_course_model(CFG_MobileMamba_T4, num_classes=num_classes, distillation=distillation, fuse=fuse)


@MODEL.register_module
def MobileMamba_S6_MRFFI(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    return _build_course_model(CFG_MobileMamba_S6, num_classes=num_classes, distillation=distillation, fuse=fuse)


@MODEL.register_module
def MobileMamba_B1_MRFFI(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    return _build_course_model(CFG_MobileMamba_B1, num_classes=num_classes, distillation=distillation, fuse=fuse)


@MODEL.register_module
def MobileMamba_B2_MRFFI(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    return _build_course_model(CFG_MobileMamba_B2, num_classes=num_classes, distillation=distillation, fuse=fuse)


@MODEL.register_module
def MobileMamba_B4_MRFFI(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    return _build_course_model(CFG_MobileMamba_B4, num_classes=num_classes, distillation=distillation, fuse=fuse)
