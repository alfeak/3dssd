from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
import pcdet.utils.SSD as SSD
import pcdet.ops.pointnet2.pointnet2_3DSSD.pointnet2_utils as pointnet2_3DSSD

#         :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
#         :param features: (B, N, C) tensor of the descriptors of the the features
#         :param new_xyz:
#         :return:
#             new_xyz: (B, npoint, 3) tensor of the new features' xyz
#             new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
#         """

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    # sqrdists = SSD.calc_square_dist(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return group_idx

class PointNorm(nn.Module):
    def __init__(self,in_channels,**kwargs):
        super(PointNorm, self).__init__()
        self.in_channels = in_channels
        self.alpha = nn.Parameter(torch.ones(1,in_channels,1,1))
        self.beta = nn.Parameter(torch.zeros(1,in_channels,1,1))
        self.eps = 1e-6

    def __repr__(self):
        return f"PointNorm(in_channels={self.in_channels})"

    def forward(self, points):
        anchor = points[:,:,:,0].unsqueeze(-1)
        std = torch.std(points - anchor, dim=1, keepdim=True, unbiased=False)
        points = (points - anchor) / (std + self.eps)

        return points * self.alpha + self.beta

class maxpool(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.pool = nn.MaxPool2d((1,k))
    def forward(self,x):
        return self.pool(x).squeeze(-1)

class avgpool(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.pool = nn.AvgPool2d((1,k))
    def forward(self,x):
        return self.pool(x).squeeze(-1)

class set_conv(nn.Module):
    def __init__(self, in_channels, out_channels,nsample):
        super().__init__()
        self.conv = nn.Sequential(
            # PointNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            maxpool(nsample),
        )
        self.pool = maxpool(nsample)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        return self.act(self.conv1(x) + self.pool(x))

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, npoint, kneighbors=32,fps_type='F-FPS',fps_range=-1,use_xyz=True,**kwargs):
        super(PointConv, self).__init__()
        self.fps_types = fps_type
        self.fps_ranges = fps_range
        self.use_xyz = use_xyz
        self.in_channels = in_channels + 3 if use_xyz else in_channels
        self.out_channels = out_channels
        self.npoint = npoint
        # self.radius = radius
        self.kneighbors = kneighbors

        if out_channels != -1:
            self.convs = nn.Sequential(
            set_conv(self.in_channels ,out_channels, kneighbors),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels else nn.Identity()

            self.act = nn.ReLU(inplace=True)

    def points_sorted(self,grouped_xyz: torch.Tensor = None,grouped_features: torch.Tensor = None, query_xyz: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        # decenter
        grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        # normalize
        # grouped_xyz /= self.radius
        distance = torch.norm(grouped_xyz, dim=1)  # (B, npoint, nsample)
        index = torch.argsort(distance, dim=-1)  # (B, npoint, nsample)
        grouped_xyz = torch.gather(grouped_xyz, dim=-1, index=index.unsqueeze(1).repeat(1,3,1,1))  # (B, npoint, nsample, 3)
        grouped_features = torch.gather(grouped_features, dim=-1, index=index.unsqueeze(1).repeat(1,grouped_features.shape[1],1,1))
        return grouped_xyz, grouped_features

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, ctr_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        B, N, C = xyz.shape
        xyz = xyz.contiguous()
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        features = features.contiguous() if features is not None else None
        last_fps_end_index = 0
        fps_idxes = []
        if ctr_xyz is not None:
            new_xyz = ctr_xyz
        else:  
            for i in range(len(self.fps_types)):
                fps_type = self.fps_types[i]
                fps_range = self.fps_ranges[i]
                npoint = self.npoint[i]
                if npoint == 0:
                    continue
                if fps_range == -1:
                    xyz_tmp = xyz[:, last_fps_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
                else:
                    xyz_tmp = xyz[:, last_fps_end_index:fps_range, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:fps_range, :]
                    last_fps_end_index += fps_range
                if fps_type == 'D-FPS':
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                elif fps_type == 'F-FPS':
                    # features_SSD = xyz_tmp
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                    # features_for_fps_distance = square_distance(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                elif fps_type == 'FS':
                    # features_SSD = xyz_tmp
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                    # features_for_fps_distance = square_distance(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx_1 = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    fps_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)  # [bs, npoint * 2]
                fps_idxes.append(fps_idx)
            fps_idxes = torch.cat(fps_idxes, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, fps_idxes).transpose(1, 2).contiguous() if self.npoint is not None else None

        if self.out_channels != -1:
            idx = knn_point(self.kneighbors, xyz, new_xyz).to(torch.int32)
            # idx = pointnet2_utils.ball_query(self.radius, self.kneighbors, xyz, new_xyz).to(torch.int32)
            grouped_xyz = pointnet2_utils.grouping_operation(xyz.transpose(1, 2).contiguous(), idx)
            grouped_features = pointnet2_utils.grouping_operation(features, idx)
            # grouped_xyz, grouped_features = self.points_sorted(grouped_xyz,grouped_features,new_xyz)
            grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

            # print(grouped_xyz.shape, grouped_features.shape, new_xyz.shape, sampled_features.shape)
            sampled_features = grouped_features[:,:,:,0]

            if self.use_xyz:
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=1)

            new_features = self.act(self.convs(grouped_features) + self.skip_conv(sampled_features))
        else:
            new_features = pointnet2_utils.gather_operation(features, fps_idxes).contiguous()
        return new_xyz, new_features


class Vote_layer(nn.Module):
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        shared_mlps = []
        for i in range(len(mlp_list)):
            shared_mlps.extend([
                nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_list[i]),
                nn.ReLU()
            ])
            pre_channel = mlp_list[i]
        self.mlp_modules = nn.Sequential(*shared_mlps)

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.min_offset = torch.tensor(max_translate_range).float().view(1, 1, 3)

    def forward(self, xyz, features):

        new_features = self.mlp_modules(features)
        ctr_offsets = self.ctr_reg(new_features)

        ctr_offsets = ctr_offsets.transpose(1, 2)

        min_offset = self.min_offset.repeat((xyz.shape[0], xyz.shape[1], 1)).to(xyz.device)

        limited_ctr_offsets = torch.where(ctr_offsets < min_offset, min_offset, ctr_offsets)
        min_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(limited_ctr_offsets > min_offset, min_offset, limited_ctr_offsets)
        xyz = xyz + limited_ctr_offsets
        return xyz, new_features, ctr_offsets


if __name__ == "__main__":
    # pass
    # 1th layer input 16384,3 3,16384
    layer = PointConv(3,64,[4096],radius=0.2,kneighbors=64,fps_type=['D-FPS'],fps_range=[-1]).cuda()
    data = torch.randn(3, 16384, 3).cuda()
    feature = torch.randn(3, 3, 16384).cuda()
    new_xyz, new_feature = layer(data, feature,None)
    print(new_xyz.shape, new_feature.shape)

    # # 2th layer input 4093,3 64,4096
    # layer = PointConv(64,128,[512],24,fps_type=['FS'],fps_range=[-1]).cuda()
    # data = torch.randn(3, 4096, 3).cuda()
    # feature = torch.randn(3, 64, 4096).cuda()
    # new_xyz, new_feature = layer(data, feature,None)
    # print(new_xyz.shape, new_feature.shape)

    # # 3th layer input 1024,3 128,1024
    # layer = PointConv(128,256,[256, 256],24,fps_type=['F-FPS', 'D-FPS'],fps_range=[512, -1]).cuda()
    # data = torch.randn(3, 1024, 3).cuda()
    # feature = torch.randn(3, 128, 1024).cuda()
    # new_xyz, new_feature = layer(data, feature,None)
    # print(new_xyz.shape, new_feature.shape)

    # 4th layer input 512,3 256,512
    layer = PointConv(256,-1,[256, 0],24,fps_type=['F-FPS', 'D-FPS'],fps_range=[256, -1]).cuda()
    data = torch.randn(3, 512, 3).cuda()
    feature = torch.randn(3, 256, 512).cuda()
    new_xyz, new_feature = layer(data, feature,None)
    print(new_xyz.shape, new_feature.shape)
    
    # # 6th layer input 512,3 256,512 256,3
    # layer = PointConv(256,256,[256],24,fps_type=['D-FPS'],fps_range=[-1]).cuda()
    # data = torch.randn(3, 512, 3).cuda()
    # feature = torch.randn(3, 256, 512).cuda()
    # ctx_xyz = torch.randn(3, 256, 3).cuda()
    # new_xyz, new_feature = layer(data, feature,ctx_xyz)
    # print(new_xyz.shape, new_feature.shape)