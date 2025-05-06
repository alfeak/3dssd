import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pan_modules

class PANBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_rgb = self.model_cfg.USE_RGB

        channel_in = input_channels if self.use_rgb else input_channels - 3
        channel_out_list = [channel_in - 3]
        self.SA_modules = nn.ModuleList()
        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        
        self.layer_types = self.model_cfg.SA_CONFIG.LAYER_TYPE
        self.ctr_indexes = self.model_cfg.SA_CONFIG.CTR_INDEX
        self.layer_names = self.model_cfg.SA_CONFIG.LAYER_NAME
        self.layer_inputs = self.model_cfg.SA_CONFIG.LAYER_INPUT
        self.max_translate_range = self.model_cfg.SA_CONFIG.MAX_TRANSLATE_RANGE

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()): #[1,64,128,256,256,512]
            channel_in = channel_out_list[self.layer_inputs[k]] # [1] [64] [128]
            if self.layer_types[k] == 'SA_Layer':
                mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k].copy() #[64] [128] [256]
                if mlps.__len__() == 1:
                    mlps = mlps + [-1]
                self.SA_modules.append(
                    pan_modules.PointConv(
                        in_channels=mlps[0],
                        out_channels=mlps[-1],
                        npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                        kneighbors=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                        fps_type=self.model_cfg.SA_CONFIG.FPS_TYPE[k],
                        fps_range=self.model_cfg.SA_CONFIG.FPS_RANGE[k],
                        use_xyz=True,
                    )
                )
            elif self.layer_types[k] == 'Vote_Layer': # [256]
                self.SA_modules.append(pan_modules.Vote_layer(mlp_list=self.model_cfg.SA_CONFIG.MLPS[k], 
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range))

            channel_out_list.append(self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k])
        self.num_point_features = self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[-1]


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        if self.use_rgb:
            features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        else:
            features = (pc[:, 4:5].contiguous() if pc.size(-1) > 4 else None)
        # features = pc[:, 1:].contiguous()
        return batch_idx, xyz, features

    def range_encoded(self, xyz, feature):
        """
        POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
        feature: [b, 4, n] color: feature [:,1:,n]
        xyz: [b, n, 3]
        R = 70.4
        range = torch.norm(xyz, p=2, dim=2)
        color = range/ (R*255) * color
        """
        # 定义最大半径 R
        R = 70.4
        # 计算每个点到原点的欧式距离
        range_ = torch.norm(xyz, p=2, dim=2)  # (b, n)
        # 取原始RGB特征
        color = feature[:, 1:, :]  # (b, 3, n)
        range_scale = range_ / (R*255)
        # 调整维度以便广播相乘
        range_scale = range_scale.unsqueeze(1)  # (b, 1, n)
        # 颜色根据range缩放
        color_scaled = color * range_scale
        # 合并强度和新的颜色
        new_feature = torch.cat([feature[:, 0:1, :], color_scaled], dim=1)  # (b, 7, n)
        return new_feature

    def color_normalize(self, feature):
        """
        对输入特征中的颜色信息进行归一化处理。
        假设颜色部分在 feature 中是第 1:4 个通道（RGB），形状为 (b, 3, n)，范围为 0-255。
        """
        color = feature[:, 1:, :]  # 提取颜色部分，shape: (b, 3, n)

        # 将颜色值从 0-255 缩放到 0-1
        color = color / 255.0

        # 计算每个样本每个通道的均值和标准差
        mean = color.mean(dim=2, keepdim=True)  # shape: (b, 3, 1)
        std = color.std(dim=2, keepdim=True) + 1e-5  # 避免除以 0

        # 标准化
        normalized_color = (color - mean) / std  # shape: (b, 3, n)

        # 替换回原始 feature 中
        feature[:, 1:, :] = normalized_color
        return feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None
        if self.use_rgb:
            features = self.range_encoded(xyz,features)
            # features = self.color_normalize(features)
        # features = self.embedding(features)

        encoder_xyz, encoder_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]  # [0, 1, 2, 3, 4, 3]
            feature_input = encoder_features[self.layer_inputs[i]]
            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = None
                if self.ctr_indexes[i] != -1: # [-1, -1, -1, -1, -1, 5]
                    ctr_xyz = encoder_xyz[self.ctr_indexes[i]]
                    # print(xyz_input.shape,feature_input.shape,ctr_xyz.shape)
                li_xyz, li_features = self.SA_modules[i](xyz_input, feature_input, ctr_xyz=ctr_xyz)
                # print(li_xyz.shape,li_features.shape)
            elif self.layer_types[i] == 'Vote_Layer':
                li_xyz, li_features, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_input
            encoder_xyz.append(li_xyz)
            encoder_features.append(li_features)
        # for idx in range(len(encoder_xyz)):
        #     print(encoder_xyz[idx].shape)
        #     print(encoder_features[idx].shape)
        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :ctr_offsets.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx

        return batch_dict
