import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pan_modules

class PANBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # channel_in = input_channels
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        channel_out_list = [32]
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
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        # features = pc[:, 1:].contiguous()
        return batch_idx, xyz, features

    def pc_normalize(self, pc):
        """
        Normalize point cloud to unit sphere (zero mean, max radius 1)
        Args:
            pc: torch.Tensor of shape [B, 3, N] (batch of point clouds)
        Returns:
            Normalized point cloud with same shape
        """
        # Calculate centroid (mean along N dimension)
        centroid = torch.mean(pc, dim=2, keepdim=True)  # [B, 3, 1]
        # Center the point cloud
        pc = pc - centroid
        # Calculate max radius (norm of each point)
        # We only normalize using XYZ coordinates (ignore 4th dimension if it exists)
        norms = torch.norm(pc, p=2, dim=1)  # [B, N]
        max_radius = torch.max(norms, dim=1, keepdim=True)[0].unsqueeze(1)  # [B, 1, 1]
        # Avoid division by zero
        max_radius = torch.clamp(max_radius, min=1e-8)
        # Normalize all dimensions (including 4th if it exists)
        pc = pc / max_radius
        return pc

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
        features = torch.cat([self.pc_normalize(xyz.permute(0,2,1)),features],dim=1)
        features = self.embedding(features)

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
