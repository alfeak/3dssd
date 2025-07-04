from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
# from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
# from .spconv_unet import UNetV2
from .SSD_backbone import SSDBackbone
from .pan_backbone import PANBackbone
__all__ = {
    # 'VoxelBackBone8x': VoxelBackBone8x,
    # 'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    # 'VoxelResBackBone8x': VoxelResBackBone8x,
    'SSDBackbone': SSDBackbone,
    'PANBackbone': PANBackbone,
}
