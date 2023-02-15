import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

import pointnet2_utils
from point_4d_convolution import *
from transformer import *

class PrimitiveTransformer(nn.Module):
    def __init__(self, radius=0.9, nsamples=32, num_classes=49):
        super(PrimitiveTransformer, self).__init__()

        self.conv1 = P4DConv(in_planes=3,
                             mlp_planes=[32,64,128],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.conv2 = P4DConv(in_planes=128,
                             mlp_planes=[128, 128, 256],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.conv3 = P4DConv(in_planes=256,
                             mlp_planes=[256,256,512],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*radius, nsamples],
                             temporal_kernel_size=3,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[1,1])

        self.conv4 = P4DConv(in_planes=512,
                             mlp_planes=[512,512,1024],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*2*radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=2,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.deconv4 = P4DTransConv(in_planes=1024,
                                    mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=512)

        self.deconv3 = P4DTransConv(in_planes=256,
                                    mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=256)

        self.deconv2 = P4DTransConv(in_planes=256,
                                    mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=128)

        self.deconv1 = P4DTransConv(in_planes=256,
                                    mlp_planes=[512, 1024],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=3)

        self.deconv_cat = P4DTransConv(in_planes=1024,
                                    mlp_planes=[1024, 1024],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=1024)

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = Transformer(dim=1024, depth=2, heads=4, dim_head=256, mlp_dim=1024)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = Transformer(dim=1024, depth=2, heads=4, dim_head=256, mlp_dim=1024)
        self.outconv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.outconv2 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.outconv3 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, xyzs, rgbs):
        coors = xyzs.clone()

        new_xyzs1, new_features1 = self.conv1(xyzs, rgbs)

        new_xyzs2, new_features2 = self.conv2(new_xyzs1, new_features1)

        new_xyzs3, new_features3 = self.conv3(new_xyzs2, new_features2)

        new_xyzs4, new_features4 = self.conv4(new_xyzs3, new_features3)

        new_xyzsd4, new_featuresd4 = self.deconv4(new_xyzs4, new_xyzs3, new_features4, new_features3)

        new_xyzsd3, new_featuresd3 = self.deconv3(new_xyzsd4, new_xyzs2, new_featuresd4, new_features2)

        new_xyzsd2, new_featuresd2 = self.deconv2(new_xyzsd3, new_xyzs1, new_featuresd3, new_features1)

        new_xyzsd1, new_featuresd1 = self.deconv1(new_xyzsd2, xyzs, new_featuresd2, rgbs)

        raw_feat = new_featuresd1.transpose(2, 3).contiguous()      #  B ,L , N, C

        B, L, N, C = raw_feat.size()

        # raw_feat = xyzs

        # print(raw_feat.shape)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        raw_feats = torch.split(tensor=raw_feat, split_size_or_sections=1, dim=1)
        raw_feats = [torch.squeeze(input=raw_feat, dim=1).contiguous() for raw_feat in raw_feats]

        fea_ball = []
        anchor = []
        for t in range(len(xyzs)):
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], 64)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                                                             # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()
            anchor.append(anchor_xyz)

            idx = pointnet2_utils.ball_query(0.9, 64, xyzs[t], anchor_xyz)
            # unique_idx = torch.unique(idx)
            # print(unique_idx.shape)

            neighbor_fea_flipped = raw_feats[t].transpose(1, 2).contiguous()                                                    # (B, 3, N)
            neighbor_fea_grouped = pointnet2_utils.grouping_operation(neighbor_fea_flipped, idx).permute(0,2,3,1)
            fea_ball.append(neighbor_fea_grouped)
        fea_ball = torch.stack(fea_ball,dim=1)
        anchor = torch.stack(anchor,dim=1)

        point_feat = torch.reshape(input=fea_ball, shape=(B*L*64, -1, C))  # [B*L*200, n', C]
        point_feat = self.emb_relu1(point_feat)
        point_feat = self.transformer1(point_feat)  # [B*L*200, n', C]
        # point_feat_backup = torch.reshape(input=point_feat, shape=(B, L, N, C))  # [B, L, N, C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))    # B*l*200, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L*64, C))  # [B, L*200, C]
        primitive_feature = self.emb_relu2(primitive_feature)
        primitive_feature = self.transformer2(primitive_feature)   #B. L*200, C
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L, 64, C))  # [B, L, 200, C]

        # primitive_feature = torch.repeat_interleave(primitive_feature, int(N/64), dim=2)  # B, L , N, C

        _, fused_feature = self.deconv_cat(anchor, coors, primitive_feature.permute(0,1,3,2), raw_feat.permute(0,1,3,2))


        # fused_feature = torch.cat((raw_feat, point_feat_backup, primitive_feature),dim=3)
        # fused_feature = fused_feature.transpose(2, 3).contiguous()  # B L C*3 N
        fused_feature = self.outconv1(fused_feature.transpose(1, 2).contiguous())
        fused_feature = self.outconv2(fused_feature)
        out = self.outconv3(fused_feature).transpose(1, 2).contiguous()
        return out
