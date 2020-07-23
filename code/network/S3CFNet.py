import torch
import torch.nn as nn

from base.BaseBlocks import BasicConv2d
from base.BaseOps import cus_sample, upsample_add
from base.ResNet import Backbone_ResNet50_in3
from base.VGG import Backbone_VGG16_in3
from models.MyModule import (AIMRGBD, SIM)
from .Decoder import Decoder
from .Squeezer import Squeezer

class S3CFNet_Res50(nn.Module):
    def __init__(self, inference_study = False):
        super().__init__()
        self.inference_study = inference_study
        self.rgb_div_2, self.rgb_div_4, self.rgb_div_8, self.rgb_div_16, self.rgb_div_32 = Backbone_ResNet50_in3()
        self.depth_div_2, self.depth_div_4, self.depth_div_8, self.depth_div_16, self.depth_div_32 = Backbone_ResNet50_in3()
        
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        
        self.trans = AIMRGBD(iC_list=(64, 256, 512, 1024, 2048),
                         oC_list=(64, 64, 64, 64, 64))
        
        self.sim32 = SIM(64 * 2, 32)
        self.sim16 = SIM(64 * 2, 32)
        self.sim8 = SIM(64 * 2, 32)
        self.sim4 = SIM(64 * 2, 32)
        self.sim2 = SIM(64 * 2, 32)
        
        self.upconv32 = BasicConv2d(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64 * 2, 32 * 2, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32 * 2, 32 * 2, kernel_size=3, stride=1, padding=1)
        
        self.classifier = nn.Conv2d(32 * 2, 1, 1)
        self.rotation_classifier = Decoder(64 * 2, 8, 4)
        self.squeezer = Squeezer()

        self.grad = None

    def forward(self, rgb, depth):
        rgb_data_2 = self.rgb_div_2(rgb)
        rgb_data_4 = self.rgb_div_4(rgb_data_2)
        rgb_data_8 = self.rgb_div_8(rgb_data_4)
        rgb_data_16 = self.rgb_div_16(rgb_data_8)
        rgb_data_32 = self.rgb_div_32(rgb_data_16)
        xs = [ rgb_data_2, rgb_data_4, rgb_data_8, rgb_data_16, rgb_data_32 ]
        
        depth = torch.cat([ depth ] * 3, dim = 1)
        depth_data_2 = self.depth_div_2(depth)
        depth_data_4 = self.depth_div_4(depth_data_2)
        depth_data_8 = self.depth_div_8(depth_data_4)
        depth_data_16 = self.depth_div_16(depth_data_8)
        depth_data_32 = self.depth_div_32(depth_data_16)
        ds = [ depth_data_2, depth_data_4, depth_data_8, depth_data_16, depth_data_32 ]

        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 \
        = self.trans(xs, ds)

        out_data_32 = self.upconv32(self.sim32(in_data_32) + in_data_32)  # 1024
        
        out_data_16 = self.upsample_add(out_data_32, in_data_16)  # 1024
        out_data_16 = self.upconv16(self.sim16(out_data_16) + out_data_16)
        
        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8) + out_data_8)  # 512
        
        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4) + out_data_4)  # 256
        
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2) + out_data_2)  # 64
        
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        out_data = self.classifier(out_data_1) # resolution identity

        def extract(g):
            self.grad = g
        # in_data_32.register_hook(extract)

        if not self.inference_study:
            return out_data.sigmoid(), self.rotation_classifier(in_data_32)
        else:
            return (out_data.sigmoid(), self.rotation_classifier(in_data_32), in_data_32) + self.squeezer(in_data_2)

    def get_grad(self):
        return self.grad