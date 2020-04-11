import torch
import torch.nn as nn


class SIM(nn.Module):
    def __init__(self, h_C, l_C):
        super(SIM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)
        
        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)
        
        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(h_C)
        
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))
        
        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))
        
        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))
        
        return x_h

class conv_2nV1_rgbd(nn.Module):
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super().__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        self.relu = nn.ReLU(True)
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        self.h2h_depth_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_depth_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_depth_0 = nn.BatchNorm2d(mid_c)
        self.bnl_depth_0 = nn.BatchNorm2d(mid_c)        
        
        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        self.h2h_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2l_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2h_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnl_depth_1 = nn.BatchNorm2d(mid_c)
        self.bnh_depth_1 = nn.BatchNorm2d(mid_c)

        self.h2h_d2rgb_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_d2rgb_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)

        self.h2h_rgb2d_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_rgb2d_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        
        if self.main == 0:
            # stage 2
            self.h2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            self.h2h_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2h_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnh_depth_2 = nn.BatchNorm2d(mid_c)
            
            self.h2h_d2rgb_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.h2h_rgb2d_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)

            # stage 3
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)            
            self.identity = nn.Conv2d(in_hc, out_c, 1)

            self.h2h_depth_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_depth_3 = nn.BatchNorm2d(out_c)            
            self.identity_depth = nn.Conv2d(in_hc, out_c, 1)

        elif self.main == 1:
            # stage 2
            self.h2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            self.h2l_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnl_depth_2 = nn.BatchNorm2d(mid_c)

            self.l2l_d2rgb_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_rgb2d_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            
            # stage 3
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)           
            self.identity = nn.Conv2d(in_lc, out_c, 1)
     
            self.l2l_depth_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_depth_3 = nn.BatchNorm2d(out_c)           
            self.identity_depth = nn.Conv2d(in_lc, out_c, 1)        
        else:
            raise NotImplementedError
        self.out_conv = nn.Sequential(
                nn.Conv2d(out_c * 2, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(out_c, out_c, 3, 1, 1),
            )
    
    def forward(self, in_h, in_l, in_h_depth, in_l_depth):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        h_depth = self.relu(self.bnh_depth_0(self.h2h_depth_0(in_h_depth)))
        l_depth = self.relu(self.bnl_depth_0(self.l2l_depth_0(in_l_depth)))
        
        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))

        h2h_depth = self.h2h_depth_1(h_depth)
        h2l_depth = self.h2l_depth_1(self.h2l_pool(h_depth))
        l2l_depth = self.l2l_depth_1(l_depth)
        l2h_depth = self.l2h_depth_1(self.l2h_up(l_depth))

        h2h_d2rgb = self.h2h_d2rgb_1(h_depth)
        l2l_d2rgb = self.l2l_d2rgb_1(l_depth)

        h2h_rgb2d = self.h2h_rgb2d_1(h)
        l2l_rgb2d = self.l2l_rgb2d_1(l)   

        h = self.relu(self.bnh_1(h2h + l2h + h2h_d2rgb))
        l = self.relu(self.bnl_1(l2l + h2l + l2l_d2rgb))

        h_depth = self.relu(self.bnh_depth_1(h2h_depth + l2h_depth + h2h_rgb2d))
        l_depth = self.relu(self.bnl_depth_1(l2l_depth + h2l_depth + l2l_rgb2d))

        if self.main == 0:
            # stage 2
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))

            h2h_depth = self.h2h_depth_2(h_depth)
            l2h_depth = self.l2h_depth_2(self.l2h_up(l_depth))
            
            h2h_d2rgb = self.h2h_d2rgb_2(h_depth)
            h2h_rgb2d = self.h2h_rgb2d_2(h)

            h_fuse = self.relu(self.bnh_2(h2h + l2h + h2h_d2rgb))
            h_fuse_depth = self.relu(self.bnh_depth_2(h2h_depth + l2h_depth + h2h_rgb2d))
            
            # stage 3
            out = self.relu(
                self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
            out_depth = self.relu(
                self.bnh_depth_3(self.h2h_depth_3(h_fuse_depth)) + self.identity_depth(in_h_depth))
            # 这里使用的不是in_h(_depth)，而是h(_depth)
        elif self.main == 1:
            # stage 2
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)

            h2l_depth = self.h2l_depth_2(self.h2l_pool(h_depth))
            l2l_depth = self.l2l_depth_2(l_depth)

            l2l_d2rgb = self.l2l_d2rgb_2(l_depth)
            l2l_rgb2d = self.l2l_rgb2d_2(l)

            l_fuse = self.relu(self.bnl_2(h2l + l2l + l2l_d2rgb))
            l_fuse_depth = self.relu(self.bnl_depth_2(h2l_depth + l2l_depth + l2l_rgb2d))
            
            # stage 3
            out = self.relu(
                self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
            out_depth = self.relu(
                self.bnl_depth_3(self.l2l_depth_3(l_fuse_depth)) + self.identity_depth(in_l_depth))
        else:
            raise NotImplementedError
        out = torch.cat([ out, out_depth ], dim = 1)
        out = self.out_conv(out)
        return out

class conv_3nV1_rgbd(nn.Module):
    def __init__(self, in_hc=64, in_mc=256, in_lc=512, out_c=64):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.AvgPool2d((2, 2), stride=2)
        
        mid_c = min(in_hc, in_mc, in_lc)
        self.relu = nn.ReLU(True)
        
        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        self.h2h_depth_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.m2m_depth_0 = nn.Conv2d(in_mc, mid_c, 3, 1, 1)
        self.l2l_depth_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_depth_0 = nn.BatchNorm2d(mid_c)
        self.bnm_depth_0 = nn.BatchNorm2d(mid_c)
        self.bnl_depth_0 = nn.BatchNorm2d(mid_c)
        
        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        self.h2h_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2m_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2h_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2l_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_depth_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnh_depth_1 = nn.BatchNorm2d(mid_c)
        self.bnm_depth_1 = nn.BatchNorm2d(mid_c)
        self.bnl_depth_1 = nn.BatchNorm2d(mid_c)

        self.h2h_d2rgb_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_d2rgb_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_d2rgb_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)

        self.h2h_rgb2d_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_rgb2d_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_rgb2d_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        
        # stage 2
        self.h2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        self.h2m_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_depth_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm_depth_2 = nn.BatchNorm2d(mid_c)

        self.m2m_d2rgb_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_rgb2d_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        
        # stage 3
        self.m2m_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
        self.bnm_3 = nn.BatchNorm2d(out_c)       
        self.identity = nn.Conv2d(in_mc, out_c, 1)

        self.m2m_depth_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
        self.bnm_depth_3 = nn.BatchNorm2d(out_c)       
        self.identity_depth = nn.Conv2d(in_mc, out_c, 1)
        self.out_conv = nn.Sequential(
                nn.Conv2d(out_c * 2, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(out_c, out_c, 3, 1, 1),
            )
    
    def forward(self, in_h, in_m, in_l, in_h_depth, in_m_depth, in_l_depth):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        h_depth = self.relu(self.bnh_depth_0(self.h2h_depth_0(in_h_depth)))
        m_depth = self.relu(self.bnm_depth_0(self.m2m_depth_0(in_m_depth)))
        l_depth = self.relu(self.bnl_depth_0(self.l2l_depth_0(in_l_depth)))
        
        # stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(self.upsample(m))
        
        h2m = self.h2m_1(self.downsample(h))
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(self.upsample(l))
        
        m2l = self.m2l_1(self.downsample(m))
        l2l = self.l2l_1(l)

        h2h_depth = self.h2h_depth_1(h_depth)
        m2h_depth = self.m2h_depth_1(self.upsample(m_depth))
        
        h2m_depth = self.h2m_depth_1(self.downsample(h_depth))
        m2m_depth = self.m2m_depth_1(m_depth)
        l2m_depth = self.l2m_depth_1(self.upsample(l_depth))
        
        m2l_depth = self.m2l_depth_1(self.downsample(m_depth))
        l2l_depth = self.l2l_depth_1(l_depth)

        h2h_d2rgb = self.h2h_d2rgb_1(h_depth)
        m2m_d2rgb = self.m2m_d2rgb_1(m_depth)
        l2l_d2rgb = self.l2l_d2rgb_1(l_depth)

        h2h_rgb2d = self.h2h_rgb2d_1(h)
        m2m_rgb2d = self.m2m_rgb2d_1(m)
        l2l_rgb2d = self.l2l_rgb2d_1(l)        
        
        h = self.relu(self.bnh_1(h2h + m2h + h2h_d2rgb))
        m = self.relu(self.bnm_1(h2m + m2m + l2m + m2m_d2rgb))
        l = self.relu(self.bnl_1(m2l + l2l + l2l_d2rgb))

        h_depth = self.relu(self.bnh_depth_1(h2h_depth + m2h_depth + h2h_rgb2d))
        m_depth = self.relu(self.bnm_depth_1(h2m_depth + m2m_depth + l2m_depth + m2m_rgb2d))
        l_depth = self.relu(self.bnl_depth_1(m2l_depth + l2l_depth + l2l_rgb2d))
        # stage 2
        h2m = self.h2m_2(self.downsample(h))
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(self.upsample(l))

        h2m_depth = self.h2m_depth_2(self.downsample(h_depth))
        m2m_depth = self.m2m_depth_2(m_depth)
        l2m_depth = self.l2m_depth_2(self.upsample(l_depth))

        m2m_d2rgb = self.m2m_d2rgb_2(m_depth)
        m2m_rgb2d = self.m2m_rgb2d_2(m)

        m = self.relu(self.bnm_2(h2m + m2m + l2m + m2m_d2rgb))

        m_depth = self.relu(self.bnm_depth_2(h2m_depth + m2m_depth + l2m_depth + m2m_rgb2d))
        
        # stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
        out_depth = self.relu(self.bnm_depth_3(self.m2m_depth_3(m_depth)) + self.identity_depth(in_m_depth))
        
        out = torch.cat([ out, out_depth ], dim = 1)
        out = self.out_conv(out)
        return out

class AIMRGBD(nn.Module):
    def __init__(self, iC_list, oC_list):
        super().__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv0 = conv_2nV1_rgbd(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
        self.conv1 = conv_3nV1_rgbd(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
        self.conv2 = conv_3nV1_rgbd(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
        self.conv3 = conv_3nV1_rgbd(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
        self.conv4 = conv_2nV1_rgbd(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)
    
    def forward(self, xs, ds):
        # rgb_data_2, rgb_data_4, rgb_data_8, rgb_data_16, rgb_data_32
        # depth_data_2, depth_data_4, depth_data_8, depth_data_16, depth_data_32
        out_xs = []
        out_xs.append(self.conv0(xs[0], xs[1], ds[0], ds[1]))
        out_xs.append(self.conv1(xs[0], xs[1], xs[2], ds[0], ds[1], ds[2]))
        out_xs.append(self.conv2(xs[1], xs[2], xs[3], ds[1], ds[2], ds[3]))
        out_xs.append(self.conv3(xs[2], xs[3], xs[4], ds[2], ds[3], ds[4]))
        out_xs.append(self.conv4(xs[3], xs[4], ds[3], ds[4]))
        
        return out_xs
