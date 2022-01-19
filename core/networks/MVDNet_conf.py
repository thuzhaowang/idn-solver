import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from core.networks.submodule import *
from core.utils.inverse_warp_d import inverse_warp_d, pixel2cam
from core.utils.inverse_warp import inverse_warp
from core.networks.loss_functions import compute_angles, cross_entropy

import matplotlib.pyplot as plt
import pdb

def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conf_out(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.Sigmoid()
    )

class MVDNet_conf(nn.Module):
    def __init__(self, cfg):
        super(MVDNet_conf, self).__init__()
        self.cfg = cfg
        self.nlabel = cfg.ndepth
        self.mindepth = cfg.mindepth
        self.no_pool = False

        self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.softmax = nn.Softmax(dim = -1)

        self.wc0 = nn.Sequential(convbn_3d(64 + 3, 32, 3, 1, 1), nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        
        self.pool1 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)), nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)), nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)), nn.ReLU(inplace=True))
        
        self.n_convs0 = nn.Sequential(
            convtext(32, 96, 3, 1, 1),
            convtext(96, 96, 3, 1, 2),
            convtext(96, 96, 3, 1, 4),
            convtext(96, 64, 3, 1, 8),
            convtext(64, 64, 3, 1, 16)
        )
        self.n_convs1 = nn.Sequential(convtext(64, 32, 3, 1, 1), convtext(32, 3, 3, 1, 1))
        self.cconvs_fea = nn.Sequential(convtext(64, 32, 3, 1, 1), convtext(32, 32, 3, 1, 1))
        self.cconvs_depth = nn.Sequential(convtext(33, 16, 3, 1, 1), convtext(16, 16, 3, 1, 1))
        self.cconvs_prob = nn.Sequential(
            convtext(128, 64, 3, 1, 1),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 1, 1, 1)
        )
        self.cconvs_joint = nn.Sequential(
            convtext(49, 64, 3, 1, 1),
            convtext(64, 64, 3, 1, 2),
            convtext(64, 64, 3, 1, 4),
            convtext(64, 32, 3, 1, 1),
            conf_out(32, 1, 1, 1, 1)
        )

        self.cconvs_nfea = nn.Sequential(convtext(64, 32, 3, 1, 1), convtext(32, 32, 3, 1, 1))
        self.cconvs_normal = nn.Sequential(convtext(35, 16, 3, 1, 1), convtext(16, 16, 3, 1, 1))
        self.cconvs_njoint = nn.Sequential(
            convtext(48, 64, 3, 1, 1),
            convtext(64, 64, 3, 1, 2),
            convtext(64, 64, 3, 1, 4),
            convtext(64, 32, 3, 1, 1),
            conf_out(32, 1, 1, 1, 1)
        )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, target, refs, pose, intrinsics, intrinsics_inv, factor = None):
        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4
    
        tgtimg_fea = self.feature_extraction(target)

        _b,_ch,_h,_w = tgtimg_fea.size()
            
        disp2depth = Variable(torch.ones(_b, _h, _w)).cuda() * self.mindepth * self.nlabel
        disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(_b,self.nlabel,_h,_w)).type_as(disp2depth)

        depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
        if factor is not None:
            depth = depth*factor
        
        refimg_feas = []
        for j, ref in enumerate(refs):
            # build cost volume for each reference image
            cost = Variable(torch.FloatTensor(tgtimg_fea.size()[0], tgtimg_fea.size()[1]*2, self.nlabel,  tgtimg_fea.size()[2],  tgtimg_fea.size()[3]).zero_()).cuda()
            refimg_fea  = self.feature_extraction(ref)
            refimg_feas.append(refimg_fea)

            refimg_fea_warp = inverse_warp_d(refimg_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)

            cost[:, :refimg_fea_warp.size()[1],:,:,:] = tgtimg_fea.unsqueeze(2).expand(_b,_ch,self.nlabel,_h,_w)
            cost[:, refimg_fea_warp.size()[1]:,:,:,:] = refimg_fea_warp.squeeze(-1)
            
            cost = cost.contiguous()
            cost0 = self.dres0(cost)
            
            cost_in0 = cost0.clone()
            
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0 
            cost0 = self.dres3(cost0) + cost0 
            cost0 = self.dres4(cost0) + cost0
            
            cost_in0 = torch.cat((cost_in0, cost0.clone()), dim = 1)
            
            cost0 = self.classify(cost0)

            if j == 0:
                costs = cost0
                cost_in = cost_in0
            else:
                costs = costs + cost0
                cost_in = cost_in + cost_in0

        costs = costs / len(refs)

        # context convolution
        costs_context = Variable(torch.FloatTensor(tgtimg_fea.size()[0], 1, self.nlabel,  tgtimg_fea.size()[2],  tgtimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costs_context[:, :, i, :, :] = self.convs(torch.cat([tgtimg_fea, costt],1)) + costt

        # regress depth before and after context network
        costs_up = F.interpolate(costs, [self.nlabel,target.size()[2],target.size()[3]], mode='trilinear', align_corners = False)
        costs_up = torch.squeeze(costs_up,1)
        pred0 = F.softmax(costs_up,dim=1)
        pred0_r = pred0.clone()
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)

        costss_up = F.interpolate(costs_context, [self.nlabel,target.size()[2],target.size()[3]], mode='trilinear', align_corners = False)
        costss_up = torch.squeeze(costss_up,1)
        pred = F.softmax(costss_up,dim=1)
        softmax = pred.clone()
        pred = disparityregression(self.nlabel)(pred)
        depth1 = self.mindepth*self.nlabel/(pred.unsqueeze(1)+1e-16)

        # Warped feature, depth prediction, and probability distribution
        depth_down = F.interpolate(depth1, [tgtimg_fea.size()[2], tgtimg_fea.size()[3]], mode='bilinear', align_corners=False)
        # Detach the gradient
        depth_down = depth_down.detach()
        for j, ref in enumerate(refs):
            refimg_fea = refimg_feas[j]
            refimg_fea = inverse_warp(refimg_fea, depth_down.squeeze(1), pose[:,j], intrinsics4, intrinsics_inv4)
            concat_fea = torch.cat([tgtimg_fea, refimg_fea], 1)
            fea_conf = self.cconvs_fea(concat_fea)
            if j == 0:
                feas_conf = fea_conf.clone()
            else:
                feas_conf = feas_conf + fea_conf
        feas_conf = feas_conf / len(refs)

        # depth confidence networks
        depth_conf = self.cconvs_depth(torch.cat([depth_down, tgtimg_fea], 1))
        cost_cat = torch.cat([costs.squeeze(1), costs_context.squeeze(1)], 1)
        prob_conf = self.cconvs_prob(cost_cat)
        # Joint confidence fusion
        joint_conf = torch.cat([feas_conf, depth_conf, prob_conf], 1)
        joint_conf = self.cconvs_joint(joint_conf)
        joint_depth_conf = F.interpolate(joint_conf, [target.size()[2], target.size()[3]], mode='bilinear', align_corners=False)

        b,ch,d,h,w = cost_in.size()

        # normal network
        with torch.no_grad():
            intrinsics_inv[:,:2,:2] = intrinsics_inv[:,:2,:2] * (4)
            disp2depth = Variable(torch.ones(b, h, w).cuda() * self.mindepth * self.nlabel).cuda()
            disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(b,self.nlabel,h,w)).type_as(disp2depth)
            depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
            if factor is not None:
                depth = depth*factor  
            
            world_coord = pixel2cam(depth, intrinsics_inv)                
            world_coord = world_coord.squeeze(-1)

        if factor is not None:
            world_coord = world_coord / (2*self.nlabel*self.mindepth*factor.unsqueeze(-1))
        else:
            world_coord = world_coord / (2*self.nlabel*self.mindepth)
        
        world_coord = world_coord.clamp(-1,1)
        world_coord = torch.cat((world_coord.clone(), cost_in), dim = 1) #B,ch+3,D,H,W
        world_coord = world_coord.contiguous()
        
        if self.no_pool:
            wc0 = self.pool1(self.wc0(world_coord))
        else:
            wc0 = self.pool3(self.pool2(self.pool1(self.wc0(world_coord))))

        slices = []
        nmap = torch.zeros((b,3,h,w)).type_as(wc0)
        for i in range(wc0.size(2)):
            normal_fea = self.n_convs0(wc0[:,:,i])
            slices.append(self.n_convs1(normal_fea))
            if i == 0:
                nfea_conf = self.cconvs_nfea(normal_fea).clone()
            else:
                nfea_conf = nfea_conf + self.cconvs_nfea(normal_fea)
            nmap += slices[-1]        

        nmap_nor = F.normalize(nmap, dim=1)
        nmap_nor = nmap_nor.detach()
        nfea_conf = nfea_conf / wc0.size(2)
        # normal confidence network 
        normal_conf = self.cconvs_normal(torch.cat([nmap_nor, tgtimg_fea], 1))
        joint_normal_conf = self.cconvs_njoint(torch.cat([nfea_conf, normal_conf], 1))
        joint_normal_conf = F.interpolate(joint_normal_conf, [target.size()[2], target.size()[3]], mode='bilinear', align_corners=False)

        nmap_out = F.interpolate(nmap, [target.size(2), target.size(3)], mode = 'bilinear', align_corners = False)
        nmap_out = F.normalize(nmap_out,dim = 1)

        # add outputs
        return_vals = []
        depth0, depth1 = depth0.squeeze(1), depth1.squeeze(1)
        if self.training:
            return_vals += [depth0, depth1]
        else:
            return_vals += [depth1]
        
        joint_depth_conf, joint_normal_conf = joint_depth_conf.squeeze(1), joint_normal_conf.squeeze(1)
        return_vals += [nmap_out]
        return_vals += [joint_depth_conf, joint_normal_conf]
        return return_vals

